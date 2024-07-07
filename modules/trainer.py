import torch
from torch import optim
from tqdm import tqdm
from torch.nn import functional as F
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn import metrics

class Trainer():
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, args=None, logger=None):
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data
        self.model = model
        self.args = args
        self.train_num_steps = len(self.train_data) * args.num_epochs
        self.logger = logger
        self.refresh_nums = 2

    def train(self):
        self.before_train()

        self.model.train()
        self.logger.info("***** Running training *****")
        best_acc = 0.0
        best_f1 = 0.0

        pbar = tqdm(total=self.train_num_steps)
        avg_loss = 0
        for epoch in range(1, self.args.num_epochs + 1):
            step = 0
            pbar.set_description(f"Epoch{epoch}/{self.args.num_epochs + 1}")
            self.model.train()
            for batch in self.train_data:
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                outputs, labels = self._step(batch)
                _, preds = torch.max(outputs, 1)
                loss = F.cross_entropy(outputs, labels, reduction='mean')
            
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                avg_loss += loss.detach().cpu().item()

                step += 1
                if step % self.refresh_nums == 0:
                    pbar.set_postfix(loss=avg_loss/self.refresh_nums)
                    avg_loss = 0
                    pbar.update(self.refresh_nums)

            dev_loss, dev_acc, dev_f1, _, _ = self.evaluate(self.model, self.dev_data)
            self.logger.info("Current Dev acc is %s", str(dev_acc))
            self.logger.info("Current Dev f1 is %s", str(dev_f1))

            if dev_acc > best_acc:
                torch.save(self.model.state_dict(), "./best_model.pth")
                best_acc = dev_acc
            if dev_f1 > best_f1:
                best_f1 = dev_f1

        self.logger.info("The best Dev acc is %s", str(best_acc))
        self.logger.info("The best Dev f1 is %s", str(best_f1))

        self.model.load_state_dict(torch.load("./best_model.pth"))
        self.model.eval()
        test_loss, test_acc, test_f1, _, _ = self.evaluate(self.model, self.test_data)
        self.logger.info("Current Test acc is %s", str(test_acc))
        self.logger.info("Current Test f1 is %s", str(test_f1))

    def evaluate(self, model, loader):
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader):
                batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                outputs, labels = self._step(batch)

                preds = torch.argmax(outputs.data, 1)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

            acc = metrics.accuracy_score(all_labels, all_preds)
            f1 = metrics.f1_score(all_labels, all_preds, average='weighted')

        return test_loss / len(loader), acc, f1, all_preds, all_labels

    def before_train(self):
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


    def _step(self, batch):
        input_ids, attention_mask_bert, token_type_ids, image, attention_mask_pixel, images, labels = batch
        outputs = self.model(input_ids, attention_mask_bert, token_type_ids, image, attention_mask_pixel, images)

        return outputs, labels


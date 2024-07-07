from modules import BertPixelSubImagesDataset
from modules import Trainer
from model import BertPixelSubimagesConpromptsShortTripleModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from pixel.data import PyGameTextRenderer
from pixel.utils import resize_model_embeddings
import pandas as pd
import numpy as np
import os
import argparse
import logging
import torch
import random
# import debugpy
# debugpy.connect(('localhost', 9249))

logging.basicConfig(format = '%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
fileHandler = logging.FileHandler('./logs/log_model.txt', mode='a', encoding='utf8')
file_format = logging.Formatter('%(asctime)s - %(levelname)s -   %(message)s')
fileHandler.setFormatter(file_format)
logger = logging.getLogger(__name__)
logger.addHandler(fileHandler)

def set_seed(seed=1234):
    """set random seed"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--note',
                        type=str,
                        default='Use BertPixelSubimagesConpromptsShortModel')
    parser.add_argument('--data_path',
                        type=str,
                        default='./data/mvsa_single/')
    parser.add_argument('--text_name_or_path',
                        type=str,
                        default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--pixel_name_or_path',
                        type=str,
                        default='../pretrained_models/pixel-base')
    parser.add_argument('--vision_name_or_path',
                        type=str,
                        default='../pretrained_models/resnet50.pth')
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help="cuda or cpu")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int)
    # parser.add_argument("--learnable",
    #                     default="1",
    #                     type=int)
    parser.add_argument("--scale",
                        default="0.1",
                        type=float)
    parser.add_argument("--lr",
                        default=5e-5,
                        type=float)
    parser.add_argument("--num_epochs",
                        default=20,
                        type=int)
    parser.add_argument("--seed",
                        default=1234,
                        type=int)
    parser.add_argument('--warmup_ratio',
                        default=0.01,
                        type=float)
    parser.add_argument("--mvsa_single_sub_imgaes",
                        default="./features/mvsa-single_sub_images.pth")
    # parser.add_argument('--w2v_file',
    #                     default='../pretrained_models/glove.6B/glove.6B.300d.txt')
    # parser.add_argument('--vocab_size',
    #                     default=100000,
    #                     type=int)

    args = parser.parse_args()
    for k,v in vars(args).items():
        logger.info(" " + str(k) +" = %s", str(v))
    
    set_seed(args.seed)

    tr_ids = pd.read_csv(args.data_path + 'splits/train.txt', header=None).to_numpy().flatten()
    val_ids = pd.read_csv(args.data_path + 'splits/val.txt', header=None).to_numpy().flatten()
    te_ids = pd.read_csv(args.data_path + 'splits/test.txt', header=None).to_numpy().flatten()
    pair_df = pd.read_csv(args.data_path + 'valid_pairlist.txt', header=None)

    all_labels = pair_df[1].to_numpy().flatten()
    lab_train = all_labels[tr_ids]
    lab_val = all_labels[val_ids]
    lab_test = all_labels[te_ids]

    args.text_name_or_path = '../pretrained_models/bert-base-uncased'
    args.pixel_name_or_path = '../pretrained_models/pixel-base'
    args.vision_name_or_path = '../pretrained_models/resnet50.pth'
    tokenizer = BertTokenizer.from_pretrained(args.text_name_or_path)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    processor = PyGameTextRenderer.from_pretrained(args.pixel_name_or_path,rgb=False,revision="main")
    transform_pixel = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(size=(processor.pixels_per_patch,processor.pixels_per_patch * args.max_seq_length))
    ])
    train_dataset = BertPixelSubImagesDataset(args, os.path.join(args.data_path, "valid_pairlist.txt"), \
        tokenizer, processor, transform_pixel, img_transforms, lab_train, tr_ids)
    train_dataloder = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dataset = BertPixelSubImagesDataset(args, os.path.join(args.data_path, "valid_pairlist.txt"), \
         tokenizer, processor, transform_pixel, img_transforms, lab_val, val_ids)
    val_dataloder = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    test_dataset = BertPixelSubImagesDataset(args, os.path.join(args.data_path, "valid_pairlist.txt"), \
        tokenizer,processor, transform_pixel, img_transforms, lab_test, te_ids)
    test_dataloder = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = BertPixelSubimagesConpromptsShortTripleModel(args)
    resize_model_embeddings(model, args.max_seq_length)
    trainer = Trainer(train_dataloder, val_dataloder, test_dataloder, model, args, logger)
    trainer.train()



if __name__ == '__main__':
    main()

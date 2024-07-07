import torchvision
import torch
import math
from torch import nn
from transformers import BertModel
from pixel.models import ViTModel, AutoConfig

# Multi-head Cross-attention
class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(CrossAttentionLayer, self).__init__()
        self.config = config
        self.num_attention_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.attention_head_size = int (self.hidden_size / self.num_attention_heads)

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)

    # (bsz, seq_len, hidden_size) -> (bsz, num_heads, seq_len, head_size)
    def transpose_for_scores(self, x:torch.Tensor):
        # (bsz, seq_len, hidden_size) -> (bsz, seq_len, num_heads, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        # (bsz, seq_len, num_heads, head_size) -> (bsz, num_heads, seq_len, head_size)
        return x.permute(0,2,1,3)
    
    def forward(self, x1, x2, x2_attention:torch.Tensor):
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # (bsz, num_heads, seq_len1, head_size) * (bsz, num_heads, head_size, seq_len2)
        # -> (bsz, num_heads, seq_len1, seq_len2)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # (bsz, seq_len2) -> (bsz, 1, 1, seq_len2) -> (bsz, num_heads, seq_len1, seq_len2)
        masked_scores = x2_attention.unsqueeze(1).unsqueeze(2)
        masked_scores = (1.0 - masked_scores) * -10000.0
        attention_scores = attention_scores + masked_scores

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # (bsz, num_heads, seq_len1, seq_len2) * (bsz, num_heads, seq_len2, head_size)
        # (bsz, num_heads, seq_len1, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (bsz, num_heads, seq_len1, head_size) -> (bsz, seq_len1, num_heads, head_size)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # (bsz, seq_len1, num_heads, head_size) -> (bsz, seq_len1, hidden_size)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer

class BertPixelSubimagesConpromptsShortTripleModel(torch.nn.Module):
    def __init__(self, args):
        super(BertPixelSubimagesConpromptsShortTripleModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.text_name_or_path)
        self.config = AutoConfig.from_pretrained(args.pixel_name_or_path)
        self.n_layer = self.bert.config.num_hidden_layers
        self.n_head = self.bert.config.num_attention_heads
        self.n_embd = self.bert.config.hidden_size // self.bert.config.num_attention_heads
        self.vit = ViTModel.from_pretrained(args.pixel_name_or_path, config=self.config)
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.resnet.load_state_dict(torch.load(args.vision_name_or_path))
        self.encoder_conv_resnet = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert.config.hidden_size)
        )
        # self.encoder_conv_pixel = nn.Sequential(
        #     nn.Linear(in_features=768, out_features=1024),
        #     nn.Tanh(),
        #     nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert.config.hidden_size)
        # )
        self.cross_attention_layer1 = CrossAttentionLayer(self.bert.config)
        self.cross_attention_layer2 = CrossAttentionLayer(self.bert.config)
        self.cross_attention_layer3 = CrossAttentionLayer(self.bert.config)
        
        self.projection1 = torch.nn.Linear(768, 768)
        self.projection2 = torch.nn.Linear(768, 768)
        self.projection3 = torch.nn.Linear(768 * 2, 768)
        self.tanh = torch.nn.Tanh()
        
        self.cf = torch.nn.Linear(768+768+768, 3)

    def get_prompt(self, resnet_features=None):
        bsz, resnet_len, _ = resnet_features.size()
        past_key_values_resnet = self.encoder_conv_resnet(resnet_features)
        sentiment_prompt = past_key_values_resnet
        past_key_values_resnet = past_key_values_resnet.view(
            bsz,
            resnet_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        # (bsz, seq_len + seq_len2, 12*2, n_head(12), 64)
        past_key_values = past_key_values_resnet
        # (12*2, bsz, n_head, len, 64)
        # 12*[2,bsz,n_head,len,64]
        past_key_values = past_key_values.permute([2,0,3,1,4]).split(2)

        return past_key_values, sentiment_prompt


    def forward(self, input_ids, attention_mask_bert, token_type_ids, image, attention_mask_pixel, images:torch.Tensor):
        pixel_result = self.vit(image, attention_mask_pixel)
        # (bsz, len, 768)
        output_pixel = pixel_result[0]
        # (bsz, 4, 3, 224, 224) -> (4, bsz, 3, 224, 224)
        images = images.permute((1,0,2,3,4))
        image_outputs = []
        for image in images:
            for name, layer in self.resnet.named_children():
                if 'fc' in name:
                    break
                image = layer(image)
            image_outputs.append(image.squeeze(-1).squeeze(-1))
        
        # [4, (bsz,2048)] -> (4, bsz, 2048) -> (bsz, 4, 2048)
        image_outputs = torch.stack(image_outputs, dim=0).permute(1,0,2)
        # sentiment_prompt (bsz, 4, layer_num, 2*hidden_size)
        past_key_values, sentiment_prompt = self.get_prompt(resnet_features=image_outputs)
        prompt_guids_length = past_key_values[0][0].size(2)
        bsz = attention_mask_bert.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(attention_mask_bert.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask_bert), dim=1)

        output_bert_result = self.bert(input_ids=input_ids, 
                           attention_mask=prompt_attention_mask, 
                           token_type_ids=token_type_ids,
                           past_key_values=past_key_values)
        output_bert = output_bert_result[0]
        
        bsz = sentiment_prompt.shape[0]
        # (bsz, 48, 768*2)
        sentiment_prompt = sentiment_prompt.reshape(bsz, 48, -1)
        output_image = self.projection3(sentiment_prompt)
        
        output_image_pixel = torch.cat([output_image, output_pixel], dim=1)
        pixel_image_mask = torch.cat([torch.ones(output_pixel.shape[0], 48+1).to(attention_mask_bert.device), attention_mask_pixel], dim=1)
        bert_attention_output = self.cross_attention_layer1(output_bert, output_image_pixel, pixel_image_mask)
        output_image_bert = torch.cat([output_image, output_bert], dim=1)
        image_bert_mask = torch.cat([torch.ones(output_pixel.shape[0], 48).to(attention_mask_bert.device), attention_mask_bert], dim=1)
        pixel_attention_output = self.cross_attention_layer2(output_pixel, output_image_bert, image_bert_mask)
        output_bert_pixel = torch.cat([output_bert, output_pixel], dim=1)
        bert_pixel_mask = torch.cat([attention_mask_bert, torch.ones(output_pixel.shape[0], 1).to(attention_mask_bert.device), attention_mask_pixel], dim=1)
        image_attention_output = self.cross_attention_layer3(output_image, output_bert_pixel, bert_pixel_mask)
        
        bert_attention_output = bert_attention_output[:, 0]
        bert_pooler_output = self.tanh(self.projection1(bert_attention_output))
        pixel_attention_output = pixel_attention_output[:, 0]
        pixel_pooler_output = self.tanh(self.projection2(pixel_attention_output))
        # (bsz, 768)
        image_pooler_output = torch.mean(image_attention_output, dim=1)
        
        final_result = torch.cat([bert_pooler_output, pixel_pooler_output, image_pooler_output], dim=-1)
        
        return self.cf(final_result)

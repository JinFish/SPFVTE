import torch
from torch.utils.data import Dataset
from PIL import Image
from pixel.utils import get_attention_mask
import pandas as pd
import os
import numpy as np

class BertPixelSubImagesDataset(Dataset):
    def __init__(self, args, file_name, tokenizer, processor, transform_pixel, transform_img, labels, ids):
        self.max_seq_length = args.max_seq_length
        self.data_dir = args.data_path
        self.all_ids = pd.read_csv(file_name, header=None)[0].to_numpy().flatten()[ids]
        self.tokenizer = tokenizer
        self.processor = processor
        self.transform_pixel = transform_pixel
        self.transform_img = transform_img
        self.labels = np.array(labels).astype(np.int64)
        self.sub_images = torch.load(args.mvsa_single_sub_imgaes)

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        label = self.labels[idx]
        fid = self.all_ids[idx]
        text = open(os.path.join(self.data_dir, 'texts', str(fid) + '.txt'), 'r', encoding='utf-8',
                    errors='ignore').read().strip().lower()
        encode_dict = self.tokenizer.encode_plus(text, max_length=self.max_seq_length,
                                                 truncation=True, padding='max_length')
        input_ids = encode_dict["input_ids"]
        attention_mask_bert = encode_dict["attention_mask"]
        token_type_ids = encode_dict['token_type_ids']

        encodings = self.processor(text=text)
        image = Image.fromarray(encodings.pixel_values).convert("RGB")
        image = self.transform_pixel(image)
        attention_mask_pixel = get_attention_mask(encodings.num_text_patches, seq_length=self.max_seq_length)

        main_image = Image.open(os.path.join(self.data_dir, "images", str(fid) + '.jpg'))
        main_image = self.transform_img(main_image).unsqueeze(0)
        
        sub_images_key = str(fid) + '.jpg'
        sub_images = torch.zeros((3,3,224,224))
        if sub_images_key in self.sub_images:
            sub_image_names = self.sub_images[str(fid) + '.jpg']
            
            for i, sub_image_name in enumerate(sub_image_names):
                sub_image = Image.open(os.path.join(self.data_dir, "sub_images", str(sub_image_name)))
                sub_image = self.transform_img(sub_image)
                sub_images[i] = sub_image
        images = torch.cat([main_image, sub_images])

        return torch.tensor(input_ids, dtype=torch.int64), torch.tensor(attention_mask_bert, dtype=torch.int64), \
            torch.tensor(token_type_ids, dtype=torch.int64), image, attention_mask_pixel, images, torch.tensor(label)
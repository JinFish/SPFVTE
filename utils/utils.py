import os
import shutil
import random
from PIL import Image

# Processing of MVSA-single
def Processing_single(data_dir):
    rel_to_id = {"neutral":'0', "positive":'1', "negative":'2'}

    error = []
    if not os.path.exists('../data'):
        os.mkdir('../data')
    if not os.path.exists('../data/mvsa_single'):
        os.mkdir('../data/mvsa_single')

    fw = open('../data/mvsa_single/valid_pairlist.txt', 'w', encoding='utf8')
    with open(os.path.join(data_dir, "labelResultAll.txt"), 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if "ID" in line:
                continue
            line = line.strip()
            id, sentiments = line.split("\t")
            text_sent, image_sent = sentiments.split(",")

            # If the text and image have the same label, both can be the final label
            if text_sent == image_sent:
                fw.write(id + "," + rel_to_id[text_sent] + "," + rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")
            # If the text label is postivie
            elif text_sent == "positive":
                # If the image label is neutral, the text label is the final label
                if image_sent == "neutral":
                    fw.write(id + "," + rel_to_id[text_sent] + "," +  rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")
                # If the image label is negative, drop the data
                elif image_sent == "negative":
                    error.append(id)
            # If the text label is neutral, the image label is the finale label
            elif text_sent == "neutral":
                fw.write(id + "," + rel_to_id[image_sent] + "," + rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")
            # If the text label is negative
            elif text_sent == "negative":
                # If the image label is postivie, drop the data
                if image_sent == "positive":
                    error.append(id)
                # else, the text label is the final label
                elif image_sent == "neutral":
                    fw.write(id + "," + rel_to_id[text_sent] + "," + rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")

    print("the conflict num:", len(error))

# Processing of MVSA-multi
def Processing_multi(data_dir):
    rel_to_id = {"neutral":'0', "positive":'1', "negative":'2'}

    error = []
    if not os.path.exists('../data'):
        os.mkdir('../data')
    if not os.path.exists('../data/mvsa_multiple'):
        os.mkdir('../data/mvsa_multiple')

    fw = open('../data/mvsa_multiple/valid_pairlist.txt', 'w', encoding='utf8')
    with open(os.path.join(data_dir, "labelResultAll.txt"), 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if "ID" in line:
                continue
            line = line.strip()
            id, sentiments1, sentiments2, sentiments3 = line.split("\t")
            if os.stat(os.path.join(data_dir, "data", id+".txt")).st_size < 1:
                continue
            if os.stat(os.path.join(data_dir, "data", id+".jpg")).st_size < 1:
                continue
            try:
                image = Image.open(os.path.join(data_dir, "data", id+".jpg")).convert('RGB')
            except:
                continue

            text_sent1, image_sent1 = sentiments1.split(",")
            text_sent2, image_sent2 = sentiments2.split(",")
            text_sent3, image_sent3 = sentiments3.split(",")

            sent_dict = {"positive":0, "negative":0, "neutral":0}
            sent_dict[text_sent1] += 1
            sent_dict[text_sent2] += 1
            sent_dict[text_sent3] += 1

            text_sent = ""

            for k,v in sent_dict.items():
                if v >=2:
                    text_sent = k

            if text_sent == "":
                continue

            sent_dict = {"positive": 0, "negative": 0, "neutral": 0}
            sent_dict[image_sent1] += 1
            sent_dict[image_sent2] += 1
            sent_dict[image_sent3] += 1

            image_sent = ""

            for k, v in sent_dict.items():
                if v >= 2:
                    image_sent = k
            if image_sent == "":
                continue

            # If the text and image have the same label, both can be the final label
            if text_sent == image_sent:
                fw.write(id + "," + rel_to_id[text_sent] + "," + rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")
            # If the text label is postivie
            elif text_sent == "positive":
                # If the image label is neutral, the text label is the final label
                if image_sent == "neutral":
                    fw.write(id + "," + rel_to_id[text_sent] + "," +  rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")
                # If the image label is negative, drop the data
                elif image_sent == "negative":
                    error.append(id)
            # If the text label is neutral, the image label is the finale label
            elif text_sent == "neutral":
                fw.write(id + "," + rel_to_id[image_sent] + "," + rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")
            # If the text label is negative
            elif text_sent == "negative":
                # If the image label is postivie, drop the data
                if image_sent == "positive":
                    error.append(id)
                # else, the text label is the final label
                elif image_sent == "neutral":
                    fw.write(id + "," + rel_to_id[text_sent] + "," + rel_to_id[text_sent] + "," + rel_to_id[image_sent] + "\n")

    print("the conflict num:", len(error))

# Split MVSA-single
def Split_single():
    positive_sum = 0
    negative_sum = 0
    neutral_sum = 0
    with open('../data/mvsa_single/valid_pairlist.txt', 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            id, label, _, __ = line.strip().split(",")
            if label== '1':
                positive_sum += 1
            elif label=='2':
                negative_sum += 1
            else:
                neutral_sum += 1

    train_positive = round(positive_sum * 0.8)
    val_positive = round(positive_sum * 0.1)
    test_positive = positive_sum - train_positive -val_positive

    train_negative = round(negative_sum * 0.8)
    val_negative = round(negative_sum * 0.1)
    test_negative = negative_sum - train_negative - val_negative

    train_neutral = round(neutral_sum * 0.8)
    val_neutral = round(neutral_sum * 0.1)
    test_neutral = neutral_sum - train_neutral - val_neutral

    if not os.path.exists("../data/mvsa_single/splits"):
        os.mkdir("../data/mvsa_single/splits")
    f_list = []
    f_train = open('../data/mvsa_single/splits/train.txt', 'w', encoding='utf8')
    f_val = open('../data/mvsa_single/splits/val.txt', 'w', encoding='utf8')
    f_test = open('../data/mvsa_single/splits/test.txt', 'w', encoding='utf8')

    with open('../data/mvsa_single/valid_pairlist.txt', 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            f_list.append((i, line))
    random.shuffle(f_list)

    for i, tpl in enumerate(f_list):
        idx = tpl[0]
        line = tpl[1]
        id, label, _, __ = line.strip().split(",")
        if label == '1':
            if train_positive > 0:
                f_train.write(str(idx) + "\n")
                train_positive -= 1
            elif val_positive > 0:
                f_val.write(str(idx) + "\n")
                val_positive -= 1
            elif test_positive > 0:
                f_test.write(str(idx) + "\n")
                test_positive -= 1
        elif label == '2':
            if train_negative > 0:
                f_train.write(str(idx) + "\n")
                train_negative -= 1
            elif val_negative > 0:
                f_val.write(str(idx) + "\n")
                val_negative -= 1
            elif test_negative > 0:
                f_test.write(str(idx) + "\n")
                test_negative -= 1
        else:
            if train_neutral > 0:
                f_train.write(str(idx) + "\n")
                train_neutral -= 1
            elif val_neutral > 0:
                f_val.write(str(idx) + "\n")
                val_neutral -= 1
            elif test_neutral > 0:
                f_test.write(str(idx) + "\n")
                test_neutral -= 1

# Split MVSA-multiple
def Split_multiple():
    positive_sum = 0
    negative_sum = 0
    neutral_sum = 0
    with open('../data/mvsa_multiple/valid_pairlist.txt', 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            id, label, _, __ = line.strip().split(",")
            if label== '1':
                positive_sum += 1
            elif label=='2':
                negative_sum += 1
            else:
                neutral_sum += 1

    train_positive = round(positive_sum * 0.8)
    val_positive = round(positive_sum * 0.1)
    test_positive = positive_sum - train_positive -val_positive

    train_negative = round(negative_sum * 0.8)
    val_negative = round(negative_sum * 0.1)
    test_negative = negative_sum - train_negative - val_negative

    train_neutral = round(neutral_sum * 0.8)
    val_neutral = round(neutral_sum * 0.1)
    test_neutral = neutral_sum - train_neutral - val_neutral

    if not os.path.exists("../data/mvsa_multiple/splits"):
        os.mkdir("../data/mvsa_multiple/splits")
    f_list = []
    f_train = open('../data/mvsa_multiple/splits/train.txt', 'w', encoding='utf8')
    f_val = open('../data/mvsa_multiple/splits/val.txt', 'w', encoding='utf8')
    f_test = open('../data/mvsa_multiple/splits/test.txt', 'w', encoding='utf8')

    with open('../data/mvsa_multiple/valid_pairlist.txt', 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            f_list.append((i, line))

    random.shuffle(f_list)

    for i, tpl in enumerate(f_list):
        idx = tpl[0]
        line = tpl[1]
        id, label, _, __ = line.strip().split(",")
        if label == '1':
            if train_positive > 0:
                f_train.write(str(idx) + "\n")
                train_positive -= 1
            elif val_positive > 0:
                f_val.write(str(idx) + "\n")
                val_positive -= 1
            elif test_positive > 0:
                f_test.write(str(idx) + "\n")
                test_positive -= 1
        elif label == '2':
            if train_negative > 0:
                f_train.write(str(idx) + "\n")
                train_negative -= 1
            elif val_negative > 0:
                f_val.write(str(idx) + "\n")
                val_negative -= 1
            elif test_negative > 0:
                f_test.write(str(idx) + "\n")
                test_negative -= 1
        else:
            if train_neutral > 0:
                f_train.write(str(idx) + "\n")
                train_neutral -= 1
            elif val_neutral > 0:
                f_val.write(str(idx) + "\n")
                val_neutral -= 1
            elif test_neutral > 0:
                f_test.write(str(idx) + "\n")
                test_neutral -= 1

# Store text and images in separate folders for MVSA-single
def mv_dataset_single(data_dir):
    if not os.path.exists("../data/mvsa_single/images"):
        os.mkdir("../data/mvsa_single/images")
    if not os.path.exists("../data/mvsa_single/texts"):
        os.mkdir("../data/mvsa_single/texts")

    file_list = os.listdir(data_dir)
    for i, file in enumerate(file_list):
        if "txt" in file:
            shutil.move(os.path.join(data_dir, file), os.path.join("../data/mvsa_single/texts", file))
        elif "jpg" in file:
            shutil.move(os.path.join(data_dir, file), os.path.join("../data/mvsa_single/images", file))

# Store text and images in separate folders for MVSA-multiple
def mv_dataset_multiple(data_dir):
    if not os.path.exists("../data/mvsa_multiple/images"):
        os.mkdir("../data/mvsa_multiple/images")
    if not os.path.exists("../data/mvsa_multiple/texts"):
        os.mkdir("../data/mvsa_multiple/texts")

    file_list = os.listdir(data_dir)
    for i, file in enumerate(file_list):
        if "txt" in file:
            shutil.move(os.path.join(data_dir, file), os.path.join("../data/mvsa_multiple/texts", file))
        elif "jpg" in file:
            shutil.move(os.path.join(data_dir, file), os.path.join("../data/mvsa_multiple/images", file))



if __name__ == '__main__':
    Processing_single("../datasets/MVSA_Single")
    Processing_multi("../datasets/MVSA-multiple")
    Split_single()
    Split_multiple()
    mv_dataset_single("../datasets/MVSA_Single/data")
    mv_dataset_multiple("../datasets/MVSA-multiple/data")
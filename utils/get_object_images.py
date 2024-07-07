import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm

def get_sub_images():
    image_list = os.listdir("./data/mvsa_single/images")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load('../pretrained_models/fasterrcnn_resnet50_fpn_coco.pth'))
    model = model.eval()
    model = model.to("cuda")

    img_dict = {}
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    pbar = tqdm(total=len(image_list))
    pbar.set_description("Iter...")
    error_cnt = 0
    for i, image_name in enumerate(image_list):
        try:
            image_pil = Image.open(os.path.join("./data/mvsa_single/images", str(image_name)))
            image:torch.Tensor = trans(image_pil).unsqueeze(0)
            image = image.to("cuda")
            result = model(image)
            boxes = result[0]["boxes"]
            scores = result[0]["scores"]
            cnt = 0
            for b, s in zip(boxes, scores):
                # b->(x0,y0,x1,y1)
                b = b.detach().cpu().numpy()
                s = s.detach().cpu().numpy()
                # keep P > 0.5 and top 3
                if s < 0.5 or cnt >= 3:
                    break

                sub_img_path = f"./data/mvsa_single/sub_images/{image_name.split('.')[0]}_{str(cnt)}.jpg"
                image_pil.crop(b).save(sub_img_path)
                if image_name not in img_dict:
                    img_dict[image_name] = []
                    img_dict[image_name].append(f"{image_name.split('.')[0]}_{str(cnt)}.jpg")
                else:
                    img_dict[image_name].append(f"{image_name.split('.')[0]}_{str(cnt)}.jpg")
                cnt += 1
            pbar.update(1)
        except:
            error_cnt += 1
            continue
        
    
    pbar.close()
    torch.save(img_dict, './features/mvsa-single_sub_images.pth')
    print("error nums:", error_cnt)


if __name__ == "__main__":
    get_sub_images()
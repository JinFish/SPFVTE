# SPFVTE
This is the repository of *Sentimental Prompt Framework with Visual Text Encoder for Multimodal Sentiment Analysis* (ICMR 2024)

![image](https://github.com/JinFish/SPFVTE/assets/42674937/29c6d923-2390-4995-a9df-ee83f30363f0)

## Enviroment
We recommend the following actions to create the environment:
```bash
conda create -n  SPFVTE python==3.8.16
conda activate SPFVTE
```

and you should install the required packages in `PIXEL`: see `Setup` section in https://github.com/xplip/pixel. (especially pycairo, pygobject, and manimpango should be used by conda.)

The `pixel` folder in this repository comes from https://github.com/xplip/pixel which has undergone some modifications, if you want to know the original code, please refer to the above link.

## Dataset
For  **MSVA-S** and **MVSA-M**, you can download from http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/, and use `utils/utils.py` to process it:

```python
cd utils
python utils.py
```

For **HFM**, see in https://github.com/Link-Li/CLMLF.

**Note that:** we put the raw datasets into the `datasets` folder, and the processed datasets will be in the `data` folder.


## Required pre-trained models
In this paper, we use `BERT`, `FasterRcnn`, `ResNet` and `PIXEL` as our text encoder, object detection model, visual enoder and visual text encoder, repectively.
For the code implementation, we utilized the models and weights provided by Hugging Face and PyTorch.

You can download the model weights from https://huggingface.co/google-bert/bert-base-uncased, https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth, https://download.pytorch.org/models/resnet50-19c8e357.pth and https://huggingface.co/spaces/Team-PIXEL/PIXEL.



## Running
After you prepare the models, you should first run `utils/get_object_images.py` to get the image regions.
And you can run `python run.py` to train a model.
import torch
import torch.nn as nn
import torchvision.transforms.functional as TTF
import torchvision.transforms as transforms
import random
import copy
from PIL import Image

class RandomHorizontalFlip(nn.Module):
    def __init__(self,p:float=0.5):
        super(RandomHorizontalFlip,self).__init__()
        self.prob = p
        random.seed(777)

    def forward(self,pil_img,bbox):
        prob = random.random()

        if prob < self.prob:
            return pil_img,bbox

        else:
            w,h = pil_img.size
            r_bbox = copy.deepcopy(bbox)
            r_bbox[0] = w - bbox[2]
            r_bbox[2] = w - bbox[0]
            img = pil_img.copy()

            return img.transpose(Image.FLIP_LEFT_RIGHT),r_bbox

class RandomVerticalFlip(nn.Module):
    def __init__(self,p:float=0.5):
        super(RandomVerticalFlip,self).__init__()
        self.prob = p
        random.seed(685)

    def forward(self,pil_img,bbox):
        prob = random.random()

        if prob < self.prob:
            return pil_img,bbox

        else:
            w,h = pil_img.size
            r_bbox = copy.deepcopy(bbox)
            r_bbox[1] = h - bbox[3]
            r_bbox[3] = h - bbox[1]
            img = pil_img.copy()

            return img.transpose(Image.FLIP_TOP_BOTTOM),r_bbox

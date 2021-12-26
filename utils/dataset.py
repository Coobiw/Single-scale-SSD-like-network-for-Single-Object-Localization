import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import torch as t
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.anchor_tool import anchor_generate
from utils.anchor_tool import kms_result_anchor
from utils.anchor_tool import bbox_encode
from utils.anchor_tool import label_assignment

class tiny_dataset(Dataset):
    def __init__(self,train_num_per_class:int = 150,val_num_per_class:int = 30,
                 augmentation :bool = False,
                 root:str = r'E:\BS_learning\4_1\CV_basis\experiment\SSD-like method\tiny_vid'):
        self.root = root
        data_dir = []
        label_dir = []
        for each in os.listdir(self.root):
            if '.txt' in each:
                label_dir.append(each)
            elif '.md' not in each:
                data_dir.append(each)
        self.imgs = read_img(root=self.root,data_dir=data_dir)
        self.labels = get_label() # （900，）的float64的Tensor
        self.bboxs,self.bbox_hw = read_bbox(root=self.root,label_dir=label_dir)
        # （900，4）的float64的Tensor

        self.img_size = self.imgs[0].size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]
        )

        return {'img':img_transform(self.imgs[index]),
                'label':self.labels[index],'bbox':self.bboxs[index]}

def read_img(root,data_dir):
    img_path = []
    for each in data_dir:
        for img in os.listdir(os.path.join(root, each))[:180]:
            img_path.append(os.path.join(os.path.join(root, each), img))
    imgs = [Image.open(i) for i in img_path]
    return imgs

def get_label():
    l1 = np.zeros((180,))
    l2 = np.ones((180,))
    l3 = l2.copy() * 2
    l4 = l2.copy() * 3
    l5 = l2.copy() * 4

    labels = np.concatenate((l1, l2, l3, l4, l5))
    # print(labels)
    labels = t.from_numpy(labels)

    return labels

def read_bbox(root,label_dir):
    bbox_path = []
    for each in label_dir:
        bbox_path.append((os.path.join(root, each)))

    for each in bbox_path:
        if each == bbox_path[0]:
            bbox_data = pd.read_csv(each, sep=' ', header=None, index_col=None)
            bbox_data[0] -= 1
            bbox_data = bbox_data.iloc[:180]
            bbox_data = bbox_data.drop(0, axis=1)
            bbox_data = bbox_data.values
        else:
            data = pd.read_csv(each, sep=' ', header=None, index_col=None)
            data[0] -= 1
            data = data.iloc[:180]
            data = data.drop(0, axis=1)
            data = data.values
            bbox_data = np.concatenate((bbox_data, data), axis=0)

    bbox_w = bbox_data[:, 2] - bbox_data[:, 0]
    bbox_h = bbox_data[:, 3] - bbox_data[:, 1]
    bbox_hw = np.stack((bbox_h, bbox_w), axis=1)
    bbox_data = t.from_numpy(bbox_data)

    return bbox_data,bbox_hw

if __name__ == '__main__':
    dataset = tiny_dataset()
    item = dataset[188]
    img = item['img']
    bbox = item['bbox'].data.numpy()
    # print(bbox.shape)
    # print(bbox)
    img = img.numpy().transpose(1,2,0)
    img = img *[0.229,0.224,0.225] + [0.485,0.456,0.406]
    img = img*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    k_selected = 3

    anchors = anchor_generate(kms_anchor=kms_result_anchor(dataset.bbox_hw, k_selected))

    h,w = img.shape[0],img.shape[1]
    assert h==w,'please let the input image be a square'

    anchors = anchors.numpy() * h
    anchors_xyxy  = np.zeros(anchors.shape,dtype=anchors.dtype)
    anchors_xyxy[:, 0] = anchors[:, 0] - 0.5 * anchors[:, 2]
    anchors_xyxy[:, 1] = anchors[:, 1] - 0.5 * anchors[:, 3]
    anchors_xyxy[:, 2] = anchors[:, 0] + 0.5 * anchors[:, 2]
    anchors_xyxy[:, 3] = anchors[:, 1] + 0.5 * anchors[:, 3]

    # print(anchors_xyxy)
    anchors_xyxy += 128
    anchors_xyxy = anchors_xyxy.astype('int32')
    padding_img = np.zeros((128*3,128*3,3),dtype=img.dtype)
    padding_img[128:256,128:256,:] = img

    colors = [[0,0,255],[0,255,0],[255,0,0]]
    for i,anchor in enumerate(anchors_xyxy):
        padding_img = cv2.rectangle(padding_img,(anchor[0],anchor[1]),(anchor[2],anchor[3]),
                                    color = colors[i%3])

    img_bbox = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[0, 0, 255])

    cv2.namedWindow('anchors', 0)
    cv2.imshow('anchors', padding_img)
    cv2.resizeWindow('anchors', 640, 480)
    cv2.namedWindow('demo',0)
    cv2.imshow('demo',img_bbox)
    cv2.resizeWindow('demo', 640, 480)
    cv2.waitKey(0)
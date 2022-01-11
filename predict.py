from PIL import Image
import torch
import pickle
import os
import argparse
import torchvision.transforms as transforms
import numpy as np
from utils.anchor_tool import anchor_generate,offset_decode
import cv2

basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ]
        )

classes_list = ['background','bird','car','dog','lizard','turtle']
colors_dict = {'background':[0,0,0],'bird':[0,0,255],'car':[0,255,0],'dog':[255,0,0],
              'lizard':[255,0,255],'turtle':[0,255,255]}

def capture_img(img_path):
    return Image.open(img_path)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', help='the path pf .pth file of model', type=str,
                        required=True, dest='load_model')
    parser.add_argument('--device', type=str,default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--kms-anchors-path', type=str, required=True)

    return parser

def get_anchor(kms_anchor_path):
    with open(kms_anchor_path,'rb') as af:
        kms_anchors = pickle.load(af)

    anchors = anchor_generate(kms_anchor=kms_anchors)

    return anchors

def predict(img_path):
    parser = arg_parse()
    args = parser.parse_args()

    device = torch.device(args.device)
    anchors = get_anchor(args.kms_anchors_path).to(device)
    net = torch.load(args.load_model).to(device)

    pil_img = capture_img(img_path)
    w,h = pil_img.size
    img_np = np.array(pil_img,dtype='uint8')
    img_tensor = basic_transform(pil_img).to(device)

    net.eval()
    with torch.no_grad():
        offset,score = net(img_tensor.unsqueeze(dim=0))
        loc, score = offset_decode(offset[0], score[0], anchors)
        loc[0] = torch.round(w * loc[0])
        loc[1] = torch.round(h * loc[1])
        loc[2] = torch.round(w * loc[2])
        loc[3] = torch.round(h * loc[3])
        loc = loc.to('cpu')
        print(loc)
        cls = classes_list[score.to('cpu')]

    origin_img = img_np.copy()
    img_np = cv2.rectangle(img_np,(int(loc[0]),int(loc[1])),(int(loc[2]),int(loc[3])),
                           color=colors_dict[cls])
    img_np = cv2.putText(img_np,cls,(int(loc[0]),int(loc[1])),cv2.FONT_HERSHEY_SIMPLEX,0.4,
                         color=colors_dict[cls],thickness=1)
    cv2.namedWindow('origin_img', 0)
    cv2.imshow('origin_img', origin_img)
    cv2.resizeWindow('origin_img', 640, 480)
    cv2.namedWindow('prediction', 0)
    cv2.imshow('prediction', img_np)
    cv2.resizeWindow('prediction', 640, 480)
    cv2.waitKey(0)

if __name__ == "__main__":
    img_path = './tiny_vid/car/000002.JPEG'
    predict(img_path)




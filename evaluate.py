from utils.anchor_tool import offset_decode
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.dataset import tiny_dataset
from torch.utils.data import random_split, DataLoader
import argparse
import pickle

def args_parser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--load-model',help='the path pf .pth file of model',type=str,
                        required=True,dest='load_model')
    parser.add_argument('--device', type=str,
                        default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--kms-anchors-path',type=str,required=True)

    return parser

def bbox_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0

def compute_three_acc(offsets,scores,anchors,label,bbox,img_size):
    regression_acc = 0
    class_acc = 0
    acc = 0
    h,w = img_size

    assert offsets.shape[0]==1 and scores.shape[0]==1,'please let the val batch size equal to 1'
    assert h==w , 'please let the input img be a square'

    loc,score = offset_decode(offsets[0],scores[0],anchors)

    if score == label[0]+1:
        class_acc += 1

    bbox_loc = loc
    iou = bbox_iou(t.round(h * bbox_loc), bbox[0])
    # print(iou)
    if iou >= 0.5:
        regression_acc += 1

    if score == label[0]+1:
        bbox_loc = loc
        iou = bbox_iou(t.round(w * bbox_loc), bbox[0])
        # print(iou)
        if iou >= 0.5:
            acc += 1

    return class_acc , regression_acc , acc

if __name__ == '__main__':
    import tqdm
    from utils.anchor_tool import anchor_generate

    t.manual_seed(777)
    t.cuda.manual_seed(777)

    args = args_parser().parse_args()
    device = args.device
    with open(args.kms_anchors_path,'rb') as af:
        kms_anchors = pickle.load(af)

    train_set = tiny_dataset(mode='train', augment=False)
    val_set = tiny_dataset(mode='val', augment=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    print('loading ... ')
    net = t.load(args.load_model).to(device)
    print('evaluating ... ')
    net.eval()

    print('generating anchors ... ')
    k_selected = net.anchor_num_per_point
    anchors = anchor_generate(kms_anchor=kms_anchors).to(device)
    anchor_num = anchors.shape[0]
    print('finish generating !')

    print('start evaluating ... ')
    with t.no_grad():
        regression_acc = 0
        class_acc = 0
        acc = 0

        for i,item in tqdm.tqdm(enumerate(val_loader)):
            img = item['img'].to(device)
            label = item['label'].to(device)
            bbox = item['bbox'].to(device)

            offsets,scores = net(img)
            loc,score = offset_decode(offsets[0],scores[0],anchors)
            if score == label[0]+1:
                class_acc+=1

            bbox_loc = loc
            iou = bbox_iou(t.round(128*bbox_loc),bbox[0])
            # print(iou)
            if iou >= 0.5:
                regression_acc +=1

            if score == label[0]+1:
                bbox_loc = loc
                iou = bbox_iou(t.round(128*bbox_loc),bbox[0])
                # print(iou)
                if iou >= 0.5:
                    acc +=1


        print(class_acc/len(val_set))
        print(regression_acc/len(val_set))
        print(acc/len(val_set))


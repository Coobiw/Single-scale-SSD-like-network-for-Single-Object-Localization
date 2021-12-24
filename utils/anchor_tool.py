import numpy as np
import torch as t
import math
from scipy.cluster.vq import kmeans
import torch.nn.functional as F
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

def get_iou_matrix(anchors,gts): # anchor:[A,4] gt:[N,4] iou_matrix:[A,N]
    iou_matrix = t.zeros((anchors.shape[0],gts.shape[0]),dtype=t.float32,device=anchors.device)
    for ia in range(anchors.shape[0]):
        for ig in range(gts.shape[0]):
            iou_matrix[ia,ig] = bbox_iou(anchors[ia],gts[ig])
    # print(iou_matrix)
    return iou_matrix

def kms_result_anchor(bbox_hw,k_selected,max_k=21,device=t.device('cuda')):
    seed = 729608
    np.random.seed(seed)
    iter_time = 100
    k_list = list(range(1, max_k, 1))
    distortion_list = []
    centroid_list = []
    for k in k_list:
        centroid, distortion = kmeans(obs=bbox_hw.astype(np.float32), k_or_guess=k, iter=iter_time)
        centroid_list.append(centroid)
        distortion_list.append(distortion)

    centroid = centroid_list[k_selected - 1]

    return t.from_numpy(centroid)

def xywh2xxyy(bbox):
    cx,cy,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
    return t.tensor([cx-w/2,cy-h/2,cx+w/2,cy+h/2],dtype=bbox.dtype,device=bbox.device)

def xywh2offset(anchor,bbox):
    acx,acy,aw,ah = anchor[0],anchor[1],anchor[2],anchor[3]
    gcx,gcy,gw,gh = bbox[0],bbox[1],bbox[2],bbox[3]
    offset0 = (gcx - acx) / acx
    offset1 = (gcy - acy) / acy
    offset2 = t.log(gw/aw)
    offset3 = t.log(gh/ah)

    return t.tensor([offset0,offset1,offset2,offset3])

def offset2xywh(anchor,offset):
    acx, acy, aw, ah = anchor[0], anchor[1], anchor[2], anchor[3]
    max_exp_w_clamp = math.log(1./aw)
    max_exp_h_clamp = math.log(1./ah)
    cx = offset[0]*acx + acx
    cy = offset[1]*acy + acy

    w = t.exp(t.clamp(offset[2],max=max_exp_w_clamp)) * aw
    h = t.exp(t.clamp(offset[3],max=max_exp_h_clamp)) * ah

    x1 = cx - w/2
    x2 = cx + w/2
    y1 = cy - h/2
    y2 = cy + h/2

    return t.tensor([x1,y1,x2,y2])

def anchor_generate(kms_anchor,hh = 4,ww = 4,h = 128,w = 128):
    dtype = kms_anchor.dtype
    device = kms_anchor.device
    assert h == w , 'input image is not a square'

    # 归一化anchor的长宽
    kms_anchor = kms_anchor/h

    shifts_x = ((t.arange(0, ww)+0.5) / ww).to(dtype=dtype,device=device)
    shifts_y = ((t.arange(0, hh) + 0.5) / hh).to(dtype=dtype,device=device)
    shifts_y,shifts_x = t.meshgrid(shifts_x,shifts_y)
    shifts_x = shifts_x.reshape(-1,1)
    shifts_y = shifts_y.reshape(-1,1)
    anchor_list = []
    anchor_num_per_point = kms_anchor.shape[0]

    for i in range(hh*ww):
        for ia in range(anchor_num_per_point):
            anchor_list.append(t.tensor([shifts_x[i],shifts_y[i],kms_anchor[ia,1],kms_anchor[ia,0]],
                                        dtype = dtype,device=device))

    return t.cat(anchor_list,dim=0).reshape(-1,4)

def label_assignment(anchors,gts,label,threshold,img_size):
    # 如果是一张图有多个物体（bbox），则assign的应该是gt的序号
    # 这里由于每张图有且仅有一个gt，所以可以直接assign这个gt的类别
    anchors_num = anchors.shape[0]
    h,w = img_size

    assert h==w , 'please let the input image be a square'

    gts = gts/h
    iou_matrix = get_iou_matrix(anchors,gts)
    POSITIVE_SAMPLE = label+1

    assignment_matrix = t.zeros(iou_matrix.shape,dtype=label.dtype)

    # 计算对于每个anchor，最大的gt，这里不需要，因为每个图片有且仅有一个gt
    # 大于threshold的定为正样本
    assignment_matrix[iou_matrix>=threshold] = POSITIVE_SAMPLE

    # 对于每个gt，把和他iou最大的achor直接变为正样本，防止gt没有anchor匹配
    assignment_matrix[t.max(iou_matrix,dim=0)[1],0] = POSITIVE_SAMPLE

    # print(t.where(assignment_matrix !=0)[0].numel())

    return assignment_matrix

def bbox_encode(anchors,bbox,img_size): # bbox: (Batch_size,4)
    h,w = img_size
    batch_size = bbox.shape[0]
    anchor_num = anchors.shape[0]
    dtype = anchors.dtype
    device = anchors.device


    bbox_h = bbox[:, 3] - bbox[:, 1]
    bbox_w = bbox[:, 2] - bbox[:, 0]

    bbox_cx = bbox_w / 2 + bbox[:, 0]
    bbox_cy = bbox_h / 2 + bbox[:, 1]

    bbox_xywh = t.cat((bbox_cx.reshape(-1,1),bbox_cy.reshape(-1,1),
                       bbox_w.reshape(-1,1),bbox_h.reshape(-1,1)),dim=1)

    bbox_xywh = bbox_xywh/h
    offset_list = []

    for ib in range(batch_size):
        for ia in range(anchor_num):
            offset_list.append(xywh2offset(anchors[ia],bbox_xywh[ib]))

    return t.cat(offset_list,dim=0).reshape(batch_size,anchor_num,4)


def offset_decode(offsets,scores,anchors):
    # offsets:[anchors_num,4]
    # scores:[anchors_num,1+class_num]
    anchors_num = anchors.shape[0]
    results = t.zeros(offsets.shape,dtype=offsets.dtype,device=offsets.device)

    for ia in range(anchors_num):
        results[ia] = offset2xywh(anchors[ia],offsets[ia])

    classes = t.argmax(scores,dim=1)
    class_scores = F.softmax(scores,dim=1)
    class_scores = t.max(class_scores,dim=1)[0]
    foreground_anchor_index = t.where(classes!=0)[0]

    if foreground_anchor_index.numel() == 0:
        max_indice = t.argmax(class_scores)
        final_class = classes[max_indice]
        final_loc = results[max_indice, :]

    elif foreground_anchor_index.numel() ==1 :
        max_indice = foreground_anchor_index[0]
        final_class = classes[max_indice]
        final_loc = results[max_indice,:]

    else:
        max_indice = t.argmax(class_scores[foreground_anchor_index])
        final_indice = foreground_anchor_index[max_indice]
        final_class = classes[final_indice]

        final_loc = results[final_indice, :]

    return final_loc, final_class





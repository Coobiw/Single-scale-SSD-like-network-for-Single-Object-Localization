from utils.dataset import tiny_dataset
from resnet_based_ssd import Localization_anchor_net
from torch.utils.data import DataLoader,random_split
import torch as t
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import argparse
from loss_for_ssd import Loss_for_localization
from evaluate import compute_three_acc
import os

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',help='learning rate',type=float,default=1e-3,dest='lr')
    parser.add_argument('--batch-size',help='batchsize',type=int,default=8,dest='batch_size')
    parser.add_argument('--weight-decay', help='weight decay of optimizer', type=float,
                        default=1e-6, dest='weight_decay')
    parser.add_argument('--epochs', help='set the num of epochs', type=int, default=100)
    parser.add_argument('--low-lr-ratio',help='when fine-tune,pretrained model low lr ratio',
                        type = float,default = 1.)

    parser.add_argument('--anchor-num-per-point',type=int,default=3)
    parser.add_argument('--dropout', type=bool, default=False)
    parser.add_argument('--prob', type=float, default=0.5)

    parser.add_argument('--threshold',help='the threshold when assigning label',
                        type=float,default=0.4)
    parser.add_argument('--regre-weight',help='the regre-loss weight of loss function',
                        type = float,default = 1.)
    parser.add_argument('--neg-to-pos-ratio',help='hard negative sampling ratio',
                        type=float,default = 3.)


    parser.add_argument('--root',help='root directory of dataset',type=str,
                default=r'E:\BS_learning\4_1\CV_basis\experiment\SSD-like method\tiny_vid')
    parser.add_argument('--log-dir',help='tensorboard log dir',type=str,required=True)
    parser.add_argument('--save-file-name', help='the pth file name', type=str,required=True)


    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'],type=str)
    parser.add_argument('--gradient-clamp', default=False, type=bool)
    parser.add_argument('--gradient-norm2',default=20.,type=float)

    return parser


def weight_init(net):
    for name,child in net.named_children():
        if name == 'feature_extraction':
            continue

        if isinstance(child,nn.Conv2d):
            nn.init.kaiming_normal_(child.weight)
            if child.bias != None:
                nn.init.zeros_(child.bias)

        elif isinstance(child,nn.Linear):
            nn.init.kaiming_normal_(child.weight)
            if child.bias != None:
                nn.init.zeros_(child.bias)

        elif isinstance(child,nn.BatchNorm2d):
            nn.init.zeros_(child.bias)
            nn.init.ones_(child.weight)

    return net

def train():
    args = parser().parse_args()

    writer = SummaryWriter(log_dir=args.log_dir, comment='curves_log')

    with open(os.path.join(args.log_dir,'para.txt'),'w') as f:
        f.write('lr: '+str(args.lr))
        f.write('\n')
        f.write('epochs: '+str(args.epochs))
        f.write('\n')
        f.write('batch-size: ' + str(args.batch_size))
        f.write('\n')
        f.write('weight-decay: ' + str(args.weight_decay))
        f.write('\n')
        f.write('low_lr_ratio: ' + str(args.low_lr_ratio))
        f.write('\n')
        f.write('dropout: ' + str(args.dropout)+'\tprob: ' + str(args.prob))
        f.write('\n')
        f.write('anchor_num_per_point: ' + str(args.anchor_num_per_point))
        f.write('\n')
        f.write('threshold: ' + str(args.threshold))
        f.write('\n')
        f.write('regre-weight: ' + str(args.regre_weight))
        f.write('\n')
        f.write('neg-to-pos-ratio: ' + str(args.neg_to_pos_ratio))
        f.write('\n')
        f.write('gradient-clamp: ' + str(args.gradient_clamp))
        f.write('\n')
        f.write('gradient-clamp-norm: ' + str(args.gradient_norm2))
        f.write('\n\n')
        f.write('device: '+ str(args.device))

    device = t.device(args.device)
    t.manual_seed(777)
    t.cuda.manual_seed(777) # 保证每次实验结果一致

    print('loading the dataset ... ')
    dataset = tiny_dataset(root=args.root,k_selected=args.anchor_num_per_point,
                           label_assignment_threshold= args.threshold)

    # 保证每次random-split划分结果相同
    train_set,val_set = random_split(dataset=dataset,lengths=[150*5,30*5],
                                     generator=t.Generator().manual_seed(777))


    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True,num_workers=2)

    val_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=False,num_workers=0)

    print('establish the net ...')
    net = Localization_anchor_net(class_num=5,pretrained=True,
                            anchor_num_per_point=args.anchor_num_per_point,
                            dropout_or_not=args.dropout,prob=args.prob).to(device)

    print('initialize the net ... ')
    net = weight_init(net=net)

    # pretrained的网络用小学习率，从零开始的部分用大学习率
    high_lr_list = []
    low_lr_list = []
    for name,param in net.named_parameters():
        if 'feature_extraction' in name:
            low_lr_list.append(param)
        else:
            high_lr_list.append(param)

    optimizer = optim.Adam([{'params':low_lr_list,'lr':args.low_lr_ratio*args.lr},
                            {'params':high_lr_list}],
                          lr=args.lr,weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = args.epochs)


    criterion = Loss_for_localization(args.regre_weight,args.neg_to_pos_ratio).to(device)

    anchors = dataset.anchors

    print('start training ... ')
    for i in tqdm.tqdm(range(args.epochs)):
        tt_loss = 0.
        tr_loss = 0.
        tc_loss = 0.
        tc_acc = 0.
        tr_acc = 0.
        t_acc = 0.

        vt_loss = 0.
        vr_loss = 0.
        vc_loss = 0.
        vc_acc = 0.
        vr_acc = 0.
        v_acc = 0.

        tc_acc_num = 0
        tr_acc_num = 0
        t_acc_num = 0

        print('\n%dth epoch'%(i+1))

        for item in train_loader:

            net.train()
            img = item['img'].to(device)
            label = item['label'].to(device)
            bbox = item['bbox'].to(device)
            assigned_label = item['assigned_label'].to(device)
            encoded_bbox = item['encoded_bbox'].to(device)
            offsets,scores = net(img)


            loss_dict = criterion(offsets,scores,assigned_label,encoded_bbox)
            total_loss = loss_dict['total_loss']
            regre_loss = loss_dict['regre_loss']
            classification_loss = loss_dict['classification_loss']

            tt_loss += total_loss.item()
            tr_loss += regre_loss.item()
            tc_loss += classification_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            if args.gradient_clamp:
                nn.utils.clip_grad_norm_(net.parameters(),max_norm=args.gradient_norm2,norm_type=2)
            optimizer.step()


            for j in range(img.size()[0]):
                a,b,c = compute_three_acc(offsets[j].unsqueeze(dim=0),scores[j].unsqueeze(dim=0),
                                          anchors,label[j].unsqueeze(dim=0),
                                          bbox[j].unsqueeze(dim=0),img_size = (128,128))
                tc_acc_num += a
                tr_acc_num += b
                t_acc_num += c

        tc_acc = tc_acc_num/len(train_set)
        tr_acc = tr_acc_num / len(train_set)
        t_acc = t_acc_num / len(train_set)




        net.eval()
        with t.no_grad():
            for item2 in val_loader:
                img = item2['img'].to(device)
                label = item2['label'].to(device)
                bbox = item2['bbox'].to(device)
                assigned_label = item2['assigned_label'].to(device)
                encoded_bbox = item2['encoded_bbox'].to(device)
                offsets, scores = net(img)

                class_acc,regression_acc,acc = compute_three_acc(offsets,scores,anchors,
                                                                 label,bbox,img_size = (128,128))

                vc_acc += class_acc
                vr_acc += regression_acc
                v_acc += acc

                loss = criterion(offsets,scores,assigned_label,encoded_bbox)
                vt_loss += loss['total_loss'].item()
                vr_loss += loss['regre_loss'].item()
                vc_loss += loss['classification_loss'].item()

        vt_loss /= len(val_set)
        vr_loss /= len(val_set)
        vc_loss /= len(val_set)

        vc_acc /= len(val_set)
        vr_acc /= len(val_set)
        v_acc /= len(val_set)


        print('train_loss: %.5f  val_loss : %.5f' % (tt_loss/len(train_loader),vt_loss))

        writer.add_scalars('lr_curve', {'low_lr':optimizer.param_groups[0]["lr"]}, i + 1)
        writer.add_scalars('lr_curve', {'high_lr':optimizer.param_groups[1]["lr"]}, i + 1)

        writer.add_scalars('loss1', {'Train':tt_loss / len(train_loader)}, i+1)
        writer.add_scalars('loss1', {'Train_regre': tr_loss / len(train_loader)}, i + 1)
        writer.add_scalars('loss1', {'Train_cls': tc_loss / len(train_loader)}, i + 1)

        writer.add_scalars('loss2', {'Val':vt_loss}, i+1)
        writer.add_scalars('loss2', {'Val_regre': vr_loss}, i + 1)
        writer.add_scalars('loss2', {'Val_cls': vc_loss}, i + 1)

        writer.add_scalars('acc1', {'train_cls': tc_acc}, i + 1)
        writer.add_scalars('acc1', {'train_regre': tr_acc}, i + 1)
        writer.add_scalars('acc1', {'train': t_acc}, i + 1)

        writer.add_scalars('acc2',{'val_cls':vc_acc},i+1)
        writer.add_scalars('acc2', {'val_regre': vr_acc}, i + 1)
        writer.add_scalars('acc2', {'val': v_acc}, i + 1)

        scheduler.step()


    t.save(net,os.path.join(args.log_dir,args.save_file_name + 'epoch%d.pth'%(i+1)))




if __name__ == '__main__':
    train()
   #
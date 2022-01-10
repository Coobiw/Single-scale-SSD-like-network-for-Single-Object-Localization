import torch.nn.functional as F
import torch.nn as nn
import torch as t


class Loss_for_localization(nn.Module):
    def __init__(self,regre_weight : float = 1.,neg_to_pos_ratio : float = 3.):
        super(Loss_for_localization,self).__init__()
        self.alpha = regre_weight
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def forward(self,offsets,scores,assigned_labels,encoded_bboxes):
        # offsets: [batch_size,anchor_num,4]
        # scores: [batch_size,anchor_num,class_num+1]
        # assigned_labels: [batch_size,anchor_num,1]
        # encoded_bboxes : [batch_size,anchor_num,4]

        batch_size,anchor_num,class_num_include_bg = scores.shape

        foreground_anchor_idx = t.where(assigned_labels[:,:,0] != 0)
        background_anchor_idx = t.where(assigned_labels[:,:,0] == 0)

        num_foreground = foreground_anchor_idx[0].shape[0]
        num_background_wanted = min(background_anchor_idx[0].shape[0],
                             int(self.neg_to_pos_ratio * num_foreground))

        regre_loss = F.smooth_l1_loss(offsets[foreground_anchor_idx],
                                      encoded_bboxes[foreground_anchor_idx],reduction='sum')

        classification_loss_positive_sample = F.cross_entropy(scores[foreground_anchor_idx],
                                      assigned_labels[foreground_anchor_idx].view(-1).long(),
                                                              reduction='sum')

        classification_loss_per_negative_sample = F.cross_entropy(scores[background_anchor_idx],
                                      assigned_labels[background_anchor_idx].view(-1).long(),
                                                                  reduction='none')

        # Hard Negative Sampling，这里针对每个样本，而不是整个batch
        # Hard Negative Sampling是为了维持正负样本比值为neg_to_pos_ratio
        hard_negative_classification_loss,_ = classification_loss_per_negative_sample.sort(dim=0,
                                                    descending=True)[:num_background_wanted]
        hard_negative_classification_loss = hard_negative_classification_loss.sum()

        classification_loss = classification_loss_positive_sample \
                              + hard_negative_classification_loss

        regre_loss = regre_loss / num_foreground
        classification_loss = classification_loss / num_foreground
        total_loss = classification_loss + self.alpha * regre_loss

        return {'total_loss':total_loss,'regre_loss':regre_loss,
                'classification_loss':classification_loss}
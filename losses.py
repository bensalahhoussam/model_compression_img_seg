import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError:
    from itertools import  filterfalse as ifilterfalse



class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1., smooth=1e-6):

        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):

        y_pred_probs = F.softmax(y_pred, dim=1)  # Shape: (batch_size, num_classes, height, width)

        # Convert y_true to one-hot encoding
        num_classes = y_pred.shape[1]
        y_true_onehot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()  # Shape: (batch_size, num_classes, height, width)

        # Flatten the tensors
        y_true_f = y_true_onehot.reshape(-1)  # Use reshape instead of view
        y_pred_f = y_pred_probs.reshape(-1)   # Use reshape instead of view

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = (y_true_f * y_pred_f).sum()
        fp = ((1 - y_true_f) * y_pred_f).sum()
        fn = (y_true_f * (1 - y_pred_f)).sum()

        # Tversky Index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Focal Tversky Loss
        focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)

        return focal_tversky_loss
class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=27):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.long().device).scatter_(1, tensor.long().view(n, 1, h, w), 1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        inputs = F.softmax(inputs, dim=1)

        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)

        union = inputs + target_oneHot
        union = union.view(N, self.classes, -1).sum(2)

        loss = 2 * inter / union

        return 1 - loss.mean()
class FocalLoss(nn.Module):
    def __init__(self,alpha=None, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.smooth = 1e-5


    def forward(self, logit, target):
        
        logit = F.softmax(logit,dim=1)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        
        
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
        
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        
        pt = (one_hot_key * logit).sum(1) 
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        loss = loss.mean()

        return loss
class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, inputs, target):
        N = inputs.size()[0]

        inputs = F.softmax(inputs, dim=1)

        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        intersection = inter.view(N, self.classes, -1).sum(2)

        union = inputs + target_oneHot
        union = union.view(N, self.classes, -1).sum(2)

        dice_loss = 1 - ((2 * intersection + 1e-5) / (union + 1e-5))
        dice_loss = dice_loss.mean()
        return dice_loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
def isnan(x):
    return x != x
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss
def lovasz_softmax_flat(probas, labels, classes='present'):

    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)
def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

"""********************************************************"""
"""********************************************************"""
"""********************************************************"""
"""********************************************************"""

class CombinedLossMultiClass(nn.Module):
    def __init__(self, num_classes, gamma=2,alpha=0.4, beta=0.6, y1=1.0, y2=1.0, y3=1.0, y4=1.0,class_weights=None):
        super(CombinedLossMultiClass, self).__init__()
        self.num_classes = num_classes

        self.gamma = gamma
        self.dw = y1
        self.fw = y2
        self.tw = y3
        self.lw = y4

        self.class_weights = class_weights

        self.f_loss =  FocalLoss(alpha=self.class_weights, gamma=self.gamma)

        self.d_loss = DiceLoss(n_classes=self.num_classes)

        self.tv = FocalTverskyLoss(alpha= alpha, beta=beta, gamma=self.gamma, smooth=1e-6)


    def forward(self, input, target):

        fl = self.f_loss(input,target)
        dl = self.d_loss(input,target)
        tl =  self.tv(input,target)

        lv_loss = lovasz_softmax(F.softmax(input, dim=1), target)

        total_loss = (self.fw  * fl +self.dw* dl + self.tw * tl +  self.lw * lv_loss)
        return total_loss






from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F
import numpy as np




class SegmentationMetrics(object):
    def __init__(self, eps=1e-6, average=True):
        self.eps = eps
        self.average = average

    def dice_coefficient(pred_mask, true_mask, num_classes):
        dice_scores = []
        pred_mask = torch.argmax(pred_mask, dim=1)  # Convert logits to class indices
        for class_id in range(num_classes):
            pred_class = (pred_mask == class_id)
            true_class = (true_mask == class_id)
            intersection = (pred_class & true_class).sum().float()
            dice = (2 * intersection + 1e-6) / (pred_class.sum() + true_class.sum() + 1e-6)
            dice_scores.append(dice)
        return torch.tensor(dice_scores).mean()

    def pixel_accuracy(pred_mask, true_mask):
        pred_mask = torch.argmax(pred_mask, dim=1)  # Convert logits to class indices
        correct = (pred_mask == true_mask).sum().float()
        total = true_mask.numel()
        return correct / total

    def vectorized_metrics(pred_mask, true_mask, num_classes):

        # Convert logits to class indices
        pred_mask = torch.argmax(pred_mask, dim=1)  # Shape: (batch_size, height, width)

        # Create one-hot encodings for predictions and ground truth
        pred_one_hot = torch.nn.functional.one_hot(pred_mask.long(), num_classes).permute(0, 3, 1,
                                                                                          2)  # Shape: (batch_size, num_classes, height, width)
        true_one_hot = torch.nn.functional.one_hot(true_mask.long(), num_classes).permute(0, 3, 1,
                                                                                          2)  # Shape: (batch_size, num_classes, height, width)

        # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = (pred_one_hot & true_one_hot).sum(dim=(0, 2, 3)).float()  # Shape: (num_classes,)
        fp = (pred_one_hot & ~true_one_hot).sum(dim=(0, 2, 3)).float()  # Shape: (num_classes,)
        fn = (~pred_one_hot & true_one_hot).sum(dim=(0, 2, 3)).float()  # Shape: (num_classes,)

        # Compute Precision, Recall, and IoU
        precision = (tp + 1e-6) / (tp + fp + 1e-6)  # Shape: (num_classes,)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)  # Shape: (num_classes,)
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)  # Shape: (num_classes,)

        return precision, recall, iou

    @staticmethod
    def _one_hot(targets):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        target_class = F.one_hot(targets.long(), num_classes=27).permute(0, 3, 1, 2).float()

        return target_class

    @staticmethod
    def _get_stats_multilabel(target, output,num_class):


        matrix = np.zeros((3,num_class))




        for i in range(num_class):
            class_pr = output[:,i,:,:]

            class_gt = target[:,i,:,:]

            pred_flat = class_pr.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        matrix = self._get_stats_multilabel(gt, pred, class_num)

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        dice = np.average(dice)
        precision = np.average(precision)
        recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        pred_argmax = torch.argmax(y_pred,dim=1)

        activated_pred = self._one_hot(pred_argmax)

        gt_onehot = self._one_hot(y_true)

        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)

        f1_score = (2*precision*recall)/(precision+recall)
        return pixel_acc, dice, precision, recall,f1_score
metric_calculator = SegmentationMetrics()



def compute_mAP(pred_masks, true_masks, num_classes, iou_threshold=0.5):

    aps = []
    pred_masks=torch.softmax(pred_masks, dim=1)

    true_masks = torch.nn.functional.one_hot(true_masks.long(), num_classes).permute(0, 3, 1,2)

    for class_id in range(num_classes):
        # Flatten predictions and ground truth for the current class across the whole batch
        pred_mask_flat = pred_masks[:, class_id, ...].flatten()  # Shape: (batch_size * height * width,)
        true_mask_flat = true_masks[:, class_id, ...].flatten()  # Shape: (batch_size * height * width,)

        if true_mask_flat.sum() == 0:
            continue

            # Binarize predictions using a threshold (e.g., 0.5)
        pred_mask_binary = (pred_mask_flat >  iou_threshold).float()

        # Convert tensors to numpy for sklearn's average_precision_score
        pred_mask_binary_np = pred_mask_binary.cpu().numpy()
        true_mask_flat_np = true_mask_flat.cpu().numpy()

        # Compute Average Precision (AP) for the current class
        ap = average_precision_score(true_mask_flat_np, pred_mask_binary_np)
        aps.append(ap)

    # Compute mAP as the mean of APs across all classes
    mAP = np.array(aps).mean()
    return mAP
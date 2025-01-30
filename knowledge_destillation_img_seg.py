from dataloader import get_data_loaders,get_dataset,get_images,label_color_list, all_classes
import torch
from models import MACUNet,Student_MACUNet
from losses import CombinedLossMultiClass
import torch.nn.functional as F
from metrics import metric_calculator,compute_mAP
import time
from tqdm import tqdm
import torch.nn as nn
from utils import initialize_weights,get_model_size,get_metrics,get_data,save_checkpoint, load_checkpoint
from optimizer import smart_optimizer,WarmupStableDecayLR
from early_stopping import EarlyStopping
from inference import  model_prediction,get_prediction_map
import cv2 as cv
import argparse
from engine import KD_train,valid



parser = argparse.ArgumentParser()
parser.add_argument('--root_path',default="dataset/idd/",help='dataset path',type=str)
parser.add_argument('--img_size',default=256//2,help='image size',type=int)
parser.add_argument('--weight_decay',default=0.0005,help='L2 regularization',type=float)

parser.add_argument('--gradient_clipping',default=1.,help='gradient_clipping',type=float)


parser.add_argument('--lr',default=1e-3,help='learning rate',type=float)
parser.add_argument('--momentum',default=0.9,help='previous accumulation',type=float)
parser.add_argument('--optimizer',default="AdamW",help='optimizer',type=str)


parser.add_argument('--epochs',default=150,help='number of epochs to train for',type=int)
parser.add_argument('--batch',default=8,help='batch size for data loader',type=int)
parser.add_argument('--scheduler',action='store_true',)


parser.add_argument('--student_weight',default="best_model/checkpoint_16.pt",help='checkpoint for student model',type=str)
parser.add_argument('--teacher_weight',default="best_model/checkpoint_teacher.pt",help='checkpoint for teacher model',type=str)




args = parser.parse_args()




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student_model = Student_MACUNet.MACUNet(3,27).to(device)
    state_dict_student = torch.load(args.student_weight,weights_only=False)
    student_model.load_state_dict(state_dict_student['model_state_dict'],strict=False)

    teacher_model = MACUNet.MACUNet(3,27).to(device)
    state_dict_teacher = torch.load(args.teacher_weight,weights_only=False)
    teacher_model.load_state_dict(state_dict_teacher['model_state_dict'],strict=False)

    train_data, train_labels, test_data, test_labels = get_images(args.root_path)
    train_dataset, valid_dataset = get_dataset(train_data, train_labels, test_data, test_labels, label_color_list, all_classes,args.img_size, args.img_size)
    train_data_loader, valid_data_loader = get_data_loaders(train_dataset, valid_dataset, args.batch)

    alpha = [
        0.1386,  # Class 0
        0.5562,  # Class 1
        5.6027,  # Class 2
        3.8723,  # Class 3
        4.3214,  # Class 4
        4.1834,  # Class 5
        2.5492,  # Class 6
        120.3704,  # Class 7
        2.6639,  # Class 8
        1.1852,  # Class 9
        2.1454,  # Class 10
        6.0000,  # Class 11
        7.3704,  # Class 12
        3.0000,  # Class 13
        1.8426,  # Class 14
        7.0278,  # Class 15
        20.2593,  # Class 16
        2.4907,  # Class 17
        43.4815,  # Class 18
        859.2593,  # Class 19
        2.9815,  # Class 20
        0.8296,  # Class 21
        0.5111,  # Class 22
        3.0833,  # Class 23
        0.1889,  # Class 24
        0.2463,  # Class 25
        83.3333   # Class 26
    ]
    weights = torch.tensor(alpha).to(device)

    dice = 0.55
    focal = 0.
    tv = 0.
    lv = .45
    criterion  = CombinedLossMultiClass(27, gamma=2., alpha=0.3,beta=0.7, y1=dice, y2=focal, y3=tv, y4=lv,class_weights=alpha)

    early_stopping = EarlyStopping(patience=13, verbose=True)

    optimizer = smart_optimizer(student_model,args.optimizer,args.lr,args.momentum,args.weight_decay)

    warmup_steps = len(train_data_loader)*int(args.epochs*0.16)
    stable_steps = len(train_data_loader)*int(args.epochs*0.06)
    decay_steps = len(train_data_loader)*int(args.epochs*0.78)
    scheduler = WarmupStableDecayLR( optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-4, base_lr=1e-3, final_lr=1e-4)

    T=6
    soft_target_loss_weight=0.35
    ce_loss_weight=0.65

    tran_losses=[]
    train_results={'precision':[],'recall':[],"dice":[],"f1_score":[],'mAP':[]}
    valid_losses=[]
    valid_results={'precision':[],'recall':[],"dice":[],"f1_score":[],'mAP':[]}

    for epoch in range(0, args.epochs):
        print(f"Start Epoch N : {epoch + 1}/{args.epochs} \n")

        train_loss, train_metrics = KD_train(student_model,teacher_model, train_data_loader, optimizer, device, criterion, scheduler,
                                             epoch,T, soft_target_loss_weight, ce_loss_weight)

        train_results = get_data(train_results, train_metrics)

        tran_losses.append(train_loss)

        valid_loss, valid_metrics = valid(student_model, valid_data_loader, device, criterion, epoch_num=epoch)

        valid_results = get_data(valid_results, valid_metrics)

        valid_losses.append(valid_loss)

        save_checkpoint(epoch + 1, student_model, optimizer, scheduler, train_loss, train_metrics, valid_loss, valid_metrics)

        early_stopping(valid_loss, student_model, epoch + 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        student_model.eval().to(device)

        outputs, image, forward_time = model_prediction("dataset/idd/val/images/0000000_leftImg8bit.jpg", 128, student_model,
                                                        device)
        segmentation_map = get_prediction_map(outputs)
        output_final = cv.cvtColor(segmentation_map, cv.COLOR_RGB2BGR)
        cv.imwrite(f"outputs/inference_results/out_{epoch + 1}.png", output_final)

        """if early_stopping.early_stop:
            print("Early stopping triggered!")
            break"""
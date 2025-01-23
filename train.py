import numpy as np
import torch
from models.MACUNet import MACUNet
"""from models.Student_MACUNet import MACUNet"""

from losses import CombinedLossMultiClass,LovaszSoftmax,mIoULoss
from dataloader import get_data_loaders,get_dataset,get_images,label_color_list, all_classes
from optimizer import smart_optimizer,WarmupStableDecayLR
from early_stopping import EarlyStopping
from utils import initialize_weights,get_model_size,get_metrics,get_data,save_checkpoint, load_checkpoint
import os
from logger_config import logger
from engine import train,valid
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path',default="dataset/CamVid",help='dataset path',type=str)
parser.add_argument('--img_size',default=256,help='image size',type=int)
parser.add_argument('--weight_decay',default=0.0005,help='L2 regularization',type=float)

parser.add_argument('--gradient_clipping',default=1.,help='gradient_clipping',type=float)


parser.add_argument('--lr',default=1e-3,help='learning rate',type=float)
parser.add_argument('--momentum',default=0.9,help='previous accumulation',type=float)
parser.add_argument('--optimizer',default="AdamW",help='optimizer',type=str)


parser.add_argument('--epochs',default=100,help='number of epochs to train for',type=int)
parser.add_argument('--batch',default=8,help='batch size for data loader',type=int)
parser.add_argument('--scheduler',action='store_true',)
args = parser.parse_args()




if __name__ == '__main__':

    logger.info(args)

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    root_path="dataset/idd/"
    train_data, train_labels, test_data, test_labels = get_images(root_path)
    train_dataset, valid_dataset = get_dataset(train_data, train_labels, test_data, test_labels, label_color_list, all_classes,
                                               256, 256)
    train_data_loader, valid_data_loader = get_data_loaders(train_dataset, valid_dataset, 32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_weights = torch.tensor( [
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
]).to(device)

    model = MACUNet(3,27).to(device)

    model.apply(initialize_weights)

    """state_dict = torch.load("best_model/checkpoint_b_6.pt",weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'],strict=False)"""

    model_size = get_model_size(model)

    logger.info(f"Size of the model = {round(model_size,5)} Mb")

    input_x = torch.randn(1, 3, args.img_size,args.img_size).to(device)
    MMACs,MFLOPs, Mparams = get_metrics(model, input_x)
    logger.info(f"{round(MMACs,3)} MMACs, {round(MFLOPs,3)} MFLOPs and {round(Mparams,3)} M parameters")


    alpha  = [
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


    #criterion= CombinedLoss(alpha= alpha, num_classes=27,initial_weights=[1.0,1.0,1.0],smoothing_factor=0.15)
    #criterion = CombinedLossMultiClass(27, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5,class_weights=class_weights)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion =  CombinedLossMultiClass(27, gamma=2.0, dice_weight=0.4, focal_weight=0.6,class_weights=class_weights)

    early_stopping = EarlyStopping(patience=10, verbose=True)

    optimizer = smart_optimizer(model,args.optimizer,args.lr,args.momentum,args.weight_decay)



    """scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                steps_per_epoch=len(train_data_loader),anneal_strategy="cos" ,epochs=args.epochs)"""

    warmup_steps = len(train_data_loader)*int(args.epochs*0.16)
    stable_steps = len(train_data_loader)*int(args.epochs*0.06)
    decay_steps = len(train_data_loader)*int(args.epochs*0.78)
    scheduler = WarmupStableDecayLR( optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-5, base_lr=1e-3, final_lr=1e-5)


    tran_losses=[]
    train_results={'precision':[],'recall':[],"dice":[],"f1_score":[],'mAP':[]}
    valid_losses=[]
    valid_results={'precision':[],'recall':[],"dice":[],"f1_score":[],'mAP':[]}

    resume =True  # Set to None if starting from scratch, or set to the epoch number to resume from
    if resume is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pth')
        last_epoch, _, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)



    for epoch in range(last_epoch , args.epochs):

        print(f"Start Epoch N : {epoch + 1}/{args.epochs} \n")

        train_loss,train_metrics = train(model,train_data_loader,optimizer,device,criterion,scheduler,epoch)

        train_results = get_data(train_results,train_metrics)

        tran_losses.append(train_loss)

        valid_loss,valid_metrics = valid(model, valid_data_loader,device,criterion,epoch_num=epoch)
        valid_results = get_data(valid_results,valid_metrics)

        valid_losses.append(valid_loss)

        save_checkpoint(epoch+1, model, optimizer, scheduler, train_loss, train_metrics, valid_loss, valid_metrics)

        early_stopping(valid_loss, model,epoch+1)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break


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


student_weight = "best_model/checkpoint_student.pt"
teacher_weight = "best_model/checkpoint_teacher.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


student_model = Student_MACUNet.MACUNet(3,27).to(device)
state_dict_student = torch.load(student_weight,weights_only=False)
student_model.load_state_dict(state_dict_student['model_state_dict'],strict=False)



teacher_model = MACUNet.MACUNet(3,27).to(device)
state_dict_teacher = torch.load(teacher_weight,weights_only=False)
teacher_model.load_state_dict(state_dict_teacher['model_state_dict'],strict=False)



root_path="dataset/idd/"
train_data, train_labels, test_data, test_labels = get_images(root_path)
train_dataset, valid_dataset = get_dataset(train_data, train_labels, test_data, test_labels, label_color_list, all_classes,
                                               128, 128)
train_data_loader, valid_data_loader = get_data_loaders(train_dataset, valid_dataset, 32)


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


dice = 0.35
focal = 0.
tv = 0.
lv = 0.65
criterion = CombinedLossMultiClass(27, gamma=2., alpha=0.3,beta=0.7, y1=dice, y2=focal, y3=tv, y4=lv,class_weights=alpha)



def KL_Div(student_preds, teacher_preds, T=7.0):
    x = F.log_softmax(student_preds/T, dim=1)
    y = F.softmax(teacher_preds/T, dim=1)
    KLDiv = F.kl_div(x,y, reduction='batchmean')
    return KLDiv

def KD_Loss(target_labels, student_preds, teacher_preds, criterion, T=1.0, alpha=0.7):
    student_classification_loss = criterion(student_preds, target_labels)
    teacher_distillation_loss = KL_Div(student_preds, teacher_preds, T)
    KDLoss = (1.0 - alpha) * student_classification_loss + (alpha * T * T) * teacher_distillation_loss
    return KDLoss




def train(student_model,teacher_model,train_dataloader,optimizer,device,criterion,scheduler,epoch_num,T, soft_target_loss_weight, ce_loss_weight):

    print(f'start Training for epoch : {epoch_num+1}\n')
    student_model.train()
    teacher_model.eval()


    num_batches = len(train_dataloader)
    train_running_loss = 0.0
    prog_bar = tqdm(train_dataloader, total=num_batches, desc=f"Train Epoch {epoch_num+1}", unit="batch")

    counter = 0

    result_metrics = {'precision':0.0,'recall':0.0,'dice':0.0,'f1_score':0.0,'mAP' : 0.0}

    for i, data in enumerate(prog_bar):
        counter += 1


        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.zero_grad()

        images, target = data[0].to(device) , data[1].to(device),

        with torch.no_grad():
            teacher_logits = teacher_model(images).to(device)

        start_time = time.time()

        student_logits = student_model(images).to(device)

        forward_time = time.time() - start_time

        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T ** 2)
        hard_loss = criterion(student_logits, target)

        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * hard_loss



        loss.backward()
        """clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)"""
        optimizer.step()
        scheduler.step()

        train_running_loss += loss.item()

        accuracy, dice, precision, recall, f1_score = metric_calculator(target, student_logits)
        mAP = compute_mAP(student_logits, target, num_classes=27)

        result_metrics["dice"] += dice
        result_metrics["precision"] += precision
        result_metrics["recall"] += recall
        result_metrics["f1_score"] += f1_score
        result_metrics["mAP"] += mAP


        prog_bar.set_postfix({'loss':  loss.item(),'precision': precision,
            'recall': recall,'f1_score': f1_score,
            'dice': dice,'mAP': mAP,'lr': current_lr,
            'forward_time (s)': forward_time})

    result_metrics["dice"] = round(result_metrics["dice"] / counter,3)
    result_metrics["precision"] = round(result_metrics["precision"] / counter,3)
    result_metrics["recall"] = round(result_metrics["recall"] / counter,3)
    result_metrics["f1_score"] = round(result_metrics["f1_score"] / counter,3)
    result_metrics["mAP"]  = round(result_metrics["mAP"] / counter,3)

    train_loss = round(train_running_loss / counter,3)



    print("\n********* Training Results *********")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Metrics: {result_metrics}")
    print("********* ######## ####### *********\n")

    return train_loss,result_metrics
def valid(model,valid_dataloader,device,criterion,epoch_num):
    print(f'start Validation for epoch : {epoch_num+1}\n')
    model.eval()
    num_batches = len(valid_dataloader)
    train_running_loss = 0.0
    prog_bar = tqdm(valid_dataloader, total=num_batches, desc=f"Valid Epoch {epoch_num+1}", unit="batch")

    counter = 0

    result_metrics = {'precision':0.0,'recall':0.0,'dice':0.0,'f1_score':0.0,'mAP' : 0.0}

    for i, data in enumerate(prog_bar):
        counter += 1

        images, target = data[0].to(device) , data[1].to(device),

        with torch.no_grad():
            outputs = model(images)
        #loss = criterion(F.softmax(outputs, dim=1).float(), F.one_hot(target.long(), 27).permute(0, 3, 1, 2).float())
        loss = criterion(outputs, target)




        train_running_loss += loss.item()

        accuracy, dice, precision, recall, f1_score = metric_calculator(target, outputs)

        mAP = compute_mAP(outputs, target, num_classes=27)

        result_metrics["dice"] += dice
        result_metrics["precision"] += precision
        result_metrics["recall"] += recall
        result_metrics["f1_score"] += f1_score
        result_metrics["mAP"] += mAP



        prog_bar.set_postfix({'valid loss':  loss.item(),'valid precision': precision,
            'valid recall': recall,'valid f1_score': f1_score,'valid dice': dice,})


    result_metrics["dice"] = round(result_metrics["dice"] / counter,3)
    result_metrics["precision"] = round(result_metrics["precision"] / counter,3)
    result_metrics["recall"] = round(result_metrics["recall"] / counter,3)
    result_metrics["f1_score"] = round(result_metrics["f1_score"] / counter,3)
    result_metrics["mAP"]  = round(result_metrics["mAP"] / counter,3)

    train_loss = round(train_running_loss / counter,3)


    print("\n********* Validation Results *********")
    print(f"Validation Loss: {train_loss:.4f}")
    print(f"Validation Metrics: {result_metrics}")
    print("********* ######## ####### *********\n")

    return train_loss,result_metrics

early_stopping = EarlyStopping(patience=13, verbose=True)

optimizer = smart_optimizer(student_model,"AdamW",1e-3,0.9,0.0005)

epochs = 150
warmup_steps = len(train_data_loader)*int(epochs*0.16)
stable_steps = len(train_data_loader)*int(epochs*0.06)
decay_steps = len(train_data_loader)*int(epochs*0.78)
scheduler = WarmupStableDecayLR( optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-4, base_lr=1e-3, final_lr=1e-4)

T=7
soft_target_loss_weight=0.25
ce_loss_weight=0.75


tran_losses=[]
train_results={'precision':[],'recall':[],"dice":[],"f1_score":[],'mAP':[]}
valid_losses=[]
valid_results={'precision':[],'recall':[],"dice":[],"f1_score":[],'mAP':[]}

for epoch in range(0, epochs):
    print(f"Start Epoch N : {epoch + 1}/{epochs} \n")

    train_loss, train_metrics = train(student_model,teacher_model, train_data_loader, optimizer, device, criterion, scheduler, epoch,T, soft_target_loss_weight, ce_loss_weight)

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
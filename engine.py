import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from metrics import metric_calculator,compute_mAP
import time
from logger_config import logger
import torch.nn.functional as F
import torch.nn as nn



def train(model,train_dataloader,optimizer,device,criterion,scheduler,epoch_num):

    print(f'start Training for epoch : {epoch_num+1}\n')
    model.train()
    num_batches = len(train_dataloader)
    train_running_loss = 0.0
    prog_bar = tqdm(train_dataloader, total=num_batches, desc=f"Train Epoch {epoch_num+1}", unit="batch")

    counter = 0

    result_metrics = {'precision':0.0,'recall':0.0,'dice':0.0,'f1_score':0.0,'mAP' : 0.0}

    for i, data in enumerate(prog_bar):
        counter += 1

        #current_lr = scheduler.get_last_lr()[0] #"""optimizer.state_dict()["param_groups"][0]["lr"]"""
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.zero_grad()

        images, target = data[0].to(device) , data[1].to(device),

        start_time = time.time()
        outputs = model(images).to(device)

        forward_time = time.time() - start_time



        #loss = criterion(F.softmax(outputs, dim=1).float(), F.one_hot(target.long(), 27).permute(0, 3, 1, 2).float())
        loss = criterion(outputs, target)





        loss.backward()
        """clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)"""
        optimizer.step()
        scheduler.step()

        train_running_loss += loss.item()

        accuracy, dice, precision, recall, f1_score = metric_calculator(target, outputs)
        mAP = compute_mAP(outputs, target, num_classes=27)

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

    logger.info(f"Epoch {epoch_num + 1} - Training Loss: {train_loss:.4f}")
    logger.info(f"Epoch {epoch_num + 1} - Training Metrics: {result_metrics}")

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

    logger.info(f"Epoch {epoch_num + 1} - Validation Loss: { train_loss:.4f}")
    logger.info(f"Epoch {epoch_num + 1} - Validation Metrics: {result_metrics}")



    print("\n********* Validation Results *********")
    print(f"Validation Loss: {train_loss:.4f}")
    print(f"Validation Metrics: {result_metrics}")
    print("********* ######## ####### *********\n")

    return train_loss,result_metrics
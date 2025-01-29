import matplotlib.pyplot as plt
import torch
import argparse
import cv2 as cv
import os
from utils import get_segment_labels
from models.MACUNet import MACUNet
import numpy as np
import pandas as pd
from dataloader import label_color_list

def read_mask(mask_path,image_size):
    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    mask = cv.resize(mask, (int(image_size), int(image_size)))

    return mask
def label_mask(mask):
    mask= np.array(mask).astype("int32")
    label_mask = np.zeros(shape=(args.image_size,args.image_size),dtype="int32")
    for index ,color in enumerate(label_color_list):
        label_mask[np.where(np.all(mask==np.array(color),axis=-1))[:2]]=index
    label_mask = label_mask.astype('int32')
    return label_mask

def model_prediction(image_path,image_size,model,device):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (int(image_size), int(image_size)))

    image_normalized = image / 255.0
    image_tensor = torch.permute(torch.tensor(image_normalized, dtype=torch.float32), (2, 0, 1))

    outputs,forward_time = get_segment_labels(image_tensor, model, device)

    return outputs,image,forward_time
def get_prediction_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0,len(label_color_list)):
        index = labels == label_num
        red_map[index] = np.array(label_color_list)[label_num, 0]
        green_map[index] = np.array(label_color_list)[label_num, 1]
        blue_map[index] = np.array(label_color_list)[label_num, 2]

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

    return segmentation_map
def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def get_recall_and_precision(true_mask,labels):
    result_precision ={f"class_{i}":0 for i in range(27) }
    result_recall= {f"class_{i}":0 for i in range(27) }
    result_iou = {f"class_{i}":0 for i in range(27) }
    result_dice = {f"class_{i}":0 for i in range(27) }

    for i in range(27):

        intersection = np.sum((true_mask == i) * (labels == i))

        y_true_area = np.sum((true_mask == i))

        y_pred_area = np.sum((labels == i))
        combined_area = y_true_area + y_pred_area
        iou = (intersection) / (combined_area - intersection + 1e-5)
        result_iou[list(result_iou.keys())[i]] = round(iou,3)
        dice_score = 2 * ((intersection) / (combined_area + 1e-5))
        result_dice[list(result_dice.keys())[i]] = round(dice_score,3)


        true_positive = np.sum((true_mask==i)*(labels==i))
        false_positive = np.sum((labels==i)*(true_mask!=i))
        false_negative =np.sum((labels!=i)*(true_mask==i))

        if true_positive==0 or false_positive ==0:
            precision = 0
        else :
            precision = round(true_positive/(true_positive+false_positive),3)
        if true_positive==0 or false_negative==0:
            recall = 0

        else :
            recall = round(true_positive/(true_positive+false_negative),3)
        result_precision[list(result_precision.keys())[i]]=precision
        result_recall[list(result_recall.keys())[i]]=recall

    df_precision = pd.DataFrame.from_dict(result_precision, orient='index', columns=['precision'])
    df_recall = pd.DataFrame.from_dict(result_recall, orient='index', columns=['recall'])
    df_iou = pd.DataFrame.from_dict(result_iou, orient='index', columns=['iou'])
    df_dice = pd.DataFrame.from_dict(result_dice, orient='index', columns=['dice'])

    df_combined = pd.concat([df_precision, df_recall,df_iou,df_dice], axis=1)
    df_filtered = df_combined.loc[~(df_combined== 0).all(axis=1)]


    mean_pre = df_filtered.iloc[:, 0].mean()
    mean_recall = df_filtered.iloc[:, 1].mean()
    mean_iou = df_filtered.iloc[:, 2].mean()
    mean_dice = df_filtered.iloc[:, 3].mean()
    dicts = {"mean precision":round(mean_pre,3),"mean recall":round(mean_recall,3),
            "mean iou":round(mean_iou,3),"mean dice":round(mean_dice,3)}
    data = pd.DataFrame.from_dict(dicts, orient='index')

    return df_filtered,data




parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_image',default=f"dataset/images/val/image_0.jpg",help='path to input dir')
parser.add_argument('--true_label',default= "dataset/labels/val/mask_0.png",help='true label')
parser.add_argument('--model',default='outputs/best_model_iou (4).pth',help='path to the model checkpoint')
parser.add_argument('--image_size',default=128,help='image resize resolution')
parser.add_argument('--show',action='store_true',help='whether or not to show the output as inference is going')
args = parser.parse_args()

args.input_image = "dataset/idd/val/images/0002277_leftImg8bit.jpg"
args.true_label = "dataset/idd/val/rgb_labels/0002277_gtFine_polygons.png"


device = torch.device('cpu')
model = MACUNet(3, 27).to(device)
state_dict = torch.load(f"best_model/checkpoint_teacher.pt",weights_only=False)
model.load_state_dict(state_dict['model_state_dict'],strict=False)
model.eval().to(device)

outputs,image,forward_time = model_prediction(args.input_image,args.image_size, model,device)
labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

mask = read_mask(args.true_label,args.image_size)
true_mask = label_mask(mask)



data1,data2 = get_recall_and_precision(true_mask,labels)
print(data1)
print(data2)





import torch
import argparse
import cv2 as cv
import os
from utils import get_segment_labels
import numpy as np
from models.MACUNet import  MACUNet
"""from models.Student_MACUNet import MACUNet"""

from dataloader import label_color_list


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


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_image',default=f"dataset/images/val/image_0.jpg",help='path to input dir')
parser.add_argument('--output_name',default="/final_map.jpg",help='output image name')
parser.add_argument('--model',default='outputs/best_model_iou (4).pth',help='path to the model checkpoint')
parser.add_argument('--image_size',default=256,help='image resize resolution')
parser.add_argument('--show',action='store_true',help='whether or not to show the output as inference is going')
args = parser.parse_args()

args.input_image = "dataset/idd/val/images/0001256_leftImg8bit.jpg"

"""epoch_num = 22
out_dir = os.path.join('outputs', 'inference_results')
os.makedirs(out_dir, exist_ok=True)
device = torch.device('cpu')
model = MACUNet(3, len(label_color_list)).to(device)

state_dict = torch.load(f"best_model/checkpoint_6.pt",weights_only=False)
model.load_state_dict(state_dict['model_state_dict'],strict=False)

model.eval().to(device)

outputs,image,forward_time = model_prediction(args.input_image,args.image_size, model,device)
print(f"forward time inference:{round(forward_time,3)}")
segmentation_map = get_prediction_map(outputs)
output_final = cv.cvtColor(segmentation_map,cv.COLOR_RGB2BGR)

cv.imwrite(f"outputs/inference_results/out_new_2.png",output_final)"""


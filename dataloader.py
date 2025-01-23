import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from pandas.core.internals.construction import nested_data_to_arrays
from torch.utils.data import Dataset, DataLoader
import albumentations as A



all_classes = [
    'class_0',
    'class_1',
    'class_2',
    'class_3',
    'class_4',
    'class_5',
    'class_6',
    'class_7',
    'class_8',
    'class_9',
    'class_10',
    'class_11',
    'class_12',
    'class_13',
    'class_14',
    'class_15',
    'class_16',
    'class_17',
    'class_18',
    'class_19',
    'class_20',
    'class_21',
    'class_22',
    'class_23',
    'class_24',
    'class_25',
    'class_26'
]

label_color_list = [
    (128, 64, 128),
    (250, 170, 160),
    (244, 35, 232),
    (230, 150, 140),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 230),
    (119, 11, 32),
    (255, 204, 54),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 0, 90),
    (220, 190, 40),
    (102, 102, 156),
    (190, 153, 153),
    (180, 165, 180),
    (174, 64, 67),
    (220, 220, 0),
    (250, 170, 30),
    (153, 153, 153),
    (169, 187, 214),
    (70, 70, 70),
    (150, 100, 100),
    (107, 142, 35),
    (70, 130, 180),
    (0, 0, 0),
]

def set_class_values(all_classes, classes_to_train):

    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):

    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

def get_images(root_path):


    train = glob.glob(f"{root_path}/train/images/*")
    train.sort()

    train_labels = glob.glob(f"{root_path}/train/rgb_labels/*")
    train_labels.sort()

    test = glob.glob(f"{root_path}/val/images/*")
    test.sort()

    test_labels = glob.glob(f"{root_path}/val/rgb_labels/*")
    test_labels.sort()

    return train, train_labels, test, test_labels

def train_transforms(height, width):

    train_image_transform = A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomSunFlare(p=0.2),
        A.RandomFog(p=0.2),
        A.Rotate(limit=25),
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(height, width):
    valid_image_transform = A.Compose([
        A.Resize(height, width),
    ], is_check_shapes=False)
    return valid_image_transform


def get_dataset(train_image_paths,train_mask_paths,valid_image_paths,valid_mask_paths,
                label_color_list,all_classes,height, width):

    train_tfms = train_transforms(height, width)
    valid_tfms = valid_transforms(height, width)

    train_dataset = SegmentationDataset(train_image_paths,train_mask_paths, train_tfms,label_color_list,all_classes)
    valid_dataset = SegmentationDataset(valid_image_paths,valid_mask_paths,valid_tfms,label_color_list,all_classes)

    return train_dataset, valid_dataset


def get_data_loaders(train_dataset,val_dataset, batch_size):

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
    return train_data_loader, valid_data_loader

class SegmentationDataset(Dataset):
    def __init__(self,image_paths,mask_paths,transform,label_color_list,all_classes):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform =  transform
        self.colors=label_color_list
        self.all_classes=all_classes
        self.class_values= set_class_values(all_classes,all_classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv.cvtColor(cv.imread(self.image_paths[index],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB).astype('float32')/ 255.0

        mask = cv.cvtColor( cv.imread(self.mask_paths[index], cv.IMREAD_COLOR),cv.COLOR_BGR2RGB).astype('int32')


        transformed = self.transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        label = get_label_mask(mask, self.class_values, self.colors)

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return image, label




"""root_path="dataset/indian_dataset/"
train, train_labels, test, test_labels=get_images(root_path)
train_dataset, valid_dataset=get_dataset(train,train_labels,test,test_labels,label_color_list,all_classes,256,256)
train_data_loader, valid_data_loader=get_data_loaders(train_dataset, valid_dataset,16)

image,label = next(iter(train_data_loader))

print(label.shape)
mask= label[8]
plt.imshow(mask)
plt.show()"""


# model_compression_img_seg

## Model network

The model is based on U-Net architecture, it well_known for its use in image segmentation tasks it combines a contracting path (encoder) with an expansive path (decoder) with help of skip
connection between layers model can preserve spatial information during reconstruction.

For the multi scale skip connections ,the features maps are extracted at different scales using
various filter sizes and it can be achieved by applying convolutions with different kernel sizes ,
in other words it link layers that operate at different scales , for example feature maps from
deeper layer in the encoder ( which has low resolution but more semantic information ) can be
connected to a shallower layer in the decoder (which has higher resolution but less semantic
information).

For the attention mechanisms it’s method applied on the skip connection to filter out irrelevant
information before merging features from the encoder and decoder paths. These gates allow the
network to focus on relevant regions in the input image, enhancing segmentation prediction


![image](https://github.com/user-attachments/assets/0b764940-3a63-4515-a9fa-e15b3963a79d)

In the second part, we will train a smaller model and leverage knowledge transfer from the larger model to compress it. This approach enables efficient deployment on cutting-edge devices while maintaining high performance.


![téléchargement](https://github.com/user-attachments/assets/3bc0ba10-4954-4807-bb16-ed1940cc47ed)

the knowledge from big model will be transformed due to loss function 

<img width="866" alt="téléchargement" src="https://github.com/user-attachments/assets/9ca2e168-505c-4123-9a13-3f1104306086" />


## Dataset

Indian Driving Dataset (IDD) is for autonomous driving research in Indian traffic,it contains 27 classes, many are underrepresented.

Original image size: 1920x1080,reduced to 256x256 for training,that will lose fine details,especially for small objects.

Link: https://www.kaggle.com/datasets/sovitrath/indian-driving-dataset-segmentation-part-2/data


## Training Results

### Performance Metrics ( Teacher model ) 

| Class   | Dice  | IoU  | Precision | Recall |
|---------|-------|------|-----------|--------|
| class_0  | 0.957  | 0.918 | 0.954 | 0.960  |
| class_1  | 0.414  | 0.261 | 0.761 | 0.284  |
| class_2  | 0.961  | 0.925 | 0.983 | 0.939  |
| class_5  | 0.234  | 0.132 | 0.346 | 0.176  |
| class_6  | 0.362  | 0.221 | 0.950 | 0.224  |
| class_8  | 0.588  | 0.417 | 0.816 | 0.460  |
| class_9  | 0.944  | 0.895 | 0.920 | 0.970  |
| class_12 | 0.333  | 0.200 | 0.250 | 0.500  |
| class_13 | 0.372  | 0.229 | 0.258 | 0.667  |
| class_14 | 0.262  | 0.151 | 0.591 | 0.168  |
| class_17 | 0.021  | 0.011 | 0.143 | 0.011  |
| class_20 | 0.080  | 0.041 | 0.119 | 0.060  |
| class_22 | 0.814  | 0.686 | 0.760 | 0.877  |
| class_24 | 0.943  | 0.892 | 0.967 | 0.920  |
| class_25 | 0.973  | 0.948 | 0.954 | 0.994  |
| **Mean** | **0.551** | **0.462** | **0.651** | **0.547** |

### Segmentation Visualizations ( Teacher model ) 

| Input Image          | Ground Truth         | Predicted Mask       |
|----------------------|----------------------|----------------------|
| ![img_1](https://github.com/user-attachments/assets/4b9540b1-f2ea-4f32-842f-57c9c703caa4)| ![mask_1](https://github.com/user-attachments/assets/618f420d-7da0-44ce-aca2-83dc1845307e)| ![out_3](https://github.com/user-attachments/assets/c5f15d0e-9e6b-49a8-935a-3aa4a44fe225)|
| ![img_2](https://github.com/user-attachments/assets/e60c262a-d2e1-4487-bd0d-dc3afd830a2d)| ![mask_](https://github.com/user-attachments/assets/c0281f77-6e5a-4730-bfe1-c7674f2f4e6d)| ![out_new_2](https://github.com/user-attachments/assets/0bd1648d-9da2-496f-b96b-06df90a4fd8e)|

## Train (Teacher model)

To synthesize results, run the following command:
```bash
python train.py --root_path "dataset/idd/" --img_size 128 --lr 1e-3 --epochs 150 --batch 8
```bash


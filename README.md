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

## Dataset

Indian Driving Dataset (IDD) is for autonomous driving research in Indian traffic,it contains 27 classes, many are underrepresented.

Original image size: 1920x1080,reduced to 256x256 for training,that will lose fine details,especially for small objects.

Link: https://www.kaggle.com/datasets/sovitrath/indian-driving-dataset-segmentation-part-2/data


## Training Results

### Performance Metrics

| Class      | Accuracy | IoU  | Precision | Recall | F1-Score |
|------------|----------|------|-----------|--------|----------|
| Road       | 0.92     | 0.85 | 0.91      | 0.93   | 0.92     |
| Building   | 0.89     | 0.82 | 0.88      | 0.90   | 0.89     |
| Vegetation | 0.87     | 0.80 | 0.86      | 0.88   | 0.87     |
| **Overall**| **0.90** | **0.83** | **0.89** | **0.91** | **0.90** |

### Segmentation Visualizations

| Input Image          | Ground Truth         | Predicted Mask       |
|----------------------|----------------------|----------------------|
| ![img_1](https://github.com/user-attachments/assets/4b9540b1-f2ea-4f32-842f-57c9c703caa4)| ![mask_1](https://github.com/user-attachments/assets/618f420d-7da0-44ce-aca2-83dc1845307e)| ![out_3](https://github.com/user-attachments/assets/c5f15d0e-9e6b-49a8-935a-3aa4a44fe225)|
| ![img_2](https://github.com/user-attachments/assets/e60c262a-d2e1-4487-bd0d-dc3afd830a2d)| ![mask_](https://github.com/user-attachments/assets/c0281f77-6e5a-4730-bfe1-c7674f2f4e6d)| ![out_new_2](https://github.com/user-attachments/assets/0bd1648d-9da2-496f-b96b-06df90a4fd8e)|


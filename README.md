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


## Results 

### Segmentation Visualizations

| Input Image          | Ground Truth         | Predicted Mask       |
|----------------------|----------------------|----------------------|
| ![Input](images/segmentation_example1_input.png) | ![GT](images/segmentation_example1_gt.png) | ![Pred](images/segmentation_example1_pred.png) |
| ![Input](images/segmentation_example2_input.png) | ![GT](images/segmentation_example2_gt.png) | ![Pred](images/segmentation_example2_pred.png) |
| ![Input](images/segmentation_example3_input.png) | ![GT](images/segmentation_example3_gt.png) | ![Pred](images/segmentation_example3_pred.png) |

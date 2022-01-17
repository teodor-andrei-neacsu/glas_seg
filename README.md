# Image Segmentation for GlaS dataset

To generate the .npy files that will be used for training and testing:
  python get_datasets.py

- In these files will be the original dataset:
"train_img.npy"
"train_mask_img.npy"
"test_img.npy"
"test_mask_img.npy"

- In these files will be the augmented dataset (zoom, horizontal_flip, vertical_flip):
"train_img_aug.npy"
"train_mask_img_aug.npy"

To train the U-Net model:
  python train_unet.py

To test the U-Net model:
  python test_unet.py [show_samples]

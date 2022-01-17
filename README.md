# Image Segmentation for GlaS dataset

GlaS dataset : https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/

K. Sirinukunwattana, J. P. W. Pluim, H. Chen, X Qi, P. Heng, Y. Guo, L. Wang, B. J. Matuszewski, E. Bruni, U. Sanchez, A. Böhm, O. Ronneberger, B. Ben Cheikh, D. Racoceanu, P. Kainz, M. Pfeiffer, M. Urschler, D. R. J. Snead, N. M. Rajpoot, "Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest"

## To generate the .npy files that will be used for training and testing:

    python npy_glas_dataset.py

- In these .npy files will be the original dataset:
    - "train_img.npy"
    - "train_mask_img.npy"
    - "test_img.npy"
    - "test_mask_img.npy"

- In these .npy files will be the augmented dataset (zoom, horizontal_flip, vertical_flip):
    - "train_img_aug.npy"
    - "train_mask_img_aug.npy"

## To train the U-Net model:

    python train_unet.py train

## To test the U-Net model:

    python train_unet.py test

## To plot the training process (Jaccard index, Dice score and loss):

    python train_unet.py plot

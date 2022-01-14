import os
import re
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# resize the images to 512 x 768
img_h = 512
img_w = 768

def get_glas_data(ds_path, cwd_path):
    os.chdir(ds_path)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    train_no_list = range(1, 86)
    for img_no in train_no_list:

        x_img_raw = cv2.imread("train_" + str(img_no) + ".bmp")
        y_img_raw = cv2.imread("train_" + str(img_no) + "_anno.bmp", cv2.IMREAD_GRAYSCALE)

        x_img = cv2.resize(x_img_raw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        y_img = cv2.resize(y_img_raw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)

        y_img[y_img != 0] = 255.0

        X_train.append(np.array(x_img))
        y_train.append(np.array(y_img))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    testa_no_list = range(1, 61)
    for img_no in testa_no_list:
        x_img_raw = cv2.imread("testA_" + str(img_no) + ".bmp")
        y_img_raw = cv2.imread("testA_" + str(img_no) + "_anno.bmp", cv2.IMREAD_GRAYSCALE)

        x_img = cv2.resize(x_img_raw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        y_img = cv2.resize(y_img_raw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)

        y_img[y_img != 0] = 255.0

        X_test.append(np.array(x_img))
        y_test.append(np.array(y_img))


    testb_no_list = range(1, 21)
    for img_no in testb_no_list:
        x_img = cv2.imread("testB_" + str(img_no) + ".bmp")
        y_img = cv2.imread("testB_" + str(img_no) + "_anno.bmp", cv2.IMREAD_GRAYSCALE)

        # cv2.imshow("x_train sample", x_img)
        # cv2.waitKey(0)

        x_img = cv2.resize(x_img, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        y_img = cv2.resize(y_img, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)

        y_img[y_img != 0] = 255.0

        X_test.append(np.array(x_img))
        y_test.append(np.array(y_img))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    os.chdir(cwd_path)

    return X_train, np.expand_dims(y_train, axis=3), X_test, np.expand_dims(y_test, axis=3)


# Testing get_*_data
cwd = os.getcwd()
glas_path = "GlaS/"
X_train, y_train, X_test, y_test = get_glas_data(glas_path, cwd)

print("train_img shape:", X_train.shape)
print("train_mask_img shape:", y_train.shape)
print("test_img shape:", X_test.shape)
print("test_mask_img shape:", y_test.shape)

np.save("train_img.npy", X_train)
np.save("train_mask_img.npy", y_train)
np.save("test_img.npy", X_test)
np.save("test_mask_img.npy", y_test)

data_aug = True

if data_aug:
    # Data augmentation object
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        )

    train_img_aug = []
    train_mask_img_aug = []

    # Data augmentation
    aug_no = 4
    total = X_train.shape[0]
    count = 0
    for aug_img in datagen.flow(X_train, batch_size=1, seed=1337):
        train_img_aug.append(aug_img.astype('uint8'))
        count += 1
        if count == total*aug_no:
            break

    count = 0
    for aug_img in datagen.flow(y_train, batch_size=1, seed=1337):
        train_mask_img_aug.append(aug_img.astype('uint8'))
        count += 1
        if count == total*aug_no:
            break

    train_img_aug = np.array(train_img_aug)[:, 0]
    train_mask_img_aug = np.array(train_mask_img_aug)[:, 0]

    print("train_img_aug shape", train_img_aug.shape)
    print("train_mask_img_aug shape", train_mask_img_aug.shape)

    np.save('train_img_aug.npy', train_img_aug)
    np.save('train_mask_img_aug.npy', train_mask_img_aug)

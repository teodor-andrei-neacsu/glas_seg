import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import optim
from scipy import stats
from torchvision import transforms
from u_net_model import *
from sklearn import metrics

def main():

    plot_results = True
    if plot_results:
        jacc_ = np.load("jacc_aug.npy")
        loss_ = np.load("loss_aug.npy")
        dice_ = np.load("dice_aug.npy")
        sample_count = range(len(jacc_))

        plt.figure()
        plt.plot(sample_count, jacc_)
        plt.plot(sample_count, loss_)
        plt.plot(sample_count, dice_)
        plt.show()

        print("+" * 30)
        print("Jacc Test: ", np.mean(jacc_))
        print("Dice Test: ", np.mean(dice_))



        return


    show_sample = True
    train_unet = False

    print("-" * 20 + "Loading dataset" + "-" * 20)

    # Load original GlaS dataset reshaped to
    X_train = np.load('train_img_aug.npy')
    y_train = np.load('train_mask_img_aug.npy') / 255
    X_test = np.load("test_img.npy")
    y_test = np.load("test_mask_img.npy") / 255

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    if show_sample:
        sample_idx = 70
        cv2.imshow("X_train sample", X_train[sample_idx])
        cv2.imshow("y_train sample", y_train[sample_idx])
        cv2.waitKey(0)

    print("=" * 20 + "Loaded dataset" + "=" * 20 + "\n")

    print("-" * 20 + "Preprocess dataset" + "-" * 20)

    print("Convert from BHWC to BCHW")
    X_train = torch.Tensor(X_train.transpose((0, 3, 1, 2)))
    y_train = torch.Tensor(y_train.transpose((0, 3, 1, 2)).squeeze()).type(torch.LongTensor)
    X_test = torch.Tensor(X_test.transpose((0, 3, 1, 2)))
    y_test = torch.Tensor(y_test.transpose((0, 3, 1, 2)).squeeze()).type(torch.LongTensor)

    # compute mean and std
    X_train_md = X_train.view(X_train.shape[0], X_train.shape[1], -1)
    print("X_train_md shape:", X_train_md.shape)

    X_train_mean = X_train_md.mean(2).sum(0) / X_train.shape[0]
    X_trian_std = X_train_md.std(2).sum(0) / X_train.shape[0]

    print("DS mean:", X_train_mean)
    print("DS std:", X_trian_std)

    # normalize train and test sets
    norm_transform = transforms.Normalize(X_train_mean, X_trian_std)
    X_train = norm_transform(X_train)
    X_test = norm_transform(X_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # show sample after normalization
    if show_sample:
        sample_idx = 70
        cv2.imshow("X_train sample", X_train[sample_idx].numpy().transpose((1, 2, 0)))
        # cv2.imshow("y_train sample", y_train[sample_idx].numpy().transpose((1, 2, 0)))
        cv2.imshow("y_train sample", y_train[sample_idx].numpy().astype(np.float32))
        cv2.waitKey(0)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    print("=" * 20 + "Preprocessed dataset" + "=" * 20 + "\n")

    print("-" * 20 + "Training U-Net" + "-" * 20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 50
    batch_size = 1
    learning_rate = 0.0005

    print("Using device: ", device)
    print("Number of epochs: ", num_epochs)
    print("Batch size: ", batch_size)
    print("Learning_rate: ", learning_rate)

    # Init Dataloaders for training and testing
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Init U-Net
    u_net = UNet(n_channels=3, n_classes=2, bilinear=True)

    # Model to device
    u_net.to(device=device)

    if train_unet:

        # Optimizer and Criterion
        optimizer = optim.Adam(u_net.parameters(),
                                  lr=learning_rate,
                                  weight_decay=1e-4
                                 )
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        criterion = nn.CrossEntropyLoss()

        jacc_list = []
        dice_list = []
        loss_list = []

        for epoch in range(num_epochs):

            batch_loss_list = []
            batch_jacc_list = []
            batch_dice_list = []

            for train_idx, (images, true_masks) in enumerate(train_dl):

                if show_sample:
                    cv2.imshow("X_train sample", images[0].numpy().transpose((1, 2, 0)))
                    # cv2.imshow("y_train sample", y_train[sample_idx].numpy().transpose((1, 2, 0)))
                    cv2.imshow("y_train sample", true_masks[0].numpy().astype(np.float32))
                    cv2.waitKey(0)

                images = images.to(device=device)
                true_masks = true_masks.to(device=device)

                # Zero previous gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output_masks = u_net(images)

                loss = criterion(output_masks, true_masks)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Get the jaccard index and dice score
                train_out_mask = output_masks.argmax(dim=1)

                flat_out_mask = train_out_mask[0].cpu().detach().numpy().flatten()
                flat_true_mask = true_masks[0].cpu().detach().numpy().flatten()

                jacc_idx = metrics.jaccard_score(flat_out_mask, flat_true_mask)
                dice_s = 2 * jacc_idx /(jacc_idx + 1)

                batch_loss_list.append(loss.item())
                batch_jacc_list.append(jacc_idx)
                batch_dice_list.append(dice_s)


                # Average jaccard list and dice list
                if train_idx % 8 == 0:

                    loss_b = np.mean(np.array(batch_loss_list))
                    jacc_idx_b = np.mean(np.array(batch_jacc_list))
                    dice_s_b = np.mean(np.array(batch_dice_list))

                    print("Train img: " + str(train_idx))
                    print("Loss img:" + str(loss_b))
                    print("Jaccard index: " + str(jacc_idx_b))
                    print("Dice score:    " + str(dice_s_b))
                    print("=" * 30 + "\n")

                    jacc_list.append(jacc_idx_b)
                    dice_list.append(dice_s_b)
                    loss_list.append(loss_b)

                    batch_loss_list = []
                    batch_jacc_list = []
                    batch_dice_list = []


        np.save("jacc_aug.npy", np.array(jacc_list))
        np.save("dice_aug.npy", np.array(dice_list))
        np.save("loss_aug.npy", np.array(loss_list))

        torch.save(u_net.state_dict(), str(num_epochs) + "ep_" + str(batch_size) + "bs_aug.pth")
    else:

        u_net.load_state_dict(torch.load("./50ep_1bs_aug.pth"))

        for test_idx, (test_img, test_mask) in enumerate(test_dl):

            test_img = test_img.to(device=device)

            test_output_masks = u_net(test_img)

            test_output_masks = test_output_masks.argmax(dim=1)

            flat_out_mask = test_output_masks[0].cpu().detach().numpy().flatten()
            flat_true_mask = test_mask[0].cpu().detach().numpy().flatten()

            jacc_idx = metrics.jaccard_score(flat_out_mask, flat_true_mask)
            dice_s = 2 * jacc_idx /(jacc_idx + 1)

            print("Test img: " + str(test_idx))
            print("Jaccard index: " + str(jacc_idx))
            print("Dice score:    " + str(dice_s))
            print("_____________")

            if show_sample:
                cv2.imshow("X_test", test_img[0].cpu().detach().numpy().transpose((1, 2, 0)))
                cv2.imshow("y_out sample", test_output_masks[0].cpu().detach().numpy().astype(np.float32))
                cv2.imshow("y_out_test sample", test_mask[0].numpy().astype(np.float32))
                cv2.waitKey(0)


if __name__ == '__main__':
    main()

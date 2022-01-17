import os
import cv2
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import optim
from torch import Tensor
from torchvision import transforms
from u_net_model import *
from sklearn import metrics

def load_preprocess_dataset(show_sample=False):

    print("-" * 20 + "Loading dataset" + "-" * 20)

    # Load original GlaS dataset reshaped to
    X_train = np.load('train_img.npy')
    y_train = np.load('train_mask_img.npy') / 255
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

    # Convert from BHWC to BCHW"
    X_train = torch.Tensor(X_train.transpose((0, 3, 1, 2)))
    y_train = torch.Tensor(y_train.transpose((0, 3, 1, 2)).squeeze()).type(torch.LongTensor)
    X_test = torch.Tensor(X_test.transpose((0, 3, 1, 2)))
    y_test = torch.Tensor(y_test.transpose((0, 3, 1, 2)).squeeze()).type(torch.LongTensor)

    # compute mean and std
    X_train_md = X_train.view(X_train.shape[0], X_train.shape[1], -1)

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
        cv2.imshow("y_train sample", y_train[sample_idx].numpy().astype(np.float32))
        cv2.waitKey(0)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    print("=" * 20 + "Preprocessed dataset" + "=" * 20 + "\n")

    return train_ds, test_ds


def train(u_net, train_dl, device, batch_size):

    show_sample = False
    num_epochs = 30
    learning_rate = 0.0005

    print("-" * 20 + "Training U-Net" + "-" * 20)
    print("Using device: ", device)
    print("Number of epochs: ", num_epochs)
    print("Batch size: ", batch_size)
    print("Learning_rate: ", learning_rate)

    # Optimizer and Criterion
    optimizer = optim.Adam(u_net.parameters(),
                              lr=learning_rate,
                              weight_decay=1e-4
                             )
    if device == torch.device("cuda"):
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()

    batch_loss_list = []
    batch_jacc_list= []
    batch_dice_list = []

    train_jacc = []
    train_dice = []
    train_loss = []

    for epoch in range(num_epochs):

        for train_batch_idx, (images, true_masks) in enumerate(train_dl):

            # Show the first image of each batch
            if show_sample:
                cv2.imshow("X_train sample", images[0].numpy().transpose((1, 2, 0)))
                cv2.imshow("y_train sample", true_masks[0].numpy().astype(np.float32))
                cv2.waitKey(0)

            images = images.to(device=device)
            true_masks = true_masks.to(device=device)

            # Zero previous gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_masks = u_net(images)

            loss = criterion(output_masks, true_masks)

            if device == torch.device("cuda"):
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Get the jaccard index and dice score
            train_out_mask = output_masks.argmax(dim=1)

            dice_s = 0
            jacc_s = 0

            curr_bs = len(train_out_mask.cpu().detach().numpy())

            for img_idx in range(curr_bs):
                flat_out_mask = train_out_mask[img_idx].cpu().detach().numpy().flatten()
                flat_true_mask = true_masks[img_idx].cpu().detach().numpy().flatten()

                jacc_idx = metrics.jaccard_score(flat_out_mask, flat_true_mask)

                jacc_s += jacc_idx
                dice_s += 2 * jacc_idx /(jacc_idx + 1)

            dice_s /= curr_bs
            jacc_s /= curr_bs

            print("Epoch = ", epoch)
            print("Batch = ", train_batch_idx)
            print("Dice score", dice_s)
            print("Jacc score", jacc_s)

            # save the training Jacc and Dice of the last epoch
            if epoch == num_epochs - 1:
                train_loss.append(loss.item())
                train_jacc.append(jacc_s)
                train_dice.append(dice_s)

            batch_loss_list.append(loss.item())
            batch_jacc_list.append(jacc_s)
            batch_dice_list.append(dice_s)


    np.save("loss_train_aug.npy", np.array(batch_loss_list))
    np.save("jacc_train_aug.npy", np.array(batch_jacc_list))
    np.save("dice_train_aug.npy", np.array(batch_dice_list))

    print("TRAIN LOSS:          ", np.mean(np.array(train_loss)))
    print("TRAIN JACCARD SCORE: ", np.mean(np.array(train_jacc)))
    print("TRAIN DICE SCORE:    ", np.mean(np.array(train_dice)))

    torch.save(u_net.state_dict(), "dummy.pth")

def test(u_net, test_dl, device, batch_size):

    show_sample = True

    u_net.load_state_dict(torch.load("./trained_unet.pth"))

    test_jacc = []
    test_dice = []

    for test_batch_idx, (test_img, test_mask) in enumerate(test_dl):

        test_img = test_img.to(device=device)

        test_output_masks = u_net(test_img)

        test_output_masks = test_output_masks.argmax(dim=1)

        curr_bs = len(test_output_masks.cpu().detach().numpy())

        # Compute Jaccard and Dice score
        for img_idx in range(curr_bs):
            flat_out_mask = test_output_masks[img_idx].cpu().detach().numpy().flatten()
            flat_true_mask = test_mask[img_idx].cpu().detach().numpy().flatten()

            jacc_s = metrics.jaccard_score(flat_out_mask, flat_true_mask)
            dice_s = 2 * jacc_s /(jacc_s + 1)

            print("Jaccard index: " + str(jacc_s))
            print("Dice score:    " + str(dice_s))
            print("_____________")

            test_jacc.append(jacc_s)
            test_dice.append(dice_s)

            if show_sample:
                cv2.imshow("X_test", test_img[img_idx].cpu().detach().numpy().transpose((1, 2, 0)))
                cv2.imshow("y_out sample", test_output_masks[img_idx].cpu().detach().numpy().astype(np.float32))
                cv2.imshow("y_out_test sample", test_mask[img_idx].numpy().astype(np.float32))
                cv2.waitKey(0)

    test_jacc = np.array(test_jacc)
    test_dice = np.array(test_dice)

    print("=" * 30)
    print("TEST JACCARD: ", np.mean(test_jacc))
    print("TEST DICE:    ", np.mean(test_dice))

def plot():

    print("-" * 20 + "Plotting training" + "-" * 20)

    loss_ = np.load("loss_train_aug.npy")
    jacc_ = np.load("jacc_train_aug.npy")
    dice_ = np.load("dice_train_aug.npy")

    sample_cnt = len(loss_) // 8

    jacc_ = np.array([l.mean() for l in np.split(jacc_, sample_cnt)])
    dice_ = np.array([l.mean() for l in np.split(dice_, sample_cnt)])
    loss_ = np.array([l.mean() for l in np.split(loss_, sample_cnt)])

    sample_count = range(len(jacc_))

    plt.figure()
    plt.title("Training")
    plt.ylabel("Value")
    plt.gca().set_ylim(0, 1)
    plt.xlabel("Iterations")
    plt.plot(sample_count, jacc_, label='Jaccard Index - IoU')
    plt.plot(sample_count, loss_, label='Loss')
    plt.plot(sample_count, dice_, label='Dice Score - F1')
    plt.legend()
    plt.show()

    print("=" * 20 + "Plotting done" + "=" * 20)

def main():

    batch_size = 1

    train_ds, test_ds = load_preprocess_dataset()

    # Init Dataloaders for training and testing
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # Init U-Net
    u_net = UNet(n_channels=3, n_classes=2, bilinear=True)

    # Load pretrained model
    if len(sys.argv) > 2 and sys.argv[2] == "pretrained":
        u_net.load_state_dict(torch.load("./trained_unet.pth"))

    # Model to device
    u_net.to(device=device)

    if sys.argv[1] == "train":
        train(u_net, train_dl, device, batch_size)

    if sys.argv[1] == "test":
        test(u_net, test_dl, device, batch_size)

    if sys.argv[1] == "plot":
        plot()

if __name__ == '__main__':
    main()

import os

import json

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt

import resnet3d
from opts import parse_opts
from dataLoader import NiftiDataset
from torch.utils.data import random_split


def determinant(matrix):
    return torch.linalg.det(matrix)


def is_orthogonal(matrix):
    batch_size, _, _ = matrix.shape
    identity = torch.eye(3, device=matrix.device).expand(batch_size, 3, 3)
    return torch.nn.functional.mse_loss(torch.bmm(matrix.transpose(1, 2), matrix), identity)


def train(model,
          traindataloader,
          validdataloader,
          device,
          num_epochs=30,
          learning_rate=0.0007,
          mode=0,
          alpha=1): # původní learning_rate: 0.001

    # setting of criterion and optimization algorithm:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # SGD

    # initialization of supportive variables for control of error development:
    average_loss_train = []
    average_loss_valid = []


    for epoch in range(num_epochs):
        # training:
        model.train()
        running_batch_loss_train = 0.0
        running_epoch_loss_train = 0.0
        for i, (data, ground_truth) in enumerate(traindataloader):

            # Forward:
            outputs = model(data.to(device))  # outputs: (batch_size=4, 3, 3)

            if mode == 0:
                loss = criterion(outputs, ground_truth.to(device))
            elif mode == 1:
                # Ztátové funkce:
                # z predikované matice
                loss_A = criterion(outputs, ground_truth.to(device))
                # z determinantu (normalita)
                # loss_norm = criterion(determinant(outputs.view(-1, 3, 3)).view(-1, 1), torch.ones((outputs.shape[0], 1), device=device))
                # z maticového součinu (ortogonalita)
                loss_orto = is_orthogonal(outputs.view(-1, 3, 3))  # MSE ortogonality

                gamma = loss_A / (loss_orto + 1e-6)

                # celková ztráta:
                loss = alpha * loss_A + gamma * loss_orto

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_batch_loss_train += loss.item()
            running_epoch_loss_train += loss.item()

            if (i+1) % 4 == 0:  # Print every 5 mini-batches
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(traindataloader), running_batch_loss_train / 4))
                running_batch_loss_train = 0.0

        # validation:
        model.eval()
        running_epoch_loss_valid = 0.0
        with torch.no_grad():
            for j, (data, ground_truth) in enumerate(validdataloader):
                outputs = model(data.to(device))
                if mode == 0:
                    loss = criterion(outputs, ground_truth.to(device))
                elif mode == 1:
                    # Ztátové funkce:
                    # z predikované matice
                    loss_A = criterion(outputs, ground_truth.to(device))
                    # z determinantu (normalita)
                    # loss_norm = criterion(determinant(outputs.view(-1, 3, 3)).view(-1, 1),
                                          # torch.ones((outputs.shape[0], 1), device=device))
                    # z maticového součinu (ortogonalita)
                    loss_orto = is_orthogonal(outputs.view(-1, 3, 3))  # MSE ortogonality

                    gamma = loss_A / (loss_orto + 1e-6)

                    # celková ztráta:
                    loss = alpha * loss_A + gamma * loss_orto # + beta * loss_norm

                running_epoch_loss_valid += loss.item()

        average_loss_train.append(running_epoch_loss_train/len(traindataloader)) # původně děleno i
        average_loss_valid.append(running_epoch_loss_valid/len(validdataloader)) # původně děleno j
        # plt.figure(1)
        # plt.plot(np.arange(1, epoch+2), np.asarray(average_loss_train))
        # plt.plot(np.arange(1, epoch+2), np.asarray(average_loss_valid))
        # plt.xlabel('Epocha')
        # plt.ylabel('Kriteriální funkce')
        # plt.show()
    plt.figure()
    plt.plot(np.arange(1, num_epochs + 1), np.asarray(average_loss_train))
    plt.plot(np.arange(1, num_epochs + 1), np.asarray(average_loss_valid))
    plt.xlabel('Epocha')
    plt.ylabel('Průměr MSE')
#    plt.savefig('trenink_augmentace02.png')
    plt.show()

    torch.save(model, r'D:\DataSet\Data\model34_ao.pth')
        

def test(model, testdataloader, device, mode=0, alpha=1):

    model.eval()
    criterion = nn.MSELoss()

    results = []

    total_loss = 0.0
    with torch.no_grad():
        for i, (data, ground_truth) in enumerate(testdataloader):
            outputs = model(data.to(device))

            if mode == 0:
                loss = criterion(outputs, ground_truth.to(device))
            elif mode == 1:
                # Ztátové funkce:
                # z predikované matice
                loss_A = criterion(outputs, ground_truth.to(device))
                # z determinantu (normalita)
                # loss_norm = criterion(determinant(outputs.view(-1, 3, 3)).view(-1, 1),
                #                       torch.ones((outputs.shape[0], 1), device=device))
                # z maticového součinu (ortogonalita)
                loss_orto = is_orthogonal(outputs.view(-1, 3, 3))  # MSE ortogonality

                gamma = loss_A / (loss_orto + 1e-6)

                # celková ztráta:
                loss = alpha * loss_A + gamma * loss_orto

            total_loss += loss.item()

            gt_matrix = ground_truth.view(-1, 3, 3).cpu().numpy()
            out_matrix = outputs.view(-1, 3, 3).cpu().numpy()

            print(f'Provedeno testování na {i+1}/{len(testdataloader)}.')

            for j in range(gt_matrix.shape[0]):
                results.append({
                    "ground_truth": gt_matrix[j].tolist(),
                    "predicted": out_matrix[j].tolist()
                })

    average_loss = total_loss / len(testdataloader)
    print(f'Průměrná chyba na testovací sadě je: {average_loss}')
    return average_loss, results


if __name__ == "__main__":
    # loading of opts parameters:
    opt = parse_opts()
    opt.dir = r'D:\DataSet\Data'
    transform_info_file_train = os.path.join(opt.dir, 'trainInfo.json')
    transform_info_file_valid = os.path.join(opt.dir, 'validInfo.json')
    transform_info_file_test = os.path.join(opt.dir, 'testInfo.json')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loading of model:
    model = resnet3d.generate_model(model_depth=opt.model_depth,
                                    n_classes=opt.n_classes).to(device)
    # datasets creation:
    train_dataset = NiftiDataset(root_dir=opt.dir, transform_info_file=transform_info_file_train, resize=opt.resize)
    valid_dataset = NiftiDataset(root_dir=opt.dir, transform_info_file=transform_info_file_valid, resize=opt.resize)
    test_dataset = NiftiDataset(root_dir=opt.dir, transform_info_file=transform_info_file_test, resize=opt.resize)

    # loading of satasets:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    # training of model:
    train(model,
          train_loader,
          valid_loader,
          device,
          num_epochs=opt.num_epochs,
          learning_rate=opt.learning_rate,
          mode=1)

    # testing of model:
    avg_loss, results = test(model,
                             test_loader,
                             device,
                             mode=1)

    # storing of results from testing:
    with open('predictedMatrix_ao.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
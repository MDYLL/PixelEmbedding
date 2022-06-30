import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import tqdm
import os
import cv2
from torch.utils.data import Dataset
import pickle
import shutil
from sklearn.metrics import roc_auc_score
from unet3plus import UNet3Plus

PATH_TO_CELEBA = "/home/dmartynov/shad/celeba/celeba/"
PATH_TO_CELEBA_MASK = "/home/dmartynov/shad/CelebAMask-HQ/CelebA-HQ-img/"

np.random.seed(18)

def get_computing_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

device = get_computing_device()
print(f"Our main computing device is '{device}'")

def load_attributes(file_path):
    annots = {}
    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            values = line.split()
            annots[values[0]] = np.array(list(map(float, values[1:])))
    return annots

annots = load_attributes(PATH_TO_CELEBA + "list_attr_celeba.txt")
attributes_list = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']
segm_classes = ['cloth', 'hair', 'mouth', 'neck', 'nose', 'skin', 'l_eye', 'r_eye', 'l_lip', 'u_lip']


def get_paths(path, first, last, max_ident=None):
    _, _, filenames = next(os.walk(path))

    images_paths = []
    if max_ident:
        for filename in sorted(filenames):
            if identity[filename.split('/')[-1]] < max_ident:
                images_paths.append(os.path.join(path, filename))
    else:
        for filename in sorted(filenames):
            images_paths.append(os.path.join(path, filename))
    images_paths = sorted(images_paths)
    # print(np.stack(images_paths[first:last]))
    return np.stack(images_paths[first:last])

DATASET_QUANTITY = 12
DATASET_QUANTITY_CLASS = 12
DATASET_QUANTITY_SEGM = 0
DATASET_SIZE = 32
DATASET_SIZE_VALIDATION = 32
train_datasets = [None] * DATASET_QUANTITY
val_datasets = [None] * DATASET_QUANTITY
nn_modules = [None] * DATASET_QUANTITY
# backbone = UNet3Plus().to(device)

from datasets import CelebaBinaryCalssification, CelebaSegmentation, CelebaBinaryCalssificationPairwise
from nn_modules import Image2VectorWithCE, Image2VectorWithMSE, Image2Image, Image2VectorWithPairwise, Decoder2Vector

for i in range(DATASET_QUANTITY):    
    if i < DATASET_QUANTITY_CLASS:    
        train_images = get_paths(PATH_TO_CELEBA + "img_align_celeba/", 160_000, 160_000 + DATASET_SIZE)
        test_images = get_paths(PATH_TO_CELEBA + "img_align_celeba/", 170_000, 170_000 + DATASET_SIZE_VALIDATION)

        train_datasets[i] = CelebaBinaryCalssificationPairwise(train_images, attributes_list, annots, attributes_list[i + 8])
        val_datasets[i] = CelebaBinaryCalssificationPairwise(test_images, attributes_list, annots, attributes_list[i + 8])
        nn_modules[i] = Image2VectorWithPairwise()
    else:
        j = i - DATASET_QUANTITY_CLASS
        train_images = get_paths(PATH_TO_CELEBA_MASK, j * DATASET_SIZE, j * DATASET_SIZE + DATASET_SIZE)
        test_images = get_paths(PATH_TO_CELEBA_MASK, 25000, 25000 + DATASET_SIZE_VALIDATION)
        train_datasets[i] = CelebaSegmentation(train_images, segm_classes[j])
        val_datasets[i] = CelebaSegmentation(test_images, segm_classes[j])
        nn_modules[i] = Image2Image()          


for i in range(DATASET_QUANTITY):
    nn_modules[i].encoder = UNet3Plus()

for i in range(4, 12):
    nn_modules[i].load_state_dict(torch.load("nn_module_common_wd1e5_12_512_0.pt"))
    nn_modules[i].decoder = Decoder2Vector(num_out=1)

for i in range(4, 11):
    for j in range(i+1, 12):
        print(nn_modules[i].encoder is nn_modules[j].encoder)

batch_size = 2
train_batch_gens = [torch.utils.data.DataLoader(train_datasets[i],
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0) for i in range(DATASET_QUANTITY)]

val_batch_gens = [torch.utils.data.DataLoader(val_datasets[i],
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0) for i in range(DATASET_QUANTITY)]

for i in range(DATASET_QUANTITY):
    # nn_modules[i].load_state_dict(torch.load("nn_module_train_backbone_pairwise1e4_" + str(DATASET_QUANTITY) + '_' + str(DATASET_SIZE) + '_' + str(i) + ".pt"))
    nn_modules[i].to(device)

# was weight_decay=3e-5
optimizers = [torch.optim.Adam(nn_modules[i].parameters(), lr=1e-4, weight_decay=1e-6) for i in range(DATASET_QUANTITY)]
# schedulers = [torch.optim.lr_scheduler.CyclicLR(optimizers[i], base_lr=1e-6, max_lr=1e-3, gamma=0.9, cycle_momentum=False, mode="exp_range") for i in range(DATASET_QUANTITY)]

# for i in range(DATASET_QUANTITY):
    # optimizers[i].load_state_dict(torch.load("optimizer1e4_" + str(DATASET_QUANTITY) + '_' + str(DATASET_SIZE) + '_' + str(i) + ".pt"))

NUM_EPOCHS = 200
train_loss = [list() for _ in range(DATASET_QUANTITY)]
results = [list() for _ in range(DATASET_QUANTITY)]


NUM_EPOCHS = 200
for epoch in range(NUM_EPOCHS):
    print("Epoch", len(results[0]))

    for i in range(DATASET_QUANTITY):
        nn_modules[i].train(True)
        optimizers[i].zero_grad()

    for idx, batch in tqdm.tqdm(enumerate(zip(*train_batch_gens))):
        for i in range(DATASET_QUANTITY_CLASS):

            if i >=4 and i <= 7:
                for p in nn_modules[i].encoder.parameters():
                    p.requires_grad = False
            else:
                for p in nn_modules[i].encoder.parameters():
                    p.requires_grad = True

            X_batch = batch[i][0].to(device)
            y_batch = batch[i][1].to(device)
            positive = nn_modules[i](X_batch)
            negative = nn_modules[i](y_batch)
            loss = nn_modules[i].compute_loss(positive, negative)
            train_loss[i].append(loss.cpu().data.numpy())
            loss.backward()
            optimizers[i].step()
            # schedulers[i].step()
            optimizers[i].zero_grad()
        
        for i in range(DATASET_QUANTITY_CLASS, DATASET_QUANTITY):
            X_batch = batch[i][0].to(device)
            y_batch = batch[i][1].to(device)
            predictions = nn_modules[i](X_batch)
            loss = nn_modules[i].compute_loss(predictions,y_batch)
            train_loss[i].append(loss.cpu().data.numpy())

            loss.backward()
            optimizers[i].step()      
            # schedulers[i].step()
            optimizers[i].zero_grad()                  

    print("train loss")
    for i in range(DATASET_QUANTITY):
        print(i, np.mean(train_loss[i]))
        results[i].append(list())
        results[i][-1].append(np.mean(train_loss[i]))
        train_loss[i] = []

    print("validation loss")
    for i in range(DATASET_QUANTITY_CLASS):
        nn_modules[i].train(False)
        with torch.no_grad():
            for batch in val_batch_gens[i]:
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)

                positive = nn_modules[i](X_batch)
                negative = nn_modules[i](y_batch)
                loss = nn_modules[i].compute_loss(positive, negative)
                train_loss[i].append(loss.cpu().data.numpy())

    for i in range(DATASET_QUANTITY_CLASS, DATASET_QUANTITY):
        nn_modules[i].train(False)
        with torch.no_grad():
            for X_batch, y_batch in val_batch_gens[i]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = nn_modules[i](X_batch)
                loss = nn_modules[i].compute_loss(predictions,y_batch)
                train_loss[i].append(loss.cpu().data.numpy())                  

    for i in range(DATASET_QUANTITY):
        print(i, np.mean(train_loss[i]))
        results[i][-1].append(np.mean(train_loss[i]))
        train_loss[i] = []

    print("train metric")
    for i in range(DATASET_QUANTITY_CLASS):
        nn_modules[i].train(False)
        metric = [[], [], []]
        with torch.no_grad():
            for first, second, target, idx, _ in train_datasets[i]:
                if target:
                    first = first.to(device)
                    y_pred = nn_modules[i](first[None, ...])[:,0]
                else:
                    second = second.to(device)
                    y_pred = nn_modules[i](second[None, ...])[:,0]
                metric[0].append(target)
                metric[1].append(y_pred.cpu().detach().numpy()[0])
                metric[2].append(idx)
        res = roc_auc_score(metric[0], metric[1])
        metric = [[metric[0][i], metric[1][i], metric[2][i]] for i in range(DATASET_SIZE)]
        metric = sorted(metric, key=lambda x: x[1], reverse=True)
        train_datasets[i].opposite_indexes = [None] * DATASET_SIZE
        
        negative_idx = -1
        for el in metric:
            if el[0] == 0:
                negative_idx = el[2]
            elif negative_idx == -1:
                train_datasets[i].opposite_indexes[el[2]] = np.random.choice(train_datasets[i].negative_indexes)
            else:
                train_datasets[i].opposite_indexes[el[2]] = negative_idx
        
        positive_idx = -1
        for el in reversed(metric):
            if el[0] == 1:
                positive_idx = el[2]
            elif positive_idx == -1:
                train_datasets[i].opposite_indexes[el[2]] = np.random.choice(train_datasets[i].positives_indexes)
            else:
                train_datasets[i].opposite_indexes[el[2]] = positive_idx
        
        print(i, res)
        results[i][-1].append(res)

    for i in range(DATASET_QUANTITY_CLASS):
        train_batch_gens[i] = torch.utils.data.DataLoader(train_datasets[i],
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    for i in range(DATASET_QUANTITY_CLASS, DATASET_QUANTITY):
        nn_modules[i].train(False)
        metric = []
        with torch.no_grad():
            for X_batch, y_batch in train_batch_gens[i]:
                X_batch = X_batch.to(device)
                y_pred = nn_modules[i](X_batch)
                y_pred = nn_modules[i].post_processing(y_pred)
                metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
        print(i, np.mean(metric))
        results[i][-1].append(np.mean(metric))                                              

    print("validation metric")
    for i in range(DATASET_QUANTITY_CLASS):
        nn_modules[i].train(False)
        metric = [[], [], []]
        with torch.no_grad():
            for first, second, target, idx, _ in val_datasets[i]:
                if target:
                    first = first.to(device)
                    y_pred = nn_modules[i](first[None, ...])[:,0]
                else:
                    second = second.to(device)
                    y_pred = nn_modules[i](second[None, ...])[:,0]
                metric[0].append(target)
                metric[1].append(y_pred.cpu().detach().numpy()[0])
                metric[2].append(idx)
        res = roc_auc_score(metric[0], metric[1])
        metric = [[metric[0][i], metric[1][i], metric[2][i]] for i in range(DATASET_SIZE_VALIDATION)]
        metric = sorted(metric, key=lambda x: x[1], reverse=True)
        val_datasets[i].opposite_indexes = [None] * DATASET_SIZE_VALIDATION
        
        negative_idx = -1
        for el in metric:
            if el[0] == 0:
                negative_idx = el[2]
            elif negative_idx == -1:
                val_datasets[i].opposite_indexes[el[2]] = np.random.choice(val_datasets[i].negative_indexes)
            else:
                val_datasets[i].opposite_indexes[el[2]] = negative_idx
        
        positive_idx = -1
        for el in reversed(metric):
            if el[0] == 1:
                positive_idx = el[2]
            elif positive_idx == -1:
                val_datasets[i].opposite_indexes[el[2]] = np.random.choice(val_datasets[i].positives_indexes)
            else:
                val_datasets[i].opposite_indexes[el[2]] = positive_idx
        
        print(i, res)
        results[i][-1].append(res)

    
    for i in range(DATASET_QUANTITY_CLASS):
        val_batch_gens[i] = torch.utils.data.DataLoader(val_datasets[i],
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    for i in range(DATASET_QUANTITY_CLASS, DATASET_QUANTITY):
        nn_modules[i].train(False)
        metric = []
        with torch.no_grad():
            for X_batch, y_batch in val_batch_gens[i]:
                X_batch = X_batch.to(device)
                y_pred = nn_modules[i](X_batch)
                y_pred = nn_modules[i].post_processing(y_pred)
                metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
        print(i, np.mean(metric))
        results[i][-1].append(np.mean(metric))                                                 

    with open('common_val_' + str(DATASET_QUANTITY) + '_' + str(DATASET_SIZE) + '.pkl', 'wb') as f:
        pickle.dump(results, f)

    # for i in range(DATASET_QUANTITY):
    #     torch.save(nn_modules[i].state_dict(), "nn_module_common_wd0_" + str(DATASET_QUANTITY) + '_' + str(DATASET_SIZE) + '_' + str(i) + ".pt")
    #     torch.save(optimizers[i].state_dict(), "optimizer_wd0_" + str(DATASET_QUANTITY) + '_' + str(DATASET_SIZE) + '_' + str(i) + ".pt")
    #     torch.save(schedulers[i].state_dict(), "scheduler_wd0_" + str(DATASET_QUANTITY) + '_' + str(DATASET_SIZE) + '_' + str(i) + ".pt")
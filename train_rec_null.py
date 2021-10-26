import os
import sys
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import utils
from module.forward_step import *
import torch.optim as optim
import utils.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from module.str_dataloader import StrTrain
from module.str_net import model_select
from pathlib import Path
import yaml

with Path('configs/device.conf').open('r', encoding='utf-8') as rf:
    device_conf = yaml.load(rf)

device = device_conf['TrainDevice']
device = torch.device('cuda:%d' % (
    device) if device != 'cpu' else 'cpu')

train_target = sys.argv[1]


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


model_name = sys.argv[2]
print(model_name)
dataset_train = StrTrain('train', model_name, train_target, get_transform(train=True))
dataset_dev = StrTrain('dev', model_name, train_target, get_transform(train=False))

print(dataset_train.num_class)

data_loader_train = torch.utils.data.DataLoader(
    dataset_dev, batch_size=64, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_dev = torch.utils.data.DataLoader(
    dataset_dev, batch_size=48, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

model = model_select(model_name, dataset_train.num_class, device)
# model.to(device)

for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 100
NUM_BATCH_EVAL = 70000
EARLY_STOP = 50
best_val_loss = 1e10
best_val_accn = 0
early_count = 0
stopstop = 0
len_train, len_valid = len(data_loader_train), len(data_loader_dev)
save_path = './model/'
for epoch in range(EPOCHS):
    batch_train_loss, batch_train_acc, batch_val_loss, batch_val_acc = [], [], [], []
    train_loop = tqdm(enumerate(data_loader_train), total=len_train)
    for batch_idx, batch in train_loop:
        train_loss, train_acc = training_step(batch, model, optimizer, criterion, device)
        batch_train_loss.append(train_loss)
        batch_train_acc.append(train_acc)
        epoch_idx = '{0:02d}'.format(epoch+1)
        train_loop.set_description(f'Epoch [{epoch_idx}/{EPOCHS}] ')
        try:
            train_loop.set_postfix(train_loss=train_loss, val_loss=avg_val_loss,
                                   train_acc=train_acc, val_acc=avg_val_acc)
        except:
            train_loop.set_postfix(loss=train_loss, acc=train_acc)
        if (batch_idx+1) % NUM_BATCH_EVAL == 0 or (batch_idx+1) == len(data_loader_train):
            valid_loop = tqdm(enumerate(data_loader_dev),
                              total=len_valid, position=0, leave=False)
            batch_val_loss, batch_val_acc = [], []
            for batch_idx_val, batch in valid_loop:
                valid_loss, val_acc = validation_step(batch, model, criterion, device)
                batch_val_loss.append(valid_loss)
                batch_val_acc.append(val_acc)
                valid_loop.set_description(
                    f'Epoch [{epoch+1}/{EPOCHS}] : Validating... [{batch_idx_val+1}/{len_valid}]')
            sub_batch = int(len_train*0.05)
            avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc = np.mean(batch_train_loss[-sub_batch:]), np.mean(batch_val_loss), np.mean(batch_train_acc), np.mean(batch_val_acc)
            train_loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss,
                                   train_acc=avg_train_acc, val_acc=avg_val_acc)
            if (batch_idx+1) % NUM_BATCH_EVAL == 0: print('')
            ##############################
            ##### Save log and model #####
            ##############################
            if avg_val_loss < best_val_loss:
                # save best
                torch.save(model.state_dict(), './model/weight/{}_{}.ckpt'.format(train_target, model_name))
                best_val_loss = avg_val_loss
                early_count = 0
            else:
                early_count += 1
            if early_count == EARLY_STOP:
                print('\nEarly stop\n')
                stopstop = 1
                break
    if stopstop:
        break
    
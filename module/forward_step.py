import torch
import torch.nn as nn
import torch.nn.functional as F


def categorical_accuracy(logits, labels):
    # correct = torch.argmax(logits, dim=1).eq(torch.argmax(labels, dim=1))
    correct = torch.argmax(logits, dim=1).eq(labels)
    return correct.sum()/torch.FloatTensor([labels.shape[0]])


def pad_to_same(input_imgs):
    max_w = 224
    max_h = 224

    # for x in input_imgs:
    #     img_dim = x.shape
    #     if img_dim[1] > max_h:
    #         max_h = img_dim[1]
    #     if img_dim[2] > max_w:
    #         max_w = img_dim[2]
            
    input_imgs = list(input_imgs)
    for i, x in enumerate(input_imgs):
        diff_h = max_h-x.shape[1]
        diff_w = max_w-x.shape[2]
        pad_w_left = diff_w//2
        if pad_w_left*2 != diff_w:
            pad_w_right = diff_w - pad_w_left
        else:
            pad_w_right = pad_w_left

        pad_h_top = diff_h//2
        if pad_h_top*2 != diff_h:
            pad_h_bottom = diff_h - pad_h_top
        else:
            pad_h_bottom = pad_h_top

        x = F.pad(x, [pad_w_left, pad_w_right, pad_h_top, pad_h_bottom])
        input_imgs[i] = x.unsqueeze(0)
        
    return torch.cat(input_imgs, axis=0)


def training_step(batch, model, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    input_imgs, labels = batch[0], batch[1]
    input_imgs = pad_to_same(input_imgs)
    # labels = torch.cat(list(labels), axis=0)
    labels = torch.tensor(labels)
    input_imgs = input_imgs.to(device)
    labels = labels.to(device)
    logits = model(input_imgs)
    loss = criterion(logits, labels)
    acc = categorical_accuracy(logits, labels)
    acc = acc.detach().cpu().numpy()[0]
    loss.backward()
    optimizer.step()

    # del input_imgs, labels, logits

    # with torch.cuda.device(device):
    #     torch.cuda.empty_cache()

    return loss.item(), acc


def validation_step(batch, model, criterion, device):
    lambda_ = 0.8
    model.eval()

    with torch.no_grad():
        input_imgs, labels = batch[0], batch[1]
        input_imgs = pad_to_same(input_imgs)
        # labels = torch.cat(list(labels), axis=0)
        labels = torch.tensor(labels)
        input_imgs = input_imgs.to(device)
        labels = labels.to(device)
        logits = model(input_imgs)
        loss = criterion(logits, labels)
        acc = categorical_accuracy(logits, labels)
        acc = acc.detach().cpu().numpy()[0]
    
    del input_imgs, labels, logits

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    return loss.item(), acc
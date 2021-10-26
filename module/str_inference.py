import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def pad_to_same(input_imgs):
    max_w = 224
    max_h = 224
            
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


# def inference(input_imgs, model, device):
#     model.eval()
#     with torch.no_grad():
#         input_imgs = input_imgs.to(device)
#         input_imgs = input_imgs.unsqueeze(0)
#         logits = model(pad_to_same(input_imgs))
#         logits = nn.Softmax(dim=1)(logits)
#         logits = logits.detach().cpu().numpy()[0]
#         zh_id = np.argmax(logits)
#         prob = logits[zh_id]
#     return cls_token_index_rev[zh_id], prob


# def inference_topk(input_imgs, model, device, top=1):
#     model.eval()
#     with torch.no_grad():
#         input_imgs = input_imgs.to(device)
#         input_imgs = input_imgs.unsqueeze(0)
#         logits = model(pad_to_same(input_imgs))
#         logits = nn.Softmax(dim=1)(logits)
#         logits = torch.topk(logits, top)
#         probs = logits[0].detach().cpu().numpy()[0]
#         indices = logits[1].detach().cpu().numpy()[0]
#         words = [cls_token_index_rev[x] for x in indices]
#     return words, probs

# def inference_ensemble_topk(input_imgs, model1, model2, device, top=1):
#     model1.eval()
#     model2.eval()
#     with torch.no_grad():
#         # channel 1
#         input_imgs1 = input_imgs[1].to(device)
#         input_imgs1 = input_imgs1.unsqueeze(0)
#         logits1 = model1(pad_to_same(input_imgs1))
#         logits1 = nn.Softmax(dim=1)(logits1)
        
#         # channel 3
#         input_imgs2 = input_imgs[0].to(device)
#         input_imgs2 = input_imgs2.unsqueeze(0)
#         logits2 = model2(pad_to_same(input_imgs2))
#         logits2 = nn.Softmax(dim=1)(logits2)
        
#         # ensemble
#         logits = torch.mean(torch.cat((logits1, logits2), axis=0), axis=0)
#         logits = torch.topk(logits, top)
#         probs = logits[0].detach().cpu().numpy()
#         indices = logits[1].detach().cpu().numpy()
#         words = [cls_token_index_rev[x] for x in indices]
    
#     del input_imgs, input_imgs1, input_imgs2, logits, logits1, logits2
#     if device.type != 'cpu':
#         with torch.cuda.device(device):
#             torch.cuda.empty_cache()
#     return words, probs

# def inference_ensemble(input_imgs, model1, model2, model3, device, top=3):
#     model1.eval()
#     model2.eval()
#     model3.eval()
#     with torch.no_grad():
#         # channel 1
#         input_imgs1 = input_imgs[1].to(device)
#         input_imgs1 = input_imgs1.unsqueeze(0)
#         logits1 = model1(pad_to_same(input_imgs1))
#         logits1 = nn.Softmax(dim=1)(logits1)
        
#         # channel 3
#         input_imgs2 = input_imgs[0].to(device)
#         input_imgs2 = input_imgs2.unsqueeze(0)
#         logits2 = model2(pad_to_same(input_imgs2))
#         logits2 = nn.Softmax(dim=1)(logits2)
        
#         # channel 1
#         input_imgs3 = input_imgs[1].to(device)
#         input_imgs3 = input_imgs3.unsqueeze(0)
#         logits3 = model3(pad_to_same(input_imgs3))
#         logits3 = nn.Softmax(dim=1)(logits3)
        
        
#         # ensemble
#         logits = torch.mean(torch.cat((logits1, logits2, logits3), axis=0), axis=0)
#         logits = torch.topk(logits, top)
#         probs = logits[0].detach().cpu().numpy()
#         indices = logits[1].detach().cpu().numpy()
#         words = [cls_token_index_rev[x] for x in indices]
    
#     del input_imgs, input_imgs1, input_imgs2, input_imgs3, logits, logits1, logits2, logits3
#     if device.type != 'cpu':
#         with torch.cuda.device(device):
#             torch.cuda.empty_cache()
#     return words, probs


# def inference_null(input_imgs, model, device):
#     model.eval()
#     with torch.no_grad():
#         input_imgs = input_imgs[1].to(device)
#         input_imgs = input_imgs.unsqueeze(0)
#         logits = model(pad_to_same(input_imgs))
#         logits = nn.Softmax(dim=1)(logits)
#         logits = logits.detach().cpu().numpy()[0]
#         is_word = np.argmax(logits)
#         prob = logits[is_word]
#     del input_imgs, logits
#     if device.type != 'cpu':
#         with torch.cuda.device(device):
#             torch.cuda.empty_cache()
#     return is_word, prob

def inference_null_batch(input_imgs, model, device):
    model.eval()
    with torch.no_grad():
        input_imgs = [x[1].to(device) for x in input_imgs]
        input_imgs = tuple(input_imgs)
        logits = model(pad_to_same(input_imgs))
        logits = nn.Softmax(dim=1)(logits)
        logits = logits.detach().cpu().numpy()
        is_word = np.argmax(logits, axis=1)
        prob = np.max(logits, axis=1)
    del input_imgs, logits
    if device.type != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return is_word, prob

def inference_ensemble_batch(input_imgs, model1, model2, model3, device, cls_token_index_rev, top=3):
    model1.eval()
    model2.eval()
    model3.eval()
    with torch.no_grad():
        # channel 1
        input_imgs1 = [x[1].to(device) for x in input_imgs]
        logits1 = model1(pad_to_same(input_imgs1))
        logits1 = nn.Softmax(dim=1)(logits1)
        
        # channel 3
        input_imgs2 = [x[0].to(device) for x in input_imgs]
        logits2 = model2(pad_to_same(input_imgs2))
        logits2 = nn.Softmax(dim=1)(logits2)

        # channel 1
        input_imgs3 = [x[1].to(device) for x in input_imgs]
        logits3 = model3(pad_to_same(input_imgs3))
        logits3 = nn.Softmax(dim=1)(logits3)

        
        # ensemble
        logits = (logits1+logits2+logits3)/3
        logits = torch.topk(logits, top)
        probs = logits[0].detach().cpu().numpy()
        indices = logits[1].detach().cpu().numpy()
        words = [[cls_token_index_rev[y] for y in x] for x in indices]
    
    del input_imgs, input_imgs1, input_imgs2, input_imgs3, logits, logits1, logits2, logits3
    if device.type != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return words, probs
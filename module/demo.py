import os
import sys
import cv2
import torch
import pickle
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
from PIL import Image
# sys.path.append(os.getcwd())
import utils.transforms as T
from utils.engine import evaluate
from module.std_net import get_instance_segmentation_model
from module.nms import soft_nms
from module.std_inference import *

import utils.transforms as T
from module.forward_step import *
from module.str_net import model_select
from module.str_inference import *
import  matplotlib.pyplot as plt
from module.str_dataloader import StrInference

from pathlib import Path
import yaml

with Path('configs/device.conf').open('r', encoding='utf-8') as rf:
    device_conf = yaml.load(rf)

device = device_conf['InferenceDevice']
device = torch.device('cuda:%d' % (
    device) if device != 'cpu' else 'cpu')

with open('./token_info/token_index.pickle', 'rb') as file:
    token_index = pickle.load(file)
file.close()

rev_token_index = {v: k for k, v in token_index.items()}

det_model_name = 'char_gray_blur_resnet152_25'
det_model = get_instance_segmentation_model(2, device, det_model_name)

null_model_name = 'null_ba_resnet18_68'
null_model = model_select('resnet18', 2, device, null_model_name)

rec_model_name_1 = 'cls_ba_resnet50_10'
rec_model_1 = model_select('resnet50', len(token_index), device, rec_model_name_1)

rec_model_name_2 = 'cls_ba_efficientb1_7'
rec_model_2 = model_select('efficientnetb1', len(token_index), device, rec_model_name_2)

rec_model_name_3 = 'cls_ba_densenet201_10'
rec_model_3 = model_select('densenet201', len(token_index), device, rec_model_name_3)

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def load_image(filename):
    image = Image.open(f'./dataset/demo_imgs/{filename}.jpg').convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1, img2 = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 1, axis=2), np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 3, axis=2)
    return (image, img1, img2)

def det_image_preprocess(img):
    blur = min(img.shape[0], img.shape[1])
    k = int(blur*0.08)
    if k > 0:
        img = cv2.blur(img, (k, k))
    return img

def show_prediction(img, model, device, filename, show=True, threshold=[0.5, 0.7, 0.9]):
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        
    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    valid_boxes, _ = to_dets_score(boxes.tolist(), scores.tolist(), threshold)
    img = cv2.imread(f'./dataset/demo_imgs/{filename}.jpg')
    res = []
    for index in valid_boxes:
        bb = np.round(boxes[index]).astype(int)
        res.append(list(bb))
    res = valid_box(res)
    res, _ = mean_slope(res)
    for bb in res:
        img = cv2.polylines(img, [np.array([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])], True, (0, 255, 0), 2)
    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(img[:,:,::-1])
        plt.show()
    if device.type != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return img, res

def Demo(filename, show_size=(8, 8)):
    image, img1, img2 = load_image(filename)
    det_img, _ = get_transform()(img1, {})
    _, det = show_prediction(det_img, det_model, device, filename, False)
    if det:
        image = np.array(image)
        for i, bb in enumerate(det):
            char_img = image[bb[1]:bb[3], bb[0]:bb[2], ::-1]
            cv2.imwrite('/tmp/char_{}.jpg'.format(str(i+1)), char_img)
    else:
        plt.imshow(image)
        print('###')
        plt.show()
        return '###'
    
    char_img_path = '/tmp/'
    dataset_test = StrInference('test', char_img_path, get_transform())
    
    input_imgs = [dataset_test[x] for x in range(len(dataset_test)) if 'jpg' in os.listdir(char_img_path)[x]]
    
    tmp = []
    is_word, is_word_prob = inference_null_batch(input_imgs, null_model, device)

    if (len(is_word)-sum(is_word))/len(is_word) > 0.5:
        tmp = [{'candidates': ['NULL'], 'prob': [1.00]}]
    else:
        words, probs = inference_ensemble_batch(input_imgs, rec_model_1, rec_model_2, rec_model_3, device, rev_token_index)
        for iii, is_is_is_word in enumerate(is_word):
            if is_is_is_word and is_word_prob[iii] >= 0.9:
                tmp.append({'candidates': words[iii], 'prob': probs[iii]})
            else:
                tmp.append({'candidates': ['NULL'], 'prob': [1.00]})
                
    res = ''
    valid_det = []
    if len(tmp) == 1 and v[0]['candidates'] == 'NULL':
        res += '###'
    else:
        if len(tmp) == 1 and v[0]['prob'][0] < 0.5:
            res += '###'
        else:
            for i, j in enumerate(tmp):
                if j['prob'][0] < 0.35:
                    res += '#'
                else:
                    if j['candidates'][0] == 'NULL':
                        res += '#'
                    else:
                        res += j['candidates'][0]
                        valid_det.append(det[i])
    for file in os.listdir('/tmp/'):
        if 'jpg' not in file:
            continue
        os.remove(f'/tmp/{file}')
    
    img = image.copy()
        
    for bb in valid_det:
        img = cv2.polylines(img, [np.array([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])], True, (0, 255, 0), 2)
    
    # plt.imshow(image)
    # plt.show()
    print(res)
    plt.figure(figsize=show_size)
    plt.imshow(img)
    plt.show()    
    return image, img, res

import os
import sys
import cv2
import torch
import pickle
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
sys.path.append(os.getcwd())
import utils.transforms as T
from utils.engine import evaluate
from module.std_net import get_instance_segmentation_model
from module.nms import soft_nms
from module.std_inference import *
from module.std_dataloader import StdInference
from pathlib import Path
import yaml

with Path('configs/device.conf').open('r', encoding='utf-8') as rf:
    device_conf = yaml.load(rf)

device = device_conf['InferenceDevice']
device = torch.device('cuda:%d' % (
    device) if device != 'cpu' else 'cpu')


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

if __name__ == '__main__':

    inference_data = sys.argv[1] # public private

    index_map_path = f'./dataset/{inference_data}_data/index_map.pickle'
    test_img_poly_pts_path = f'./dataset/{inference_data}_data/'
    img_path = f'./dataset/{inference_data}_data/img/'
    char_img_save_path = f'./dataset/{inference_data}_data/char_img/'
    model_name = 'char_gray_blur_resnet152_25'

    if not os.path.isdir(char_img_save_path):
        os.mkdir(char_img_save_path)

    # dataset_test = TestDataset(get_transform(train=False))
    dataset_test = StdInference(index_map_path, img_path, get_transform(train=False))

    num_classes = 2
    model = get_instance_segmentation_model(num_classes, device, model_name)
    # model.to(device)

    with open(test_img_poly_pts_path+'test_img_poly_pts.pickle', 'rb') as file:
        test_img_poly_pts = pickle.load(file)
    file.close()


    for idx in tqdm(range(len(dataset_test))):
        img_name, img, kk = show_prediction(dataset_test, model, device, idx+1, img_path, False, [0.85, 0.85, 0.85])
        vbb = valid_box(kk)
        try:
            line_poly = Polygon(test_img_poly_pts[img_name])
            vbb = in_poly(line_poly, vbb)
        except:
            pass
        if vbb:
            vbb, ms = mean_slope(vbb)
        img = cv2.imread(img_path+f'{img_name}.jpg')
        if vbb:
            for i, bb in enumerate(vbb):
                char_img = img[bb[1]:bb[3], bb[0]:bb[2], :]
                cv2.imwrite(char_img_save_path+f'{img_name}_{str(i+1)}.jpg', char_img)

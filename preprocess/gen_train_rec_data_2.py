import os
import re
import random
import numpy as np
import json
import cv2
import pickle
from tqdm import tqdm
from itertools import chain
from PIL import Image, ExifTags

## generate null character img
def rnd_gen_box(width, height):
    r_xmin = random.randint(0, width)
    r_ymin = random.randint(0, height)
    r_height = random.randint(1, int(height*0.2))
    r_width = random.randint(1, int(width*0.2))
    r_xmax = r_xmin + r_width
    r_ymax = r_ymin + r_height
    return r_xmin, r_ymin, r_xmax, r_ymax


def IOU(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return interArea

## generate 150 null img
if __name__ == '__main__':
    null_count = 150

    img_id = 1

    while img_id <= null_count:
        file_name = random.choice(os.listdir('./dataset/train_data/aicup/train/json'))
        file_name = file_name.split('.')[0]
        json_file = f'./dataset/train_data/aicup/train/json/{file_name}.json'
        img_file = f'./dataset/train_data/aicup/train/img/{file_name}.jpg'
        
        with open(json_file, 'r', encoding='utf8') as file:
            data = json.load(file)
        file.close()
        
        boxes = []
        for k in data['shapes']:
            if k['group_id'] == 1:
                pts = list(chain(*k['points']))
                xmin = min([pts[0], pts[2], pts[4], pts[6]])
                ymin = min([pts[1], pts[3], pts[5], pts[7]])
                xmax = max([pts[0], pts[2], pts[4], pts[6]])
                ymax = max([pts[1], pts[3], pts[5], pts[7]])
                boxes.append([xmin, ymin, xmax, ymax])
        
        img = cv2.imread(img_file)
        height, width = img.shape[0], img.shape[1]
        
        
        r_xmin, r_ymin, r_xmax, r_ymax = rnd_gen_box(width, height)
        
        while True:
            if r_xmax > width or r_ymax > height or r_xmax == r_xmin or r_ymax == r_ymin:
                r_xmin, r_ymin, r_xmax, r_ymax = rnd_gen_box(width, height)
                continue
            again = 0
            for bb in boxes:
                if IOU([r_xmin, r_ymin, r_xmax, r_ymax], bb) > 0:
                    again = 1
                    break
            if again:
                r_xmin, r_ymin, r_xmax, r_ymax = rnd_gen_box(width, height)
                continue
            break
            
        null_crop = img[r_ymin: r_ymax+1, r_xmin: r_xmax+1, :]
        
        cv2.imwrite('./dataset/train_data/rec/char_img/img_null_{}.jpg'.format(img_id), null_crop)

        with open('./dataset/train_data/rec/char_gt/gt_null_{}.pickle'.format(img_id), 'wb') as output_file:
            pickle.dump({'ground_truth': 'NULL'}, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.close()
        
        img_id += 1
        if img_id % 100 == 0:
            print(img_id)
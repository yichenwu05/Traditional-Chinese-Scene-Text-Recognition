import re
import os
import cv2
import json
import copy
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import chain

output_path = './dataset/train_data/det'
if not os.path.isdir(output_path):
    os.mkdir(output_path)

def crop_and_mask(points, img):
    pts = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]])
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    mask = gen_mask(points, img)
    return croped, mask

def gen_mask(points, img):
    masked_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask.fill(255)
    
    roi_corners = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]], dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], 0)
    
    masked_image = cv2.bitwise_or(masked_image, mask)
    masked_image[masked_image == 0] = 1
    masked_image[masked_image == 255] = 0
    return masked_image

def map_to_line(data):

    # to box
    for char in data['chars']:
        char['points'] = check_points_order(char['points'])
        char['box'] = convert2box(char['points'])

    for line in data['lines']:
        line['points'] = check_points_order(line['points'])
        line['box'] = convert2box(line['points'])

    line_char = {}

    for char in data['chars']:
        if not re.findall(r'[\da-zA-Z\u4e00-\u9fff]', char['transcription'].strip()):
            continue
        max_iou = 0
        tmp_i = None
        for i, line in enumerate(data['lines']):
            k = IOU(char['box'], line['box'])
            if k > max_iou:
                max_iou = k
                tmp_i = i
        if tmp_i != None:
            if char['transcription'].strip() in data['lines'][tmp_i]['transcription']:
                if tmp_i not in line_char:
                    line_char[tmp_i] = []
                line_char[tmp_i].append(char)
        
    return line_char


def img_gt_mask_aicup(line_char, data, img_cv, _id):
    bbox = 0
    for i in line_char:
        gt_info = {
            'img_name': 'aicup_{0:06d}'.format(_id) + '_{}'.format(i+1),
            'transcription': [],
            'boxes' : [],
            'area': [],
            'num_objs': 0,
            'num_lines': 0,
            'size': []
        }

        the_line = line_char[i]
        croped, _ = crop_and_mask(data['lines'][i]['points'], img_cv)

        num_objs = len(the_line)
        num_lines = len(the_line)
        
        gt_info['num_objs'] = num_objs
        gt_info['num_lines'] = num_lines
        gt_info['size'] = list(croped.shape)

        masks = np.zeros((num_objs, croped.shape[0], croped.shape[1]))
        for j, the_char in enumerate(the_line):
            r_points, r_box  = resize_bb(the_char['points'], the_char['box'], data['lines'][i]['box'])
            the_char['points'] = r_points
            the_char['box'] = r_box
            _, mask = crop_and_mask(the_char['points'], croped)
            masks[j, :, :] = mask
            gt_info['boxes'].append(r_box)
            gt_info['area'].append((r_box[2] - r_box[0]+1) * (r_box[3] - r_box[1]+1))
            gt_info['transcription'].append(the_char['transcription'])
        
        assert len(gt_info['boxes']) == len(gt_info['area']) == gt_info['num_objs'] == gt_info['num_lines'] == len(masks)
        
        ###### save croped image, masks and gt_info
        if not os.path.isdir(output_path+'/gt/'):
            os.mkdir(output_path+'/gt/')

        if not os.path.isdir(output_path+'/mask/'):
            os.mkdir(output_path+'/mask/')

        if not os.path.isdir(output_path+'/img/'):
            os.mkdir(output_path+'/img/')
        
        with open(output_path+'/gt/gt_aicup_{0:06d}_'.format(_id) + '{}'.format(i+1) + '.pickle', 'wb') as output_file:
            pickle.dump(gt_info, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.close()
        
        masks_img = np.reshape(masks, (masks.shape[0]*masks.shape[1], masks.shape[2]))
        cv2.imwrite(output_path+'/mask/mask_aicup_{0:06d}_'.format(_id) + '{}'.format(i+1) + '.png', masks_img)

        cv2.imwrite(output_path+'/img/img_aicup_{0:06d}_'.format(_id) + '{}'.format(i+1) + '.jpg', croped)
        
        bbox += gt_info['num_objs']
        
    return bbox


def to_data(json_file):
    data = {}
    data['chars'] = []
    data['lines'] = []
    for k in json_file['shapes']:
        
        info = {
            'transcription': k['label'].strip(),
            'points': [x if x > 0 else 0 for x in list(chain(*k['points']))]
        }

        if not info['transcription']:
            continue
        
        if k['group_id'] == 0:
            data['lines'].append(info)
        else:
            data['chars'].append(info)
            
    return data    

def IOU(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = (interArea**2 / float(boxAArea + boxBArea - interArea))# * ((interArea/boxAArea)*)
    if interArea/boxAArea < 0.8:
        iou = 0
    return iou


def convert2box(points):
    xmin = min([points[0], points[2], points[4], points[6]])
    ymin = min([points[1], points[3], points[5], points[7]])
    xmax = max([points[0], points[2], points[4], points[6]])
    ymax = max([points[1], points[3], points[5], points[7]])
    
    return [xmin, ymin, xmax, ymax]


def check_points_order(points):
    i = 0
    while not (points[0] <= points[4] and points[1] <= points[5]) and i < 4:
        points = points[2:] + points[:2]
        i += 1
    return points


def resize_bb(points, box, pbox):
    pxmin, pymin, pxmax, pymax = pbox
    move_x = copy.copy(pxmin)
    move_y = copy.copy(pymin)
    pxmin, pymin, pxmax, pymax = 0, 0, pxmax-move_x, pymax-move_y
    xmin, ymin, xmax, ymax = box
    xmin, ymin, xmax, ymax = xmin-move_x, ymin-move_y, xmax-move_x, ymax-move_y
    
    x0, x1, x2, x3 = points[0]-move_x, points[2]-move_x, points[4]-move_x, points[6]-move_x 
    y0, y1, y2, y3 = points[1]-move_y, points[3]-move_y, points[5]-move_y, points[7]-move_y
    
    if xmin < 0:
        xmin = 0
    if xmax > pxmax:
        xmax = pxmax
    if ymin < 0:
        ymin = 0
    if ymax > pymax:
        ymax = pymax
    
    if x0 < xmin:
        x0 = xmin
    if x1 > xmax:
        x1 = xmax
    if x2 > xmax:
        x2 = xmax
    if x3 < xmin:
        x3 = xmin
    
    if y0 < ymin:
        y0 = ymin
    if y1 > ymax:
        y1 = ymax
    if y2 > ymax:
        y2 = ymax
    if y3 < ymin:
        y3 = ymin
        
    return [x0, y0, x1, y1, x2, y2, x3, y3], [xmin, ymin, xmax, ymax]

if __name__=='__main__':
    load_path = './dataset/train_data/aicup/train'
    for filename in tqdm(os.listdir(load_path+'/json/')):
        filename = filename.split('.')[0]
        with open(load_path+f'/json/{filename}.json', encoding='utf8') as file:
            json_file = json.load(file)
        file.close()
        img_cv = cv2.imread(load_path+f'/img/{filename}.jpg')
        data = to_data(json_file)
        if data['lines'] and data['chars']:
            line_char = map_to_line(data)
            _id = int(filename.split('_')[-1])
            bbox = img_gt_mask_aicup(line_char, data, img_cv, _id)

    all_files = ['_'.join(x.split('.')[0].split('_')[1:]) for x in os.listdir(output_path+'/img/') if 'jpg' in x]
    dev_files = random.sample(all_files, int(len(all_files)*0.1))
    train_files = [x for x in all_files if x not in dev_files]
    
    aicup_train_index_map = {i: x for i, x in enumerate(train_files)}
    aicup_dev_index_map = {i: x for i, x in enumerate(dev_files)}

    with open(output_path+'/aicup_train_index_map.pickle', 'wb') as output_file:
        pickle.dump(aicup_train_index_map, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

    with open(output_path+'/aicup_dev_index_map.pickle', 'wb') as output_file:
        pickle.dump(aicup_dev_index_map, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()
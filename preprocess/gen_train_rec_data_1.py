import os
import re
import random
import numpy as np
import json
import cv2
import pickle
from tqdm import tqdm
from PIL import Image, ExifTags

## crop character from img
if __name__ == '__main__':

    load_path = './dataset/train_data/aicup/train'
    bow = {}

    if not os.path.isdir('./dataset/train_data/rec'):
        os.mkdir('./dataset/train_data/rec')

    if not os.path.isdir('./dataset/train_data/rec/char_gt'):
        os.mkdir('./dataset/train_data/rec/char_gt')

    if not os.path.isdir('./dataset/train_data/rec/char_img'):
        os.mkdir('./dataset/train_data/rec/char_img')

    for filename in os.listdir('./dataset/train_data/aicup/train/json'):
        if 'json' in filename:
            with open(f'./dataset/train_data/aicup/train/json/{filename}', encoding='utf8') as file:
                label = json.load(file)
                for k in label['shapes']:
                    for s in k['label']:
                        if not re.search(r'[\u4e00-\u9fff]+', s):
                            continue
                        if s.strip() not in bow:
                            bow[s.strip()] = 0
                        bow[s.strip()] += 1      

            file.close()


    for file_name in tqdm(os.listdir('./dataset/train_data/det/img')):
        
        file_name = '_'.join(file_name.split('_')[1:]).split('.')[0]
        if 'aicup' not in file_name:
            continue
        img = Image.open(f'./dataset/train_data/det/img/img_{file_name}.jpg').convert("RGB")
        img = np.array(img)

        with open(f'./dataset/train_data/det/gt/gt_{file_name}.pickle', 'rb') as file:
            gt = pickle.load(file)
        file.close()
        
        j = 0
        for i, k in enumerate(gt['boxes']):
            s = gt['transcription'][i]
            crop_img = img[k[1]: k[3]+1, k[0]: k[2]+1, :]
                    
            cv2.imwrite(f'./dataset/train_data/rec/char_img/img_{file_name}' + '_{}'.format(str(j+1)) + '.jpg', crop_img)

            with open(f'./dataset/train_data/rec/char_gt/gt_{file_name}' + '_{}'.format(str(j+1)) + '.pickle', 'wb') as output_file:
                pickle.dump({'ground_truth': s}, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.close()

            j += 1
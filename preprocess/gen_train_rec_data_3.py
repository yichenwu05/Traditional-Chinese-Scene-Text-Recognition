import os
import re
import cv2
import copy
import time
import json
import pickle
import random
import numpy as np
from p_tqdm import p_map
from tqdm import tqdm
from itertools import chain
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode

## generate fake image


the_fonts = [
 'setofont.ttf',
 'HanyiSentyTang.ttf',
 'Kaisotai-Next-UP-B.ttf',
 'Senobi-Gothic-Regular.ttf',
 'SNsanafonkaku.ttf',
 'HanaMinA.ttf',
 'RiiT_F.otf',
 'Brush.ttf',
 'Calligraphy.ttf',
 'Advertisement.ttf',
 'Stamp.ttf'
]

all_files = os.listdir('./dataset/train_data/aicup/train/json/')


def has_glyph(glyph, fontfile):
    the_font = TTFont(f"./fonts/{fontfile}")
    for table in the_font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False


def gen_img(s, size, fonts):
    W, H = (size, size)
    font_size = int(size * np.random.uniform(0.95,1.1,1)[0])
    # Font
    font = ImageFont.truetype(f"./fonts/{fonts}", font_size, encoding='utf-8')

    # Image
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    offset_w, offset_h = font.getoffset(s)
    w, h = draw.textsize(s, font=font)

    pos = ((W-w-offset_w)/2, (H-h-offset_h)/2)

    # Draw
    draw.text(pos, s, "black", font=font)

    image = np.array(image)
    
    return image


def rnd_gen_box(width, height, size):
    r_xmin = random.randint(0, width)
    r_ymin = random.randint(0, height)
    r_height = size
    r_width = size
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


def gen_background(size):
    osize = copy.copy(size)
        
    file_name = random.choice(all_files).split('.')[0]

    json_file = f'./dataset/train_data/aicup/train/json/{file_name}.json'
    img_file = f'./dataset/train_data/aicup/train/img/{file_name}.jpg'
    
    with open(json_file, 'r', encoding='utf8') as file:
        data = json.load(file)
    file.close()
    
    img = cv2.imread(img_file)
    height, width = img.shape[0], img.shape[1]
    
    
    r_xmin, r_ymin, r_xmax, r_ymax = rnd_gen_box(width, height, size)
    tries = 0
    while True and tries < 20:
        if r_xmax > width or r_ymax > height or r_xmax == r_xmin or r_ymax == r_ymin:
            r_xmin, r_ymin, r_xmax, r_ymax = rnd_gen_box(width, height, size)
            tries += 1
            continue
            
        break
        
    if tries == 100:
        return None
    
    background = img[r_ymin: r_ymax, r_xmin: r_xmax, :]
    
    if osize > size:
        background = cv2.resize(background, (osize, osize), interpolation=cv2.INTER_AREA)
    
    return background

def merge(img, background):
    img = np.array(img)  
    bg_mean = list(np.mean(background, axis=(0, 1), dtype=int))
    adjust = list(np.random.uniform(-50, 50, 3))
    img_contrast = [255-x for x in bg_mean]
    img_contrast = [x+adjust[i] if x+adjust[i] <= 255 else 255 for i, x in enumerate(img_contrast)]
    img_contrast = [x if x > 0 else 0 for x in img_contrast]
    mean_img = np.mean(img, axis=2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mean_img[i, j] >= 100:
                img[i, j, :] = bg_mean
            else:
                img[i, j, :] = img_contrast
    return img

def gen_fake_char(word, index):
        
    no_good_font = set()
    good_font = set()
    stop = 0
    try:
        while True and len(no_good_font) < len(the_fonts) and stop < 20:
            img_size = random.randint(60, 200)
            bk = gen_background(img_size)
            try:
                bk.shape
            except:
                stop += 1
                continue    
            tar_font = random.choice(the_fonts)

            if tar_font in no_good_font:
                stop += 1
                continue
            elif tar_font in good_font:
                pass
            else:
                if not has_glyph(word, tar_font):
                    no_good_font.add(tar_font)
                    stop += 1
                    continue
            img = gen_img(word, img_size, tar_font)

            ddd = np.mean(img, axis=2)
            check_none = np.sum(ddd == 255)
            if check_none >= (img.shape[0]**2)*0.95:
                no_good_font.add(tar_font)
                stop += 1
                continue

            img = merge(img, bk)

            cv2.imwrite(f'./dataset/train_data/rec/char_img/img_fake_{fake_id}_'+ '{}.jpg'.format(str(index)), img)
            with open(f'./dataset/train_data/rec/char_gt/gt_fake_{fake_id}_'+ '{}.pickle'.format(str(index)), 'wb') as output_file:
                pickle.dump({'ground_truth': word}, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.close()

            break
    except:
        pass

# if char count < 400, then generate 50 fake image
if __name__ == '__main__':

    gt_chars = {}
    for filename in os.listdir('./dataset/train_data/rec/char_gt/'):
        with open(f'./dataset/train_data/rec/char_gt/{filename}', 'rb') as output_file:
            gt_info = pickle.load(output_file)
        output_file.close()
        if gt_info['ground_truth'] not in gt_chars:
            gt_chars[gt_info['ground_truth']] = 0
        gt_chars[gt_info['ground_truth']] += 1
    gt_chars = dict(sorted(gt_chars.items(), key=lambda x: x[1]))

    num_to_gen = 50

    fake_id = 1
    for k, v in tqdm(gt_chars.items()):

        if v >= 400:
            continue
        
        p_map(gen_fake_char, [k] * num_to_gen, [i for i in range(num_to_gen)], num_cpus=60)
        
        fake_id += 1
        
        print('{}/{}'.format(fake_id-1, len(gt_chars)))

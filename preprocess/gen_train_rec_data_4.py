import os
import json
import pickle
import random
import numpy as np
from PIL import Image, ExifTags

# rec train dev split
if __name__ == '__main__':

    img_files = os.listdir('./dataset/train_data/rec/char_img/')

    # dev
    dev_seen = set()

    dev_char = [x for x in img_files if 'fake' not in x and 'null' not in x]
    random.shuffle(dev_char)
    dev_1 = dev_char[:4500]

    dev_null = [x for x in img_files if 'null' in x]
    random.shuffle(dev_null)
    dev_2 = dev_null[:500]
    dev_seen = set(dev_1+dev_2)

    rec_dev_index = {}
    for i, x in enumerate(dev_1+dev_2):
        rec_dev_index[i] = x


    # train 
    gt_files = os.listdir('./dataset/train_data/rec/char_gt/')
    char_to_name = {}
    for filename in gt_files:
        with open(f'./dataset/train_data/rec/char_gt/{filename}', 'rb') as output_file:
            gt_info = pickle.load(output_file)
        output_file.close()
        s = gt_info['ground_truth']
        if s not in char_to_name:
            char_to_name[s] = []
        char_to_name[s].append('img_'+'_'.join(filename.split('.')[0].split('_')[1:])+'.jpg')


    rec_train_index = {}
    index = 0
    for s in char_to_name:
        if s == 'NULL':
            for i in range(1500):
                rec_train_index[index] = s
                index += 1
        else:
            for i in range(500):
                rec_train_index[index] = s
                index += 1    


    with open('./dataset/train_data/rec/rec_train_index_map.pickle', 'wb') as output_file:
        pickle.dump(rec_train_index, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()
        
    with open('./dataset/train_data/rec/rec_dev_index_map.pickle', 'wb') as output_file:
        pickle.dump(rec_dev_index, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

    with open('./dataset/train_data/rec/char_to_img.pickle', 'wb') as output_file:
        pickle.dump(char_to_name, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

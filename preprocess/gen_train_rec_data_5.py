import os
import json
import pickle
import random
import numpy as np
from PIL import Image, ExifTags


if __name__ == '__main__':

    img_files = os.listdir('./dataset/train_data/rec/char_img/')

    # dev
    dev_seen = set()
    dev_char = [x for x in img_files if 'fake' not in x and 'null' not in x]
    random.shuffle(dev_char)
    dev_1 = dev_char[:500]

    dev_null = [x for x in img_files if 'null' in x]
    random.shuffle(dev_null)
    dev_2 = dev_null[:500]
    dev_seen = set(dev_1+dev_2)


    null_dev_index = {}
    for i, x in enumerate(dev_1+dev_2):
        null_dev_index[i] = x

    #train
    null_to_img = {0:[], 1:[]}
    for x in img_files:
        if x in dev_seen or 'fake' in x:
            continue
        if 'null' in x:
            null_to_img[0].append(x)
        else:
            null_to_img[1].append(x)

    null_train_index = {}
    for i in range(10000):
        null_train_index[i] = 0 if i < 8500 else 1

    with open('./dataset/train_data/rec/null_train_index_map.pickle', 'wb') as output_file:
        pickle.dump(null_train_index, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()


    with open('./dataset/train_data/rec/null_dev_index_map.pickle', 'wb') as output_file:
        pickle.dump(null_dev_index, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

    with open('./dataset/train_data/rec/null_to_img.pickle', 'wb') as output_file:
        pickle.dump(null_to_img, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

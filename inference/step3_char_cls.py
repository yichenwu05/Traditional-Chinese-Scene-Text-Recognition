import os
import sys
import pickle
from tqdm import tqdm
sys.path.append(os.getcwd())
import utils.transforms as T
from module.forward_step import *
from module.str_net import model_select
from module.str_inference import *
from module.str_dataloader import StrInference
from pathlib import Path
import yaml

with Path('configs/device.conf').open('r', encoding='utf-8') as rf:
    device_conf = yaml.load(rf)

device = device_conf['InferenceDevice']
device = torch.device('cuda:%d' % (
    device) if device != 'cpu' else 'cpu')


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

if __name__ == '__main__':

    inference_data = sys.argv[1]

    char_img_path = f'./dataset/{inference_data}_data/char_img/'
    img_path = f'./dataset/{inference_data}_data/img/'
    token_id_path = './token_info/token_index.pickle'
    save_ans_path = f'./dataset/{inference_data}_data/private_submit.pickle'

    char_index_map = {}
    for i, filename in enumerate(os.listdir(char_img_path)):
        char_index_map[i] = filename.split('.')[0]
    char_index_map_rev = {v: k for k, v in char_index_map.items()}

    line_char_map = {}

    for filename in os.listdir(char_img_path):
        line_name = '_'.join(filename.split('.')[0].split('_')[:-1])
        if line_name not in line_char_map:
            line_char_map[line_name] = []
        line_char_map[line_name].append(filename)

    with open(token_id_path, 'rb') as file:
        cls_token_index = pickle.load(file)
    file.close()
    cls_token_index_rev = {v: k for k, v in cls_token_index.items()}


    model_null = model_select('resnet18', 2, device, 'null_ba_resnet18_68')
    model1 = model_select('resnet50', len(cls_token_index), device, 'cls_ba_resnet50_10')
    model2 = model_select('efficientnetb1', len(cls_token_index), device, 'cls_ba_efficientb1_7')
    model3 = model_select('densenet201', len(cls_token_index), device, 'cls_ba_densenet201_10')

    dataset_test = StrInference('test', char_img_path, get_transform(train=False))
    

    ans = {}
    for filename in tqdm(os.listdir(img_path)):
        filename = filename.split('.')[0]
        tmp = []
        if filename not in line_char_map:
            ans[filename] = [{'candidates': ['NULL'], 'prob': [1.00]}]
        else:
            charnames = line_char_map[filename]
            charnames = [x.split('.')[0] for x in charnames]
            idxs = [char_index_map_rev[x] for x in charnames]
            input_imgs = [dataset_test[x] for x in idxs]
            
            is_word, is_word_prob = inference_null_batch(input_imgs, model_null, device)
            
            if (len(is_word)-sum(is_word))/len(is_word) > 0.5:
                tmp = [{'candidates': ['NULL'], 'prob': [1.00]}]
            else:
                words, probs = inference_ensemble_batch(input_imgs, model1, model2, model3, device, cls_token_index_rev)
                for iii, is_is_is_word in enumerate(is_word):
                    if is_is_is_word and is_word_prob[iii] >= 0.9:
                        tmp.append({'candidates': words[iii], 'prob': probs[iii]})
                    else:
                        tmp.append({'candidates': ['NULL'], 'prob': [1.00]})
            ans[filename] = tmp

    with open(save_ans_path, 'wb') as output_file:
        pickle.dump(ans, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

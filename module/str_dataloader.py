import os
import cv2
import pickle
import random
import numpy as np
import torch.utils.data
import albumentations as A
from imgaug import augmenters as iaa
from PIL import Image, ExifTags, ImageFilter


class StrInference(torch.utils.data.Dataset):
    def __init__(self, dataset, char_img_path, transforms=None):
        '''
        dataset : train, dev
        '''
        self.dataset = dataset
        self.char_img_path = char_img_path
        self.transforms = transforms

        self.char_index_map = {}
        for i, filename in enumerate(os.listdir(self.char_img_path)):
            self.char_index_map[i] = filename.split('.')[0]

        self.total = len(self.char_index_map)

    def __getitem__(self, idx):
        
        file_name = self.char_index_map[idx]
        
        img = Image.open(self.char_img_path+f'{file_name}.jpg').convert("RGB")
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]
        max_size = max(height, width)
        if max_size > 224:
            rr = 224/max_size
            img = cv2.resize(img, (min(224, max(5, int(height*rr))), min(224, max(5, int(width*rr)))), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_null = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 1, axis=2)
        img = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 3, axis=2)

        target = {}

        if self.transforms is not None:
            img, _ = self.transforms(img, target)
            img_null, _ = self.transforms(img_null, target)

        return img, img_null

    def __len__(self):
        return self.total


class StrTrain(torch.utils.data.Dataset):
    def __init__(self, dataset, model_name, train_target, transforms=None):
        '''
        
        dataset : train, dev
        
        '''
        
        self.dataset = dataset
        self.model_name = model_name
        self.train_target = train_target
        self.transforms = transforms
        
        if self.train_target == 'rec':
            with open(f'./dataset/train_data/rec/rec_{self.dataset}_index_map.pickle', 'rb') as file:
                self.index_map = pickle.load(file)
            file.close()
        elif self.train_target == 'null':
            with open(f'./dataset/train_data/rec/null_{self.dataset}_index_map.pickle', 'rb') as file:
                self.index_map = pickle.load(file)
            file.close()
        
        if self.train_target == 'rec':
            with open('./token_info/token_index.pickle', 'rb') as file:
                self.token_index = pickle.load(file)
            file.close()
            self.num_class = len(self.token_index)
        elif self.train_target == 'null':
            self.num_class = 2

        if self.train_target == 'rec':
            with open('./dataset/train_data/rec/char_to_img.pickle', 'rb') as file:
                self.char_to_img = pickle.load(file)
            file.close()
        elif self.train_target == 'null':
            with open('./dataset/train_data/rec/null_to_img.pickle', 'rb') as file:
                self.null_to_img = pickle.load(file)
            file.close()

        self.total = len(self.index_map)

    def augmentation(self, image):
        '''
        Size control and Augmentation for Image 
        '''
        # albumentations
        random_brightness_contrast = A.RandomBrightnessContrast(p=0.5)
        random_elastic_transform = A.ElasticTransform(p=0.3, alpha=12, sigma=12, alpha_affine=2.5)
        shift_scale_rotate = A.ShiftScaleRotate(p=0.3, border_mode=1)
        rgb_shift = A.RGBShift(p=0.3)

        is_opposite = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
        is_rgb_reverse = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
        is_rescale = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
        is_blur = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
        is_noise = np.random.choice([0, 1, 2, 3], size=1, p=[0.7, 0.1, 0.1, 0.1])
        is_resize = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
        is_shift = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
        image = random_brightness_contrast(image=image)['image']

        if is_opposite:
            image = 255 - image
        if is_rgb_reverse:
            image = image[:, :, ::-1]
        image = rgb_shift(image=image)['image']
        if is_rescale:
            rs = np.random.uniform(0.75, 1.25)
            image = cv2.resize(image, (max(5, int(image.shape[1]*rs)), max(5, int(image.shape[0]*rs))), interpolation=cv2.INTER_AREA)
        if is_blur:
            blur = min(image.shape[0], image.shape[1])
            k = int(blur*np.random.uniform(0.03, 0.1))
            if k > 0:
                image = cv2.blur(image, (k, k))
        if is_resize:
            aspect_ratio = np.random.uniform(0.5, 2.25)
            img_size = (image.shape[0]+image.shape[1])//2
            new_h = img_size*2/(1+aspect_ratio)
            new_w = img_size*2-new_h
            image = cv2.resize(image, (max(5, int(new_w)), max(5, int(new_h))), interpolation=cv2.INTER_AREA)
        if is_noise:
            if is_noise == 1:
                noise = iaa.SaltAndPepper(p=0.05)
            elif is_noise == 2:
                noise = iaa.AdditiveGaussianNoise(loc=0, scale=0.05*255)
            elif is_noise == 3:
                noise = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
            image = noise.augment_image(image)
        img = shift_scale_rotate(image=image)['image']
        if is_shift:
            image = self.shift_image(image)
        
        return image

        def shift_image(self, image):
            ra, rb = np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)
            x = int(image.shape[1] * ra)
            y = int(image.shape[0] * rb)

            for i in range(image.shape[1] -1, image.shape[1] - x, -1):
                image = np.roll(image, -1, axis=1)
                image[:, -1] = 0
                
            for i in range(image.shape[0] -1, image.shape[0] - y, -1):
                image = np.roll(image, -1, axis=0)
                image[-1, :] = 0
            
            return image

    def __getitem__(self, idx):
        
        if self.dataset == 'train':
            if self.train_target == 'rec':
                file_name = random.choice(self.char_to_img[self.index_map[idx]])
            elif self.train_target == 'null':
                file_name = random.choice(self.null_to_img[self.index_map[idx]])
        elif self.dataset == 'dev':
            file_name = self.index_map[idx]
        file_name = '_'.join(file_name.split('.')[0].split('_')[1:])


        img = Image.open(f'./dataset/train_data/rec/char_img/img_{file_name}.jpg').convert("RGB")
        img = np.array(img)
        if self.dataset == 'train':
            img = self.augmentation(img)
        height, width = img.shape[0], img.shape[1]
        max_size = max(height, width)
        if max_size > 224:
            rr = 224/max_size
            img = cv2.resize(img, (min(224, max(5, int(height*rr))), min(224, max(5, int(width*rr)))), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.model_name == 'efficientnetb1':
            img = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 3, axis=2)
        else:
            img = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 1, axis=2)

        if self.train_target == 'rec':
            with open(f'./dataset/train_data/rec/char_gt/gt_{file_name}.pickle', 'rb') as file:
                gt = pickle.load(file)
            file.close()
            label = self.token_index[gt['ground_truth']] if gt['ground_truth'] in self.token_index else self.token_index['NULL']
        elif self.train_target == 'null':
            label = 0 if 'null' in file_name else 1

        target = {}

        if self.transforms is not None:
            img, _ = self.transforms(img, target)

        return img, label

    def __len__(self):
        return self.total

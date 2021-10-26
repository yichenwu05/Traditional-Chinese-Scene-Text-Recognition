import cv2
import PIL
import pickle
import numpy as np
import torch.utils.data
from PIL import Image, ExifTags, ImageFilter
import albumentations as A
from imgaug import augmenters as iaa
PIL.Image.MAX_IMAGE_PIXELS = 1e10

class StdInference(torch.utils.data.Dataset):
    def __init__(self, index_map_path, img_path, transforms=None):
        self.index_map_path = index_map_path
        self.img_path = img_path
        self.transforms = transforms
        with open(self.index_map_path, 'rb') as file:
            self.index_map = pickle.load(file)
        file.close()
        self.total = len(self.index_map)

    def __getitem__(self, idx):
        file_name = self.index_map[idx]
        img = Image.open(self.img_path+f'{file_name}.jpg').convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 1, axis=2)
        blur = min(img.shape[0], img.shape[1])
        k = int(blur*0.08)
        if k > 0:
            img = cv2.blur(img, (k, k))
        
        target = {}
        target["image_id"] = file_name
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.index_map)


class StdTrain(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        '''
        dataset : train, dev
        '''
        self.dataset = dataset
        self.transforms = transforms
        
        with open(f'./dataset/train_data/det/aicup_{self.dataset}_index_map.pickle', 'rb') as file:
            self.index_map = pickle.load(file)
        file.close()

        self.total = len(self.index_map)

    def __getitem__(self, idx):

        file_name = self.index_map[idx]
        
        img = Image.open(f'./dataset/train_data/det/img/img_{file_name}.jpg').convert("RGB")
        img = np.array(img)
        if self.dataset == 'train':
            img = self.augmentation(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.repeat(np.reshape(img, (img.shape[0], img.shape[1], 1)), 1, axis=2)
        blur = min(img.shape[0], img.shape[1])
        k = int(blur*0.08)
        if k > 0:
            img = cv2.blur(img, (k, k))
        
        with open(f'./dataset/train_data/det/gt/gt_{file_name}.pickle', 'rb') as file:
            gt = pickle.load(file)
        file.close()
        
        masks = Image.open(f'./dataset/train_data/det/mask/mask_{file_name}.png')

        masks = np.array(masks)
        masks = np.reshape(masks, (gt['num_objs'], int(masks.shape[0]/gt['num_objs']), masks.shape[1]))
        masks = masks.astype(np.uint8)
        boxes = gt['boxes']
        areas = gt['area']
        num_objs = gt['num_objs']

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = torch.tensor(areas)*1.0
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def augmentation(self, image):
        '''
        Size control and Augmentation for Image 
        '''
        # albumentations
        random_brightness_contrast = A.RandomBrightnessContrast(p=0.5)
        rgb_shift = A.RGBShift(p=0.3)
        is_opposite = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
        is_rgb_reverse = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
        is_noise = np.random.choice([0, 1, 2, 3], size=1, p=[0.7, 0.1, 0.1, 0.1])

        image = random_brightness_contrast(image=image)['image']
        image = rgb_shift(image=image)['image']
        if is_opposite:
            image = 255 - image
        if is_rgb_reverse:
            image = image[:, :, ::-1]
        if is_noise:
            if is_noise == 1:
                noise = iaa.SaltAndPepper(p=0.05)
            elif is_noise == 2:
                noise = iaa.AdditiveGaussianNoise(loc=0, scale=0.05*255)
            elif is_noise == 3:
                noise = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
            image = noise.augment_image(image)
        return image

    # def IsRotate(self, img): # 返回false表示存在旋转情况
    #     try:
    #         for orientation in ExifTags.TAGS.keys() :
    #             if ExifTags.TAGS[orientation]=='Orientation' :
    #                 img2 = img.rotate(0, expand = True)
    #                 break
    #         exif=dict(img._getexif().items())
    #         if  exif[orientation] == 3 :
    #             img2=img.rotate(180, expand = True)
    #         elif exif[orientation] == 6 :
    #             img2=img.rotate(270, expand = True)
    #         elif exif[orientation] == 8 :
    #             img2=img.rotate(90, expand = True)

    #         return img2.convert("RGB")
    #     except:
    #         return img.convert("RGB")

    def __len__(self):
        return self.total
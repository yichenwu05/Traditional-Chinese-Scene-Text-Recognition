## **繁體中文場景文字辨識競賽－進階賽：繁體中文場景文字辨識**

[AI CUP Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/16)

Team Info           |                               |
--------------------|-------------------------------|
Team Name           | GPU不夠了                      |
Team Member         | 吳亦振 (Wu, Yi-Chen)           |
Public Score        | 0.876377                      |
Private Score       | 0.868158                      |
Private Leaderboard | 6th/183 ( Top 4% )             | 


## `Environment`

Equipment and version    |                    |
------------|-------------------------------------------|
OS          | Ubuntu 16.04.6 LTS                        |
Image       | pytorch/pytorch:1.3-cuda10.1-cudnn7-devel |
Python      | 3.6.9                                     |


## `Demo`
You can put your own cropped image in `./dataset/demo_imgs/` and checkout `Demo.ipynb`.


## `Inference Private Dataset`

Download `Private_Dataset.zip` ( already decrypt ) from google drive and unzip.

```
$ cd ./dataset/private_data/
$ gdown https://drive.google.com/uc?id=1pJNbKPkICeMUrvq8IcihVgZe5JgHXUC1
$ unzip PD.zip 
$ cd ../..
```

Download pretrained models from google drive and unzip.

```
$ cd ./model/weight/
$ gdown https://drive.google.com/uc?id=1iFjqJM-ZDEOurIezPpcG7TN9B5RnKrdL
$ unzip weight.zip
$ cd ../..
```

Inference private dataset


Method 1 : Run shell script
```
$ sh ./inference.sh private
```

Method 2 : Run python script
```
$ python ./inference/step1_croptextarea.py private
$ python ./inference/step2_detectchar.py private
$ python ./inference/step3_char_cls.py private
$ python ./inference/step4_nlp_thres.py private
```

The output will save to `./output/` folder

## `Train(Optional)`

Download `train.zip` from google drive and unzip.

```
$ cd ./dataset/train_data/aicup
$ gdown https://drive.google.com/uc?id=1g0debEGTFWvqb_Ht2xLDNQbaDZ148Hd4
$ unzip train_rec.zip
$ cd ../../..
```

Data preprocessing

Method 1 : Run shell script
```
$ sh ./preprocess.sh
```

Method 2 : Run python script
```
$ python ./preprocess/gen_train_det_data.py
$ python ./preprocess/gen_train_rec_data_1.py
$ python ./preprocess/gen_train_rec_data_2.py
$ python ./preprocess/gen_train_rec_data_3.py
$ python ./preprocess/gen_train_rec_data_4.py
$ python ./preprocess/gen_train_rec_data_5.py
```

Model training

Train detection model
```
$ python train_det.py
```

Train null model (whether the image contains Traditional Chinese character or not)

```
$ python train_rec_null.py null resnet18
```

Train recognition model

```
$ python train_rec_null.py rec resnet50
```

### configuration

| Name                | Key            | Description                             | Sample Value  |
| ------------------- | ---------------|---------------------------------------- | ------------- |
|./configs/device.conf| TrainDevice    | Device to train (0, 1, 2, cpu, ..)      | 0             |
|./configs/device.conf| InferenceDevice| Device to inference (0, 1, 2, cpu, ..)  | 0             |




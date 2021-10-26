import torch
import torch.optim as optim
from utils import utils
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
from module.std_dataloader import StdTrain
from module.std_net import get_instance_segmentation_model
from pathlib import Path
import yaml

with Path('configs/device.conf').open('r', encoding='utf-8') as rf:
    device_conf = yaml.load(rf)

device = device_conf['TrainDevice']
device = torch.device('cuda:%d' % (
    device) if device != 'cpu' else 'cpu')


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


dataset_train = StdTrain('train', get_transform(train=True))
dataset_dev = StdTrain('dev', get_transform(train=False))

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_dev = torch.utils.data.DataLoader(
    dataset_dev, batch_size=8, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# the dataset has two classes only - background and person
num_classes = 2

# get the model using the helper function
model = get_instance_segmentation_model(num_classes, device)

for param in model.parameters():
    param.requires_grad = True


optimizer = optim.SGD(
    [
        {"params": model.backbone.parameters(), "lr": 0.005},
        {"params": model.rpn.parameters(), "lr": 0.02},
        {"params": model.roi_heads.parameters(), "lr": 0.02}
    ],
    lr=0.005,
    momentum=0.9, 
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)


num_epochs = 30
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=300)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_dev, device=device)
    # save model weight 
    torch.save(model.state_dict(), './model/weight/detect_resnet152_{}.ckpt'.format(epoch+9))

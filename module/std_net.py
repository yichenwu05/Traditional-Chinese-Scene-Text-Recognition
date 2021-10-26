import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def maskrcnn_resnet152_fpn(progress=True, num_classes=2, pretrained_backbone=True, **kwargs):
    anchor_sizes = ((24,), (48,), (64,), (96,), (128,))
    aspect_ratios = ((0.5, 1.0, 1.5, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    backbone = resnet_fpn_backbone("resnet152", pretrained_backbone)
    model = MaskRCNN(backbone, num_classes, rpn_anchor_generator=rpn_anchor_generator, **kwargs)
    return model    

def get_instance_segmentation_model(num_classes, device, load_model=None):
    # load an instance segmentation model pre-trained on COCO

    model = maskrcnn_resnet152_fpn(num_classes=num_classes)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if load_model:
        model.load_state_dict(torch.load('./model/weight/{}.ckpt'.format(load_model), map_location=device), strict=False)
    model = model.to(device)
    return model    
import cv2
import torch
import numpy as np
from module.nms import soft_nms
from shapely.geometry import Polygon


def mask_bb(_mask):
    v = np.sum(_mask, axis=1)
    h = np.sum(_mask, axis=0)
    _ymin, _ymax = 0, len(v)
    _xmin, _xmax = 0, len(h)
    
    # find xmin and xmax
    for i, k in enumerate(h):
        if k > 0:
            _xmin = i
            break
    for i, k in enumerate(h[::-1]):
        if k > 0:
            _xmax = _xmax-i-1
            break
                
    # find ymin and ymax  
    for i, k in enumerate(v):
        if k > 0:
            _ymin = i
            break
            
    for i, k in enumerate(v[::-1]):
        if k > 0:
            _ymax = _ymax-i-1
            break
    
    return [_xmin, _ymin, _xmax, _ymax]


def show_prediction(dataloader, model, device, idx, img_path, show=True, threshold=[0.5, 0.7, 0.9]):
    
    img, info = dataloader[idx-1]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        
    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    valid_boxes, _ = to_dets_score(boxes.tolist(), scores.tolist(), threshold)
    img = cv2.imread(img_path+'{}.jpg'.format(info['image_id']))
    if show:
        print(info['image_id'])
    res = []
    for index in valid_boxes:
        bb = np.round(boxes[index]).astype(int)
        res.append(list(bb))
        img = cv2.polylines(img, [np.array([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])], True, (0, 0, 255), 1)
    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(img[:,:,::-1])
        plt.show()
    if device.type != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return info['image_id'], img, res
        

def to_dets_score(boxes, scores, threshold=[0.5, 0.7, 0.9]):
    dets, areas = [], []
    for i in range(len(boxes)):
        dets.append(boxes[i] + [scores[i]])
        areas.append((boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]))
    dets = np.array(dets)
    result = soft_nms(dets, 0.5)
    
    valid_index = []
    valid_areas = []
    for i in result:
        if areas[i] < 1024:
            if scores[i] >= threshold[0]:
                valid_index.append(i)
                valid_areas.append(areas[i])
        elif areas[i] < 9216:
            if scores[i] >= threshold[1]:
                valid_index.append(i)
                valid_areas.append(areas[i])
        else:
            if scores[i] >= threshold[2]:
                valid_index.append(i)
                valid_areas.append(areas[i])

    return valid_index, valid_areas

def valid_box(boxes):
    res = []
    for i in range(len(boxes)):
        valid = 1
        for j in range(len(boxes)):
            if i != j:
                if not Within(boxes[i], boxes[j]):
                    valid = 0
                    break
        if valid:
            res.append(boxes[i])
    return res


def Within(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    within = interArea/boxAArea
    if within > 0.7:
        return 0
    return 1


def in_poly(line_poly, boxes):
    res = []
    # xmin, ymin, xmax, ymax
    for box in boxes:
        pts = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        q = Polygon(pts)
        if (line_poly.intersection(q).area)/((box[2]-box[0])*(box[3]-box[1])) >= 0.5:
            res.append(box)
    return res

def mean_slope(bb):
    if len(bb) == 1:
        return bb, 0
    bb = sorted(bb, key=lambda x: x[0])
    
    center = []
    for b in bb:
        center.append(((b[0]+b[2])/2, (b[1]+b[3])/2))
    
    slopes = []
    for i in range(len(bb)-1):
        slopes.append(abs((center[i+1][1]-center[i][1])/(center[i+1][0]-center[i][0]+1e-5)))
        
    ms = np.mean(slopes)
    
    if ms>1.73:
        bb = sorted(bb, key=lambda x: x[1])
        
    return bb, ms
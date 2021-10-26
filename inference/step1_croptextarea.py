import os
import sys
import cv2
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

def crop_and_mask(points, img):
    points = [x if x > 0 else 0 for x in points]
    pts = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]])
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    return pts, croped

if __name__ == '__main__':

    inference_data = sys.argv[1] # public private
    csv_name =inference_data[0].upper()+inference_data[1:]
    
    csv_filename = f'./dataset/{inference_data}_data/{inference_data}/Task2_{csv_name}_String_Coordinate.csv'
    img_folder = f'./dataset/{inference_data}_data/{inference_data}/img_{inference_data}/'
    img_savepath = f'./dataset/{inference_data}_data/img/'
    pickle_savepath = f'./dataset/{inference_data}_data/'

    if not os.path.isdir(img_savepath):
        os.mkdir(img_savepath)

    df = pd.read_csv(csv_filename, header=None)

    test_img_points = {}
    for i in range(len(df)):
        filename = df.loc[i][0]
        points = list(df.loc[i][1:])
        if filename not in test_img_points:
            test_img_points[filename] = []
        test_img_points[filename].append(points)

    test_index_map = {}
    test_img_poly_pts = {}
    i = 0
    for k, v in tqdm(test_img_points.items()):
        for j in range(len(v)):
            test_index_map[i] = k+'_'+str(j+1)
            i += 1 
            pts = v[j]
            img = cv2.imread(img_folder+f'{k}.jpg')
            pts, croped = crop_and_mask(pts, img)
            test_img_poly_pts[k+'_'+str(j+1)] = pts.tolist()
            cv2.imwrite(img_savepath+'{}.jpg'.format(k+'_'+str(j+1)), croped)

    with open(pickle_savepath+'index_map.pickle', 'wb') as output_file:
        pickle.dump(test_index_map, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

    with open(pickle_savepath+'test_img_poly_pts.pickle', 'wb') as output_file:
        pickle.dump(test_img_poly_pts, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

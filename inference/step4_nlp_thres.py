import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':

    inference_data = sys.argv[1]
    csv_name =inference_data[0].upper()+inference_data[1:]

    ans_path = f'./dataset/{inference_data}_data/private_submit.pickle'
    csv_filename = f'./dataset/{inference_data}_data/{inference_data}/Task2_{csv_name}_String_Coordinate.csv'
    output_path = f'./output/'

    with open(ans_path, 'rb') as output_file:
        candidate_ans = pickle.load(output_file)
    output_file.close()

    ## if you have 2 or more output, you can ensemble probability scores 
    '''
    with open(f'./dataset/{inference_data}_data/private_submit1.pickle', 'rb') as output_file:
        candidate_ans1 = pickle.load(output_file)
    output_file.close()

    with open(f'./dataset/{inference_data}_data/private_submit2.pickle', 'rb') as output_file:
        candidate_ans2 = pickle.load(output_file)
    output_file.close()

    candidate_ans = {}
    for k, v in candidate_ans1.items():
        candidate_ans[k] = []
        v1 = candidate_ans1[k]
        v2 = candidate_ans2[k]

        for i in range(len(v1)):
            candi_tmp = {} 
            for j, l in enumerate(v1[i]['candidates']):
                if l not in candi_tmp:
                    candi_tmp[l] = []
                candi_tmp[l].append(v1[i]['prob'][j])
                
            for j, l in enumerate(v2[i]['candidates']):
                if l not in candi_tmp:
                    candi_tmp[l] = []
                candi_tmp[l].append(v2[i]['prob'][j])
            
            candi_tmp = {k: np.max(v) for k, v in candi_tmp.items()}
            candi_tmp = dict(sorted(candi_tmp.items(), key=lambda x: x[1], reverse=True))
            candidate_ans[k].append({'candidates': [k for k, _ in candi_tmp.items()], 'prob': [v for _, v in candi_tmp.items()]})
    '''
    ans = []
    for k, v in tqdm(candidate_ans.items()):
        res = ''
        if len(v) == 1 and v[0]['candidates'] == 'NULL':
            res += '#'
        else:
            if len(v) == 1 and v[0]['prob'][0] < 0.5:
                res += '#'
            else:
                for j in v:
                    if j['prob'][0] < 0.35:
                        res += '#'
                    else:
                        if j['candidates'][0] == 'NULL':
                            res += '#'
                        else:
                            res += j['candidates'][0]
        ans.append(res)

    subdat = pd.read_csv(csv_filename, header=None)

    sub_ans = ['###' if y == '' else y for y in [x.replace('#', '') for x in ans]]

    sub_col = {}
    for i, k in enumerate(candidate_ans):
        sub_id = '_'.join(k.split('_')[:2])
        _id = k.split('_')[2]
        if sub_id not in sub_col:
            sub_col[sub_id] = {}
        sub_col[sub_id][int(_id)] = sub_ans[i]

    ress = []
    seen_id = set()
    for kk in subdat[0]:
        if kk in seen_id:
            continue
        seen_id.add(kk)
        for i in range(len(sub_col[kk])):
            ress.append(sub_col[kk][i+1])

    subdat[9] = ress
    subdat.to_csv(output_path+f'{inference_data}_submit.csv', header=None, encoding='utf_8_sig', index=0)
    
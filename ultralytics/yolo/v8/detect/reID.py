import pickle
import os
import cv2
import shutil
import torch
import numpy as np
from collections import *

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    # print("+++++++++++++++++++++", qf.shape, gf.shape)
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def convert_point(M, y, x):
    p = np.matmul(M, [y, x, 1])
    y, x = int(p[0] / p[2]), int(p[1] / p[2])
    return y, x

def match_id(distmat, q_pid, g_pids, forbidden):
    indices = np.argsort(distmat, axis=1)
    whole = Counter(g_pids[indices[:, 0]])
    query_ans, cnt  = whole.most_common(1)[0]
    
    if query_ans in forbidden:
        return q_pid
    
    mask = g_pids[indices[:, 0]] == query_ans
    
    # if np.mean([distmat[i, x] for i, x in enumerate(indices[:, 0])]) > 0.9:
    # if np.mean([distmat[i, x] for i, x in enumerate(indices[:, 0])]) > 0.8 or np.std([distmat[i, x] for i, x in enumerate(indices[:, 0])]) > 0.06:
    #     return q_pid
    
    if cnt > 0.7 * len(indices) and np.mean([distmat[i, x] for i, x in enumerate(indices[:, 0]) if mask[i]]) <= 0.9:
    # if cnt > 0.7 * len(indices) and np.mean([distmat[i, x] for i, x in enumerate(indices[:, 0]) if mask[i]]) <= 0.9 and np.std([distmat[i, x] for i, x in enumerate(indices[:, 0]) if mask[i]]) <= 0.09:
    # if cnt > 0.9 * len(indices):        
        # print(cnt / len(indices))
        return query_ans
    else:
        return q_pid
    
feats = torch.load('feats/feats.pt')

with open('feats/pids.pkl', 'rb') as f:
    pids = pickle.load(f)
    
with open('feats/frames.pkl', 'rb') as f:
    frames = pickle.load(f)
    
frames = np.array(frames)
pids = np.array([str(x).zfill(15) for x in pids])

print("============================== start reID ===============================")

dic_reID = dict()
visited = set()
start_frame = min(frames)
ind = 0
for x in pids[frames == start_frame]:
    ind += 1
    visited.add(x)

res = []
while ind < len(pids):
    if pids[ind] not in visited:
        past_ind = ind
        while pids[past_ind] == pids[ind]:
            past_ind -= 1
        gf = feats[:past_ind + 1]
        qf = feats[pids == pids[ind]][:100]
        # distmat = euclidean_distance(qf, gf)
        distmat = cosine_similarity(qf, gf)
        res.append(distmat.shape)
        visited.add(pids[ind])
        
        q_pid = pids[ind]
        g_pids = pids[:past_ind + 1]
        
        forbidden = pids[frames == frames[ind]]
        
        query_ans = match_id(distmat, q_pid, g_pids, forbidden)
        if query_ans != q_pid:
            pids[pids == q_pid] = query_ans
            dic_reID[q_pid] = query_ans
    ind += 1
    
with open('dic_reID.pkl', "wb") as file:
    pickle.dump(dic_reID, file)
    
if os.path.exists('./reID_new'):
    shutil.rmtree('./reID_new')
os.makedirs('./reID_new')

if os.path.exists('./reID_overlook'):
    shutil.rmtree('./reID_overlook')
os.makedirs('./reID_overlook')

print("============================== write reID_new ==================================")

# B = M * A，获得透视转换矩阵 M
A = [(520, 760), (1720, 870), (870, 870), (1290, 700)]
B = [(1078, 568), (791, 449), (963, 449), (871, 649)]

pts1 = np.float32(A)
pts2 = np.float32(B)
M = cv2.getPerspectiveTransform(pts1,pts2)

files = os.listdir('./reID')
for file in files:
    with open('./reID/' + file, 'r') as f:
        lines = f.readlines()
        lines_overlook = lines[:]
    for i in range(len(lines)):
        line = lines[i]
        if 'id:' in line:
            pos_str, id_str = line.split('id:')
            pos_str = pos_str[5:-2]
            x1, x2, y1, y2 = map(int, pos_str.split(','))
            query_id = id_str.strip()
            if query_id in dic_reID:
                lines[i] = f"pos:[{x1}, {x2}, {y1}, {y2}] id:{dic_reID[query_id]}\n"
    with open('./reID_new/' + file, 'w') as f:
        f.writelines(lines)
        
# 透视变换后的轨迹
files = os.listdir('./reID_new')
for file in files:
    with open('./reID/' + file, 'r') as f:
        lines = f.readlines()
        lines_overlook = lines[:]
    for i in range(len(lines)):
        line = lines[i]
        if 'id:' in line:
            pos_str, id_str = line.split('id:')
            pos_str = pos_str[5:-2]
            x1, x2, y1, y2 = map(int, pos_str.split(','))
            y11, x11 = convert_point(M, y1, x1)
            y22, x22 = convert_point(M, y2, x2)
            query_id = id_str.strip()
            lines_overlook[i] = f"pos:[{x11}, {x22}, {y11}, {y22}] id:{query_id}\n"
                
    with open('./reID_overlook/' + file, 'w') as f:
        f.writelines(lines_overlook)
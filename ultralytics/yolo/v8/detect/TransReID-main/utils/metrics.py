import torch
import numpy as np
import os
import pickle
from datetime import datetime
from bisect import bisect_left
from collections import Counter
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    print("+++++++++++++++++++++", qf.shape, gf.shape)
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


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    # assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.hasData = False

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.hasData = True
        
    def check(self):
        if self.hasData == True:
            return True
        else:
            return False
        
    def print_dataset_statistics(self):
        q_pids = np.asarray(self.pids[-self.num_query:])
        q_camids = np.asarray(self.camids[-self.num_query:])
        g_pids = np.asarray(self.pids[:-self.num_query])
        g_camids = np.asarray(self.camids[:-self.num_query])
        # print("@@@@@@@@@@@@@----length of self.feats:", len(self.feats), len(self.pids))
        print("  -----------------------------------------------")
        print("  subset   |    # ids   |  # images  | # cameras")
        print("  -----------------------------------------------")
        print("  query    | {:10d} | {:10d} | {:9d}".format(len(set(q_pids)), len(q_pids), len(set(q_camids))))
        print("  gallery  | {:10d} | {:10d} | {:9d}".format(len(set(g_pids)), len(g_pids), len(set(g_camids))))
        print("  -----------------------------------------------")
        
    def change(self, num_query):
        self.num_query = num_query

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!update")
        # print(len(self.pids))
        self.pids.extend(np.asarray(pid))
        # print(len(self.pids))
        self.camids.extend(np.asarray(camid))
        
    def match_id(self, distmat, q_pids, g_pids, final_output, max_M=10):
        num_q, num_g = distmat.shape
        if num_g < max_M:
            max_M = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        
        # 将每个 query 的匹配结果输出
        # test07.mp4 的参数
        # shreshold1 = 0.22
        # shreshold2 = 0.5
        # test04.mp4 的参数
        shreshold1 = 0.4
        shreshold2 = 0.7
        max_M = 16
        
        cnt_error_match = 0
        cnt_not_found = 0
        final_output_path = '../../output/final_match/' + final_output + '.txt'
        with open(final_output_path, 'w') as file:
            for q_idx in range(num_q):
                q_pid = q_pids[q_idx]
                idx_shreshold = bisect_left(distmat[q_idx][indices[q_idx][:max_M]], shreshold2)
                if idx_shreshold == 0:
                    idx_shreshold = 1
                inds_needed = indices[q_idx][:idx_shreshold]
                top_M_g_pids = g_pids[inds_needed]
                
                dic = Counter(top_M_g_pids)
                mx = max(dic.values())
                
                # not match: 最匹配的分数 > 阈值 or 匹配序列均值 > 2 * 阈值 or 出现次数最多的 id 的出现次数小于总数的 1/3
                if distmat[q_idx][indices[q_idx][0]] > shreshold1 or distmat[q_idx][inds_needed].mean() > (shreshold1 * 1.6) :#or mx < (len(top_M_g_pids) // 4):
                    cnt_not_found += 1
                    # print 偏高的 error match count
                    self.pids[-self.num_query + q_idx] = q_pid
                    # print 真实的 error match count
                    # self.pids[-self.num_query + q_idx] = q_pid + 100000000000000
                    file.write(f"q_pid: {q_pid}, g_pid: {q_pid + 100000000000000}, count: {'null'}, not found\n")
                    continue
                    
                for k, v in dic.items():
                    if v == mx:
                        file.write(f"q_pid: {q_pid}, g_pid: {k}, count: {v}\n")
                        # error match
                        if self.pids[-self.num_query + q_idx] != k:
                            cnt_error_match += 1
                            self.pids[-self.num_query + q_idx] = k
                        break
        
        print("################## ", final_output, " not found count:", cnt_not_found)
        # print("################## ", final_output, " images num:" ,self.num_query , " error match count:", cnt_error_match, "; not found count:", cnt_not_found)
        query_ans_ids = self.pids[-self.num_query:]
        return query_ans_ids

    def compute(self, final_output="query"):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            # print("The test feature is normalized")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = "The test feature is normalized"
            print(f"{current_time} {log_message}")
            
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        # qf = feats[:self.num_query]
        # q_pids = np.asarray(self.pids[:self.num_query])
        # q_camids = np.asarray(self.camids[:self.num_query])
        # print(feats[0].shape)
        
        qf = feats[-self.num_query:]
                
        q_pids = np.asarray(self.pids[-self.num_query:])
        q_camids = np.asarray(self.camids[-self.num_query:])
        
        flag = False
        if flag:
            print(qf.shape, q_pids.shape, q_camids.shape)
            with open('../../output/query_info.txt', 'w') as file:
                for i in range(len(q_pids)):
                    file.write(f"Query {i+1}:\n")
                    file.write(f"Feature: {qf[i]}\n")
                    file.write(f"PID: {q_pids[i]}\n")
                    file.write(f"CamID: {q_camids[i]}\n")
                    file.write("\n")
            print("Query information has been written to query_info.txt")
        
        # gallery
        # gf = feats[self.num_query:]
        # g_pids = np.asarray(self.pids[self.num_query:])
        # g_camids = np.asarray(self.camids[self.num_query:])
        gf = feats[:-self.num_query]
        g_pids = np.asarray(self.pids[:-self.num_query])
        g_camids = np.asarray(self.camids[:-self.num_query])
        
        if flag:
            print(gf.shape, g_pids.shape, g_camids.shape)
            with open('../../output/gallery_info.txt', 'w') as file:
                for i in range(len(g_pids)):
                    file.write(f"Gallery {i+1}:\n")
                    file.write(f"Feature: {gf[i]}\n")
                    file.write(f"PID: {g_pids[i]}\n")
                    file.write(f"CamID: {g_camids[i]}\n")
                    file.write("\n")
            print("Gallery information has been written to gallery_info.txt")
        
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            # print(self.num_query)
            # print(qf, gf)
            distmat = euclidean_distance(qf, gf)
            # with open('dist_mat.pkl', 'wb') as f:
            #     pickle.dump(distmat, f)
        
        query_ans_ids = self.match_id(distmat, q_pids, g_pids, final_output)
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf, query_ans_ids


    def get_feats(self, final_output="query"):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            # print("The test feature is normalized")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = "The test feature is normalized"
            print(f"{current_time} {log_message}")
            
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print(feats.shape)
            print(len(self.pids))
            print(self.pids[0], self.pids[-1])
            
            pids = [int(str(x)[8:]) for x in self.pids]
            frames = [int(str(x)[1:8]) for x in self.pids]
            # print(frames)
            
            with open('../feats/pids.pkl', 'wb') as f:
                pickle.dump(pids, f)
            with open('../feats/frames.pkl', 'wb') as f:
                pickle.dump(frames, f)
            
            torch.save(feats, '../feats/feats.pt')
            
            
            # l, r = 0, 0
            # while r < len(frames):
            #     if frames[l] != frames[r]:
            #         feat = feats[l:r]
            #         torch.save(feat, '../feats/' + str(frames[l]) + '.pt')
            #         l = r
            #     r += 1
            # feat = feats[l:r]
            # torch.save(feat, '../feats/' + str(frames[l]) + '.pt')
            
            
            # unique_frames, frame_indices = torch.unique(torch.tensor(frames), return_inverse=True)
            # # 计算每个帧对应的特征数量
            # frame_counts = torch.bincount(frame_indices)
            # # 保存每个帧的特征
            # start = 0
            # for count in frame_counts:
            #     end = start + count.item()
            #     feat = feats[start:end]
            #     frame = unique_frames[start].item()
            #     torch.save(feat, f'../feats/{frame}.pt')
            #     start = end
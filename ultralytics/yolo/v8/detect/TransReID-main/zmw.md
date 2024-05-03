## 1.27

问题：gallery 中的数据会随着 query 次数的增加而变多，使计算速度放缓
- 将每次 query 的结果图片也放入一个 new_gallery 文件夹下，记录 query 次数 query_cnt
- 对 gallery 中的数据进行监控，设置一个数据量阈值 如 1e5 
- 若超过 1e5，整理 new_gallery 文件夹下的数据，每一个 pid 对应的图片最多只保留 query_cnt // 10 张
- 将 new_gallery 中的数据追加进 gallery 所在文件夹 'bounding_box_test' 下
- 重新初始化任务，有效减少数据量和计算量

## 1.28

任务拆分：ReID串行化，图片的特征提取部分并行化

- 特征提取 feat = model(img, cam_label=camids, view_label=target_view)，可以在多张 gpu 上并行执行
- 现有的 ReID 依赖于计算 feat 的 欧氏距离，计算 query 中 q_pid 对于 gallery 中的 g_pid 最短的欧氏距离，且小于一定阈值的，判断为同一目标
1. 这么做需遍历 gallery 中的全部 n 条数据，若有 m 条query，复杂度为 m * n
2. 考虑到 m 不可压缩，而目标是查询 n 条数据中，距离被查询数据距离最近的一个，可利用 geohash 的理论 有效减少被查询数据
3. vit_small 提取的图片特征维度为 11，如果对每一个维度二分，编号 0 和 1，构成的11位二进制字符串可将特征空间划分为 2 ** 11 份
4. 按由特征 feat 算出的11位二进制字符串进行空间匹配，在数据足够充分且均匀的情况下，可以将被查询数据范围缩减为原来的 1 / 2048，极大减小计算量

## 1.29

ReID并行化（ReID能力有所降低，计算时间减少）

- 将视频分成 n 份，分别进行 ReID，结束后获得 n 个 gallery 文件夹,  $gallery_{1}$ - $gallery_{n}$
- 对 $gallery_{i}$ 下的文件进行整理，每个 pid 保留 1 个 （保留策略待定）
- 从 $gallery_{2}$ 开始，遍历文件夹下的所有 q_pid，匹配 $gallery_{1}$ 中的图片 g_pid，若匹配成功，修改 q_pid
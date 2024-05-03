#!/bin/bash

current_dir=$(pwd)

video_file="test3.mp4"

cd $current_dir
source activate smallscene

# 利用 yolo + deepsort 对视频中的人物进行识别，进行初步的 reID，并提取人物图片到 ./data/Small_Scenes_1，将初步识别的人物 position 和 id 记录在 ./reID 中
python predict.py model=yolov8l.pt source="$video_file"

# 利用 Vit 对提取出的人物图片进行特征提取，记录在 ./feats 中
cd TransReID-main
python ./test.py --config_file configs/DukeMTMC/vit_transreid_stride.yml
cd ..

# 利用余弦相似度将首次出现的人物 id 与既往进行匹配，对人物进行进一步的重识别，识别结果放入 ./reID_new 中
# 同时将轨迹从 透视视角 向 俯视视角转换，转换结果放在 ./reID_overlook 
python ./reID.py

# 输出透视视角下的人物轨迹，视频在 ./runs/detect
python output.py model=yolov8l.pt source="$video_file"

# 输出俯视视角下的人物轨迹，视频在 ./
python output_overlook.py model=yolov8l.pt source="$video_file"


#!/bin/bash

current_dir=$(pwd)

# 创建一个视频文件列表
# video_files=("c2.mp4")
video_files=("c1.mp4" "c2.mp4")

# 遍历每个视频文件
for i in "${!video_files[@]}"
do
    echo "Processing ${video_files[$i]}..."

    # 执行yolo + deepsort, 获取人物 方框 和 id
    # cd $current_dir
    # source activate detect
    # python predict.py model=yolov8l.pt source="${video_files[$i]}"

    # 执行vit ，提取图片特征
    source activate chatGlm
    cd TransReID-main
    python ./test.py --config_file configs/DukeMTMC/vit_transreid_stride.yml --video_path "${video_files[$i]}"
    cd ..

    # 特征相似度匹配，进行重识别
    # 以及进行 透视转换矩阵 的生成
    video_pathes=("${video_files[@]:0:$((i+1))}")
    python ./reID.py --video_pathes "${video_pathes[*]}"

    # 生成原始视角视频
    source activate detect
    python output_origin.py "${video_files[$i]}"

    # 生成俯视视角视频
    source activate detect
    python output_overlook.py "${video_files[$i]}"

    echo "Finished processing ${video_files[$i]}."
done

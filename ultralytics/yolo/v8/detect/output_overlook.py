import copy
from collections import deque
import cv2
import os
import random
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
last_occur = {}

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0), frame=None):
    vis = set()
    thickness = 1
    img_copy = copy.deepcopy(img)
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # center = (int((x2+x1)/ 2), int((y2+y2)/2))
        center = (int((x2+x1)/ 2), y2)
        id = int(identities[i]) if identities is not None else 0
        vis.add(id)
        last_occur[id] = frame
        
        first_occur = False

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen=1500)
          first_occur = True
        color = compute_color_for_labels(id)

        data_deque[id].appendleft(center)
        label = '{}{:d}'.format("", id) + ":"+ '%s' % ("person")
        UI_box(box, img, img_copy, label=label, color=color, line_thickness=2, frame=frame, first_occur=first_occur)
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
            
    # 绘制 已经离开视频范围的目标 和 追踪丢失的目标 的轨迹
    for k, v in data_deque.items():
        if k not in vis and frame - last_occur[k] <= 250:
            # print("1111111111111111111111", k, frame, last_occur[k])
            color = compute_color_for_labels(k)
            for i in range(1, len(v)):
            # check if on buffer value is none
                if v[i - 1] is None or v[i] is None:
                    continue
                # draw trails
                cv2.line(img, v[i - 1], v[i], color, thickness)
    
    return img

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, img_copy, color=None, label=None, line_thickness=None, frame=None, first_occur=False):
    idx = label.split(":")[0]
    color = compute_color_for_labels(int(label.split(":")[0]))
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    c1 = (int((x[2] + x[0]) / 2), int(x[1]))
    
    if "person" in label:
        label = label.split(":")[0].zfill(15)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0] + 3, c1[1]), color, 1, 8, 2)
        # print(label)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



video_path = 'test3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("视频的总帧数:", total_frames)


image = cv2.imread('./images/1.png')
height, width, _ = image.shape

fourcc = cv2.VideoWriter_fourcc(*'H264')

video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 25, (width, height))

for frame in range(total_frames):
    image1 = image.copy()
    if frame % 100 == 0 or frame == total_frames - 1:
        print(frame)
    outputs = []
    folder_path = './reID_overlook/' + str(frame - 3) + '.txt'
    if frame >= 3 and os.path.exists(folder_path):
        with open(folder_path, 'r') as f:
            for line in f:
                pos_str, id_str = line.split('id:')
                pos_str = pos_str[5:-2]
                x1, x2, y1, y2 = map(int, pos_str.split(','))
                q_id = int(id_str.strip())
                outputs.append([y1, x1, y2, x2, q_id, 0])
    outputs = np.array(outputs)
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -2]
        object_id = outputs[:, -1]
        image1 = draw_boxes(image1, bbox_xyxy, identities, frame=frame)
    
    video_writer.write(image1)

print("生成完毕")
video_writer.release()
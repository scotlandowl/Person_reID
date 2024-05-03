import os
import shutil

folder_path = '../../reID/'
folder_path_new = '../../reID_new/'

if os.path.exists(folder_path_new):
    shutil.rmtree(folder_path_new)
    
os.makedirs(folder_path_new)

files = os.listdir(folder_path)
g_pids = dict()
for file in files:
    # print(file)
    file_suf = int(file[:-4])
    if file_suf == 0:
        continue
    elif file_suf == 1:
        file_name = 'query'
    else:
        file_name = 'query_' + str(file_suf - 1)
    whole_name = '../../output/final_match/' + file_name + '.txt'
    if not os.path.exists(whole_name):
        continue
    with open(whole_name, 'r') as f:
        # print(whole_name)
        for line in f:
            q_pid = int(line.split(', ')[0].split(':')[1])
            g_pid = int(line.split(', ')[1].split(':')[1])
            g_pids[q_pid] = g_pid
    with open(folder_path + file, 'r') as f:
        lines = f.readlines() 
    for i in range(len(lines)):
        line = lines[i]
        if 'id:' in line:
            pos_str, id_str = line.split('id:')
            pos_str = pos_str[5:-2]
            x1, x2, y1, y2 = map(int, pos_str.split(','))
            query_id = int(id_str.strip())
            # print(x1, x2, y1, y2, query_id, g_pids[query_id])
            lines[i] = f"pos:[{x1}, {x2}, {y1}, {y2}] id:{g_pids[query_id]}\n"
            
    with open(folder_path_new + file, 'w') as f:
        f.writelines(lines)
        
print("done")
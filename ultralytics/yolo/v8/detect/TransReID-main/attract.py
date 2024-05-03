import os
import shutil
import torch
import torch.nn as nn
from config import cfg
import argparse
from datetime import datetime
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "starting"
    print(f"{current_time} {log_message}")
    
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "val_loader is ready"
    print(f"{current_time} {log_message}")

    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
        device = "cuda"
        if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)
        model.eval()
        
        folder_final_match = '../../output/final_match/'
        if os.path.exists(folder_final_match):
            shutil.rmtree(folder_final_match)
        os.makedirs(folder_final_match)
        
        if os.path.exists('../../output/query_ans/'):
            shutil.rmtree('../../output/query_ans/')
        os.makedirs('../../output/query_ans/')
        
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        
        _, _, val_loader, num_query, _, _, _ = make_dataloader(cfg, isInit=True, query_path='query')
        evaluator.change(num_query=num_query)
        do_inference(evaluator,
                    model,
                    val_loader,
                    num_query)
       
        print("!!!!!")
        
        for i in range(1, 15150):
            q_path = 'query_' + str(i)
            if not os.path.exists('../../data/Small_Scenes_1/' + q_path):
                break
            _, _, query_loader, num_query, _, _, _ = make_dataloader(cfg, isInit=False, query_path=q_path)
            evaluator.change(num_query=num_query)
            do_inference(evaluator, model, query_loader, num_query, query_path_name=q_path)
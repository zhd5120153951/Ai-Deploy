nohup python3 -m torch.distributed.launch --nproc_per_node 4 train_simpleedu_yiyuan.py --batch-size 8  >logs/log_simpleedu_yiyuan.log 2>&1 &

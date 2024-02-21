nohup python3 -m torch.distributed.launch --nproc_per_node 4 train_yaban.py --batch-size 16  >logs/log_yaban.log 2>&1 &

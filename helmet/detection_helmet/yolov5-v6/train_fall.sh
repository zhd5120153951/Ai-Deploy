nohup python3 -m torch.distributed.launch --nproc_per_node 4 train_fall.py --batch-size 16  >logs/log_fall.log 2>&1 &

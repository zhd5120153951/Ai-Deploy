nohup python3 -m torch.distributed.launch --nproc_per_node 4 train_fire_smoke.py --batch-size 16  >logs/log_fire_smoke.log 2>&1 &

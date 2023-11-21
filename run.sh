CONFIG=$1

python tracking/train.py --script ostrack --config ${CONFIG} --save_dir ./work_dirs/ --mode multiple --nproc_per_node 1 --use_wandb 0

python tracking/test.py ostrack ${CONFIG} --dataset lasot --threads 16 --num_gpus 8
python tracking/test.py ostrack ${CONFIG} --dataset trackingnet --threads 16 --num_gpus 8
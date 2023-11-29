# CLIP-MoCo
python main.py --model MoCo --wandb True --temperature 0.1

# CLIP-SimCLR
python main.py --model SimCLR --wandb True --temperature 0.1

# CCL-SimCLR
python main.py --wandb True --model SimCLR --train_bybatch True --temperature 0.3

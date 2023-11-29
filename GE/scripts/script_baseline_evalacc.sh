# your own model directory
MODEL_DIR=MODEL_DIR

# CLIP-MoCo 
python eval_acc.py --model MoCo --model_dir $MODEL_DIR

# CLIP-SimCLR
python eval_acc.py --model SimCLR --model_dir $MODEL_DIR

# CCL-SimCLR
python eval_acc.py --model SimCLR --model_dir $MODEL_DIR

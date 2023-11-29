# your own model path
MODEL_DIR=MODEL_DIR

# CLIP-MoCo 
python eval_acc.py --model MoCo --temperature 0.1 --model_dir $MODEL_DIR

# CLIP-SimCLR
python eval_acc.py --model SimCLR --temperature 0.1 --model_dir $MODEL_DIR

# CCL-SimCLR
python eval_acc.py --model SimCLR --model_dir $MODEL_DIR

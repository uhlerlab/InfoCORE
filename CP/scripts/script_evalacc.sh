# your own model directory
MODEL_DIR=MODEL_DIR

# InfoCORE-MoCo 
python eval_acc.py --model MoCo --infocore True --mi_weight_coef 5 --rewgrad_coef 0 --model_dir $MODEL_DIR

# InfoCORE-SimCLR
python eval_acc.py --model SimCLR --infocore True --mi_weight_coef 5 --rewgrad_coef 0 --model_dir $MODEL_DIR

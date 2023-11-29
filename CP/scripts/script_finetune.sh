# for InfoCORE-MoCo
MODEL_DIR=MODEL_DIR

# BACE
python finetune.py --druglistname ogbg-molbace --model_dir $MODEL_DIR --lr 0.0002 --n_epoches 35 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# BBBP
python finetune.py --druglistname ogbg-molbbbp --model_dir $MODEL_DIR --lr 0.0003 --n_epoches 50 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# ClinTox
python finetune.py --druglistname ogbg-molclintox --model_dir $MODEL_DIR --lr 0.00001 --n_epoches 100 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# HIV
python finetune.py --druglistname ogbg-molhiv --model_dir $MODEL_DIR --lr 0.00008 --n_epoches 5 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# SIDER
python finetune.py --druglistname ogbg-molsider --model_dir $MODEL_DIR --lr 0.00001 --n_epoches 20 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# Tox21
python finetune.py --druglistname ogbg-moltox21 --model_dir $MODEL_DIR --lr 0.0002 --n_epoches 20 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# ToxCast
python finetune.py --druglistname ogbg-moltoxcast --model_dir $MODEL_DIR --lr 0.0002 --n_epoches 20 --model MoCo --criterion auc --infocore True --mi_weight_coef 5 --rewgrad_coef 0

# PRISM
python finetune_PRISM.py --model_dir $MODEL_DIR --ft_lr 0.00005 --ft_n_epoches 50 --model MoCo --infocore True --mi_weight_coef 5 --rewgrad_coef 0

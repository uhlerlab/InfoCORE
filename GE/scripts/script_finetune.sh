# for InfoCORE-MoCo
MODEL_DIR=MODEL_DIR

# BACE
python finetune.py --druglistname ogbg-molbace --model_dir $MODEL_DIR --lr 0.00015 --n_epoches 35 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# BBBP
python finetune.py --druglistname ogbg-molbbbp --model_dir $MODEL_DIR --lr 0.00005 --n_epoches 50 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# ClinTox
python finetune.py --druglistname ogbg-molclintox --model_dir $MODEL_DIR --lr 0.00003 --n_epoches 80 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# HIV
python finetune.py --druglistname ogbg-molhiv --model_dir $MODEL_DIR --lr 0.00002 --n_epoches 3 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# SIDER
python finetune.py --druglistname ogbg-molsider --model_dir $MODEL_DIR --lr 0.00001 --n_epoches 20 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# Tox21
python finetune.py --druglistname ogbg-moltox21 --model_dir $MODEL_DIR --lr 0.0001 --n_epoches 60 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# ToxCast
python finetune.py --druglistname ogbg-moltoxcast --model_dir $MODEL_DIR --lr 0.0002 --n_epoches 50 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

# PRISM
python finetune_PRISM.py --model_dir $MODEL_DIR --ft_lr 0.00005 --ft_n_epoches 50 --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --dataparallel True

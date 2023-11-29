# InfoCORE-MoCo
python main.py --model MoCo --infocore True --mi_weight_coef 0.5 --rewgrad_coef 0 --clf_steps 2 --moco_m_clf 0.99 --dataparallel True --wandb True

# InfoCORE-SimCLR
python main.py --wandb True --model SimCLR --infocore True --mi_weight_coef 0.25  --rewgrad_coef 0 --clf_steps 2

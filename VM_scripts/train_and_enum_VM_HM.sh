#!/bin/bash

nohup python3 -u  training_model_VM_subplan_HM.py --score_func $1 --mn $2 --iter $6 --train_wk $7 --postfix $8 >./logs/training_model_VM_func_$1_mn_$2_iter_$6_training_data_$7_$8.log 2>&1 &


while (ps -ef | grep -v grep | grep $2 | grep $1 | grep $6 | grep $7 | grep $8 | grep training_model_VM_subplan_HM.py > /dev/null); do sleep 1; done

nohup python3 -u  LTR_enumerate_plans_VM.py --emd $3 --tq $4  --db $5 --mn MODEL_$2_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_$7_$8_$1_ITER_$6  >./logs/LTR_enum_$3_tq_$4_db_$5_MODEL_$2_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_$7_$8_score_$1_ITER_$6.log 2>&1 &


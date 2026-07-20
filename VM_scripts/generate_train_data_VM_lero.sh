#!/bin/bash

nohup python3 -u  generate_train_data_VM_lero.py --workload $1 --postfix $2 --nr_jobs $3 --score_func $4 >./logs/generate_train_data_VM_subplans_Lero_wk_$1_postfix_$2_score_func_$4_nrjobs_$3.log 2>&1 &


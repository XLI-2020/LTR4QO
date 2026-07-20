
import ltr_db_optimizer.model.featurizer_graph_lero as feature_extraction
import pickle
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--workload", type=str, default='None', help="training queries: job_part/tpch1000/stats1000")

parser.add_argument("--nr_jobs", type=int, default=1000)

parser.add_argument("--postfix", type=str, default="")

parser.add_argument("--score_func", type=str, default="linearR")


# workload = "job_part"
# workload = "tpch1000"
# workload = "stats1000"

args = parser.parse_args()
workload = args.workload
nr_jobs = args.nr_jobs
postfix = args.postfix
score_func = args.score_func
root_path = f"./LTR4QO/Data/subplans_{workload}"


training_data_path = "./LTR4QO/Data/"


vectors, plans, labels = feature_extraction.featurize_with_labels_Random(f"{root_path}", f"{root_path}",
                                                                    f"{root_path}/all_cost_labels.csv", workload=workload, nr_jobs=nr_jobs,
                                                                  score_function=score_func,  postfix=postfix)


print(f'{score_func}: number of query samples, plan samples, and label samples', len(vectors.keys()), len(plans.keys()), len(labels.keys()))


with open(f"{training_data_path}/training_data/lero_{score_func}_{workload}{postfix}_query_enc_norm.pickle", "wb") as f:
    pickle.dump(vectors, f)

with open(f"{training_data_path}/training_data/lero_{score_func}_{workload}{postfix}_plan_enc_norm.pickle", "wb") as f:
    pickle.dump(plans, f)

with open(f"{training_data_path}/training_data/lero_{score_func}_{workload}{postfix}_labels_norm.pickle", "wb") as f:
    pickle.dump(labels, f)



"""

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510251802.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510261001.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510261500.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510262126_all.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510271716_all.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510272113_COST.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510272121_COST_all.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202510272133_COST_all_Db4000.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202511242106_tpch20joins1000_subplans_shortqueryvec.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202601171018_job_test_subplans_addIndexKeyFeat.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202601180919_job_test_subplans_addSortFeat.log 2>&1 &

nohup python3 -u  generate_train_data_VM.py >./logs/generate_train_data_VM_202601180923_job_test_subplans_addIndexKeyFeat.log 2>&1 &


nohup python3 -u  generate_train_data_VM.py --workload $1 --nr_jobs $2 --score_func $3  --postfix $4    >./logs/generate_train_data_VM_subplans_wk_$1_nr_$2_score_func_$3_postfix_$4.log 2>&1 &

bash generate_train_data_VM.sh stats1000 10D95 1000 linearR  &

bash generate_train_data_VM.sh stats1000 5Q10D18 1000 linearR  &

bash generate_train_data_VM.sh stats1000 4Q10D18 1000 linearR  &

bash generate_train_data_VM.sh stats1000 10D18Log 1000 linearR  &


bash generate_train_data_VM.sh stats1000 Sq10D95P 1000 linearR  &

bash generate_train_data_VM.sh tpch1000 Sq10D95P 1000 linearR  &

bash generate_train_data_VM.sh job_part Sq10D95P 1000 linearR  &

bash generate_train_data_VM.sh stats1000 Sq10D95K 1000 linearR  &

bash generate_train_data_VM.sh tpch1000 Sq10D95P 1000 linearR  &


bash generate_train_data_VM.sh job_part Sq10DNoNormAtAll 1000 linearR  &


bash generate_train_data_VM.sh tpch1000 Sq10DNoNormAtAll 1000 linearR  &

bash generate_train_data_VM.sh stats1000reshuff 6Q10DTree17591100%1 1000 linearR  &


bash generate_train_data_VM.sh stats1000reshuff 6Q10D50pTree328064%1 1000 linearR  &

bash generate_train_data_VM.sh stats1000reshuff 6Q10D50pTree52426%1 1000 linearR  &

bash generate_train_data_VM.sh imdb39reshuff Q6D10maxTree1547830000%1 1000 linearR  &

bash generate_train_data_VM.sh imdb39reshuff Q6D10P50Tree129798%1 1000 linearR  &

bash generate_train_data_VM_lero.sh stats1000reshuff 6Q10D98pTree691628%1  1000 linearR  &

bash generate_train_data_VM_lero.sh imdb39reshuff Q6D10P98Tree36244300%1 1000 linearR  &


bash generate_train_data_VM_lero.sh tpch1000 Q6D10P98Tree36244300%1  1000 linearR  &

bash generate_train_data_VM_lero.sh stats1000reshuff 6Q10D98pTree17591100%1  1000 linearR  &


bash generate_train_data_VM_lero.sh tpch1000 Q6D10MaxTree28796000000000%1  1000 linearR  &

bash generate_train_data_VM_lero.sh imdb39reshuff  Q6D18MaxTree1547830000%1  1000 linearR  &

bash generate_train_data_VM_lero.sh imdb39reshuff  Q6D31MaxTree1547830000%1  1000 linearR  &







"""


"""
stats-r
min_value:  1.0
percent_5:  1166.86
percent_15:  9044.63
percent_25:  15828.9
median:  52426.7
percen_75:  170017.0
percen_90:  303187.0
percen_95:  328064.0
percen_98:  691628.0
max_value:  17591100.0

job77
min_value:  1.0
percent_5:  1.0
percent_15:  4.0
percent_25:  1297.98
median:  135086.0
percen_75:  2528310.0
percen_90:  4598250.0
percen_95:  14835700.0
percen_98:  36244300.0
max_value:  8607380000.0


imdb96reshuffle
min_value:  1.0
percent_5:  1.0
percent_15:  1.0
percent_25:  39.1636
median:  129798.0
percen_75:  1735760.0
percen_90:  4523930.0
percen_95:  14835700.0
percen_98:  36244300.0
max_value:  8607380000.0


job37:
min_value:  1.0
percent_5:  1.0
percent_15:  11.1934
percent_25:  3374.66
median:  189127.0
percen_75:  2609130.0
percen_90:  7328150.0
percen_95:  22255900.0
percen_98:  36244300.0
max_value:  8607380000.0


imdb39reshuffle
min_value:  1.0
percent_5:  1.0
percent_15:  1.0
percent_25:  5.61122
median:  129798.0
percen_75:  1698960.0
percen_90:  4523930.0
percen_95:  14835700.0
percen_98:  36244300.0
max_value:  1547830000.0

tpch1000:

min_value:  1.0
percent_5:  5.0
percent_15:  380.0
percent_25:  10000.0
median:  203845.0
percen_75:  1500740.0
percen_90:  6001220.0
percen_95:  36507500.0
percen_98:  480097000.0
max_value:  28796000000000.0
"""















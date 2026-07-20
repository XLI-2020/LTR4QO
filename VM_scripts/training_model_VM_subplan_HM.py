from datetime import datetime

from ltr_db_optimizer.model.model_interface_HM import create_model
import pickle
import os

from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.listnet import ListNet
from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.listmle import ListMLE

from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.lambdaloss import LambdaLoss
from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.lambdarank import LambdaRank
from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.rank_cosine import RankCosine

from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.approxNDCG import ApproxNDCG
from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.softrank import SoftRank
from argparse import ArgumentParser
from ltr_db_optimizer.ext.ptranking.ltr_adhoc.pairwise.ranknet import RankNet


from ltr_db_optimizer.extra_utils import get_the_split_of_training_data
import random


parser = ArgumentParser()

parser.add_argument("--mn", type=str, default='HM', help="model's name to indicate model's architecture")

parser.add_argument("--score_func", type=str, default='linearR')
parser.add_argument("--score_n", type=int, default=50)
parser.add_argument("--score_bord", type=int, default=97)

parser.add_argument("--loss_func", type=str, default="lambdaloss")
parser.add_argument("--loss_type", type=str, default="NDCG_Loss2++", help="NDCG_Loss2++")
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--ll_k", type=int, default=5)
parser.add_argument("--mu", type=float, default=1.0)
parser.add_argument("--presort", type=bool, default=False)

parser.add_argument("--iter", type=str, default="None")

parser.add_argument("--train_wk", type=str, default="", help="training workload/data")
parser.add_argument("--postfix", type=str, default="")


args = parser.parse_args()
model_name = args.mn

if args.loss_func == "lambdaloss":
    loss_func = LambdaLoss
elif args.loss_func == "listmle":
    loss_func = ListMLE
elif args.loss_func == "rankcosine":
    loss_func = RankCosine
elif args.loss_func == "listnet":
    loss_func = ListNet
elif args.loss_func == "lambdarank":
    loss_func = LambdaRank
elif args.loss_func == "softrank":
    loss_func = SoftRank
elif args.loss_func == "approxndcg":
    loss_func = ApproxNDCG
elif args.loss_func == "ranknet":
    loss_func = RankNet



score_func = args.score_func
score_n = args.score_n
score_bord = args.score_bord
iter = args.iter

training_workload = args.train_wk
postfix = args.postfix

directory = "./LTR4QO"

root_path = f"{directory}/Data"

print("Start Time =", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

Start_time = datetime.now()

optimizer = "adam"


with open(f"{root_path}/training_data/{score_func}_{training_workload}{postfix}_query_enc_norm.pickle", "rb") as f:
    vectors = pickle.load(f)
with open(f"{root_path}/training_data/{score_func}_{training_workload}{postfix}_plan_enc_norm.pickle", "rb") as f:
    plans = pickle.load(f)
with open(f"{root_path}/training_data/{score_func}_{training_workload}{postfix}_labels_norm.pickle", "rb") as f:
    labels = pickle.load(f)


assert len(plans.keys()) == len(vectors.keys()) == len(labels.keys()), "the number of plan encoding, query encoding, and labels does not match!!"

# split_valid, split_test = get_the_split_of_training_data(plan_keys=list(plans.keys()), valid_ratio=0.2, test_ratio=0.0, workload=training_workload)

total_subqueries = list(set(list(map(lambda x:"_".join(x.split("_")[:-1]), list(plans.keys())))))

split_valid = random.sample(total_subqueries, int(0.2*len(total_subqueries)))


split_test = []

print('split_valid detail: ',  len(split_valid), split_valid[:10])

print('split_test detail: ',  len(split_test), split_test[:10])


sorted_plans = {}
sorted_vecs = {}

for plan_key in plans.keys():
    # job_nr = plan_key.split("_")[0]
    job_nr = "_".join(plan_key.split("_")[:-1])

    if not job_nr in sorted_plans.keys():
        sorted_plans[job_nr] = {}
    sorted_plans[job_nr][plan_key] = plans[plan_key]
    if not job_nr in sorted_vecs.keys():
        sorted_vecs[job_nr] = {}
    sorted_vecs[job_nr][plan_key] = vectors[plan_key]

print('total number of jobs: ', len(list(sorted_plans.keys())))


X_test_tree = {}
X_test_vecs = {}
y_test = {}

X_train_tree = {}
X_train_vecs = {}
y_train = {}

X_valid_tree = {}
X_valid_vecs = {}
y_valid = {}
control_dict = {}

for key in sorted_plans.keys():
    # if key.split("_")[0] in split_test:
    if key in split_test:
        X_test_tree[key] = {}
        X_test_vecs[key] = {}
        for subkey in sorted_plans[key].keys():
            y_test[subkey] = labels[subkey]
            X_test_tree[key][subkey] = sorted_plans[key][subkey]
            X_test_vecs[key][subkey] = sorted_vecs[key][subkey]
    elif key in split_valid:
        X_valid_tree[key] = {}
        X_valid_vecs[key] = {}
        for subkey in sorted_plans[key].keys():
            y_valid[subkey] = labels[subkey]
            X_valid_tree[key][subkey] = sorted_plans[key][subkey]
            X_valid_vecs[key][subkey] = sorted_vecs[key][subkey]
    else:
        X_train_tree[key] = {}
        X_train_vecs[key] = {}
        for subkey in sorted_plans[key].keys():
            y_train[subkey] = labels[subkey]
            X_train_tree[key][subkey] = sorted_plans[key][subkey]
            X_train_vecs[key][subkey] = sorted_vecs[key][subkey]


print('the number of X_train vecs:', len(X_train_vecs.keys()))
print('the number of X_valid vecs:', len(X_valid_vecs.keys()))
print('the number of X_test vecs:', len(X_test_vecs.keys()))



if args.score_func == "special":
    # score_info = "_".join([args.score_func, str(args.score_n)])
    score_info = args.score_func
else:
    score_info = args.score_func


if args.loss_func == "lambdaloss":
    model_info = '_'.join(['MODEL', model_name, args.loss_func, 'sigma', str(args.sigma), 'k', str(args.ll_k), 'mu', str(args.mu), args.loss_type, 'presort', str(args.presort)])
else:
    model_info = '_'.join(['MODEL', model_name, args.loss_func])


print("Begin training with ", score_info + '###' + model_info)

model_save_name =  model_info + "_" + "TWL" + "_" + training_workload + "_" + postfix + "_" + score_info + "_" + "ITER" + "_" + iter

folder = f"./ltr_db_optimizer/model/saved_models"

os.system(f"mkdir -p {folder}")

print('model saving name: ', f"{folder}/{model_save_name}")

if os.path.exists(f"{folder}/{model_save_name}"):
    print('Model file already exists and needs to be deleted first!!!')
    os.system(f'rm -rf {folder}/{model_save_name}')
    print('after deleting, check if the model file still exists: ', os.path.exists(f"{folder}/{model_save_name}"))
else:
    print("Model file not exists!")


if args.loss_func == "lambdaloss":
    model = create_model(loss_func, folder=folder, name=model_save_name, workload=training_workload, model_para_dict={"sigma": args.sigma, 'k': args.ll_k, 'mu': args.mu, 'loss_type': args.loss_type})
elif args.loss_func in ["listmle", "rankcosine", "lambdarank", "softrank", "approxndcg"]:
    model = create_model(loss_func, folder=folder, name=model_save_name, workload=training_workload, model_para_dict={"sigma": args.sigma, 'k': args.ll_k, 'mu': args.mu, 'loss_type': args.loss_type})
elif args.loss_func == "ranknet": ### for lero model
    model = create_model(loss_func, folder=folder, name=model_save_name, workload=training_workload, model_para_dict={"sigma": 1.0})
else:
    model = create_model(loss_func, folder=folder, name=model_save_name, workload=training_workload)


model.fit(X_train_vecs, X_train_tree, y_train, X_valid_vecs, X_valid_tree, y_valid, use_presort=args.presort, optimizer=optimizer)
print('training is done!!!')
print("End Time =", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
End_time = datetime.now()
elapsed_time = round((End_time - Start_time).total_seconds()/60, 2)
print('total elapsed time for training: (mins)', elapsed_time)


# print('Model Test!!')
# avg_test_loss, avg_test_ndcg, avg_test_ndcg_new = model.test(X_test_vecs, X_test_tree, y_test)
# print('test ndcg: ', avg_test_ndcg)
# print('test ndcg sk: ', avg_test_ndcg_new)
# print('test loss: ', avg_test_loss)
# print("Finish testing", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))





"""

bash train_and_enum_VM.sh linearR HM HM imdb-o imdb tpch1k%%278thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 282thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 10D80 &


bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 283thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 10D90 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM tpch-d tpch 284thSamTrain50%FixValid20%SamSize20NoScheduler tpch1000 10D53 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 285thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 10D95 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 287thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 4Q10D18 &


bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 288thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 10D18Log &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 289thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 290thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 291thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 Sq10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 292thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 Sq10D95P &


bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 291thSamTrain50%FixValid20%SamSize20NoScheduler job_part 10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 291thSamTrain50%FixValid20%SamSize20NoScheduler job_part 10D95P &



bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 294thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 Sq10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 294thSamTrain50%FixValid20%SamSize20NoScheduler tpch1000 Sq10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 294thSamTrain50%FixValid20%SamSize20NoScheduler job_part Sq10D95P &



bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 292thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 Sq10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 295thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 Sq10D95P &

bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-o stats 296thSamTrain50%FixValid20%SamSize20NoScheduler job_part Sq10D95P &




bash train_and_enum_VM.sh linearR LTRankNet0 HM tpch-s tpch 298thSamTrain50%FixValid20%SamSize20NoScheduler tpch1000 Sq10D95P &

bash train_and_enum_VM.sh linearR LTRankNet2 HM stats-o stats 299thSamTrain50%FixValid20%SamSize20NoScheduler stats1000 Sq10DNoNormOnCard &


bash train_and_enum_VM.sh linearR LTRankNet2 HM imdb-o imdb 303thSamTrain50%FixValid20%SamSize20NoScheduler job_part Sq10DNoNormAtAll &


bash train_and_enum_VM.sh linearR LTRankNet0 HM stats-r stats 304thSamTrain50%FixValid20%SamSize20NoScheduler stats1000reshuff 6Q10Dmax &



bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 307thSamTrain50%FixValid20%SamSize20NoScheduler job77 6Q10DmaxTree8607380000%1  &


bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 308thSamTrain50%FixValid20%SamSize20NoScheduler job77 6Q10D98pTree36244300%1  &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 309thSamTrain50%FixValid20%SamSize20NoScheduler job77 6Q10D95pTree14835700%1  &


bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-o imdb 310thSamTrain50%FixValid20%SamSize20NoScheduler job77 6Q10D50pTree135086%1  &



bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 311thSamTrain50%FixValid20%SamSize20NoScheduler imdb96reshuff 6Q10D98PTree36244300%1  &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 312thSamTrain50%FixValid20%SamSize20NoScheduler imdb96reshuff 6Q10DmaxTree8607380000%1  &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 313thSamTrain50%FixValid20%SamSize20NoScheduler imdb96reshuff 6Q10D95PTree14835700%1  &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 314thSamTrain50%FixValid20%SamSize20NoScheduler imdb96reshuff  6Q10D50PTree135086%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM job-t3 imdb 315thSamTrain50%FixValid20%SamSize20NoScheduler job37  Q6D10P50Tree189127%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM job-t3 imdb 316thSamTrain50%FixValid20%SamSize20NoScheduler job37  Q6D10P95Tree22255900%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM job-t3 imdb 317thSamTrain50%FixValid20%SamSize20NoScheduler job37  Q6D10P98Tree36244300%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM job-t3 imdb 318thSamTrain50%FixValid20%SamSize20NoScheduler job37  Q6D10maxTree8607380000%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 319thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10P50Tree129798%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 320thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10P95Tree14835700%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 321thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10P98Tree36244300%1 &

bash train_and_enum_VM.sh linearR LTRankNet0 HM imdb-r imdb 322thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10maxTree1547830000%1 &



bash train_and_enum_VM_HM.sh linear HM HM stats-r stats 336thSamTrain50%FixValid20%SamSize20NoScheduler stats1000reshuff 6Q10D98pTree691628%1 &


bash train_and_enum_VM_HM.sh linear HM HM imdb-r imdb 344thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10PrevTree1547830000%1 &


bash train_and_enum_VM_HM.sh linear HM HM imdb-r imdb 345thSamTrain50%FixValid20%SamSize20NoScheduler tpch1000 Q6D10MaxTree480174000%1 &

bash train_and_enum_VM_HM.sh linear HM HM imdb-r imdb 344thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10PrevTree53435600%1 &


bash train_and_enum_VM_HM.sh linear HM HM stats-r stats 346thSamTrain50%FixValid20%SamSize20NoScheduler stats1000reshuff 6Q10D98pTree17591100%1 &


bash train_and_enum_VM_HM.sh linear HM HM imdb-r imdb 349thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10PrevTree1547830000%1 &

bash train_and_enum_VM_HM.sh linear HM HM imdb-r imdb 350thSamTrain50%FixValid20%SamSize20NoScheduler tpch1000 Q6D10MaxTree480174000%1 &

bash train_and_enum_VM_HM.sh linear HM HM job-c imdb 357thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10PrevTree1547830000%1 &


bash train_and_enum_VM_HM.sh linear HM HM imdb-r imdb 358thSamTrain50%FixValid20%SamSize20NoScheduler imdb39reshuff  Q6D10PrevTree1547830000%1 &









"""

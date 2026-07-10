# from __future__ import print_function  # `__future__` has to be before other imports
from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_5 import TPCHTableInformation5

from ltr_db_optimizer.enumeration_algorithm.table_info_imdb import IMDBTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_stats import STATSTableInformation

import pyodbc
from pyodbc import ProgrammingError, OperationalError
import os
import pandas as pd
from datetime import datetime
import time
from ltr_db_optimizer.parser import SQLParser
from argparse import ArgumentParser
import copy
import pickle
from ltr_db_optimizer.parser.SQLParser import create_query_element, clear_query_element
import random

random.seed(13)

def add_force_scan_to_sql(sql):
    # input_sql = sql.upper()
    print('initial sql: ', sql)
    input_sql = copy.copy(sql)
    before_from, after_from = input_sql.split('FROM')
    if "WHERE" in after_from:
        before_where, after_where = after_from.split('WHERE')
        tables = before_where.split(',')

        tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))

        # print(tables_added_hint)

        sql_with_hint = before_from + " FROM" + tables_added_hint + " WHERE" + after_where
        # print('input sql', input_sql)
    else:
        if "GROUP BY" in after_from:
            before_group_by, after_group_by = after_from.split('GROUP BY')
            tables = before_group_by.split(',')
            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))
            sql_with_hint = before_from + " FROM" + tables_added_hint + " GROUP BY" + after_group_by
        elif "ORDER BY" in after_from:
            before_order_by, after_order_by = after_from.split('ORDER BY')
            tables = before_order_by.split(',')
            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))
            sql_with_hint = before_from + " FROM" + tables_added_hint + " ORDER BY" + after_order_by
        else:
            tables = after_from.split(',')
            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))
            sql_with_hint = before_from + " FROM" + tables_added_hint
        # print('input sql: ', input_sql)
        # print('sql_with_hint: ', sql_with_hint)

    sql_with_hint = sql_with_hint + " OPTION (MAXDOP 1) "
    print("####")
    print('sql_with_hint: ', sql_with_hint)

    return sql_with_hint


def clean_sql_from_datafarm(ori_sql):
    before_FROM, after_FROM = ori_sql.split('FROM')

    before_WHERE, after_WHERE = after_FROM.split('WHERE')

    tables_before_WHERE = before_WHERE.split(',')

    tables_before_WHERE = list(set(list(map(lambda x: x.strip(), tables_before_WHERE))))

    new_before_WHERE = ', '.join(tables_before_WHERE)

    new_sql = before_FROM + " FROM " + new_before_WHERE + " WHERE " + after_WHERE

    return new_sql

def set_DBMS(db):
    SERVER = 'localhost,1433'
    DATABASE = db
    USERNAME = 'sa'
    PASSWORD = 'LX##1992'

    conn = pyodbc.connect(driver='{ODBC Driver 18 for SQL Server}',
                          server=SERVER,
                          database=DATABASE,
                          UID=USERNAME,
                          PWD=PASSWORD, TrustServerCertificate='Yes')


    conn.timeout = 21600  # 180
    cursor = conn.cursor()
    return cursor


def enumerate_by_SQL_Server(job, sql_full, folder_name, cursor):

    cursor.execute("SET SHOWPLAN_XML ON")

    start_time = datetime.now()

    sql_full = add_force_scan_to_sql(sql_full)

    rows = cursor.execute(sql_full).fetchall()

    enum_end_time = datetime.now()
    enum_delta_time = (enum_end_time - start_time).total_seconds()
    # enum_plan_time.append(enum_delta_time)

    print('planning time:', enum_delta_time)

    # print('rows[0]', rows[0])
    # print('rows[0][0]', rows[0][0])
    path_out = f"/home/xliq/Documents/LTR_DP/results/enumerated_plans_DB/{folder_name}/iter{iter}/{job}/"

    os.system(f"mkdir -p {path_out}")


    sql_full_without_addtional_string = sql_full.replace("OPTION (MAXDOP 1)", "")

    plan_sql = sql_full_without_addtional_string + " OPTION (RECOMPILE,  MAXDOP 1, USE PLAN N'" + rows[0][0] + "')"

    # plan_sql = sql_full_without_addtional_string + " OPTION (RECOMPILE, MAXDOP 1, USE PLAN N'" + rows[0][0] + "')"

    with open(path_out + str(0) + ".txt", "w") as f:
        f.write(plan_sql)

    cursor.execute("SET SHOWPLAN_XML OFF")

def preprocess_job_queries(sql_full):
    print('original sql: ', sql_full)

    if sql_full.strip().endswith(";"):
        sql_full = sql_full.strip()
        sql_full = sql_full[:-1]

    print('sql to enum: ', sql_full)

    with open(f"{query_path}/{job}.sql", "w+") as f:
        f.write(sql_full)

if __name__ == "__main__":
    parser = ArgumentParser()

    # random.seed(248)
    # np.random.seed(248)
    # torch.manual_seed(248)


    parser.add_argument("--emd", type=str, default='HM')

    parser.add_argument("--db", type=str, default="tpch")

    parser.add_argument('--tq', type=str, default="tpch-o")

    parser.add_argument('--topk', type=int, default=20)

    parser.add_argument('--do_enum', type=bool, default=True)

    parser.add_argument('--do_runtime', type=bool, default=False)

    parser.add_argument('--iter', type=str, default='None', help="stats_1000")


    parser.add_argument("--mn", type=str, default='special_50_97_MODEL_LTRankModel1_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False')

    print('start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    Start_time = datetime.now()

    args = parser.parse_args()

    targeted_enum_method = args.emd

    targeted_database = args.db

    targeted_test_query = args.tq

    targeted_model_name = args.mn

    iter = args.iter

    print('Argumments: ', args)

    cursor = set_DBMS(db=targeted_database)


    ### dataset info.
    if targeted_database == "tpch":
        TableInformation = TPCHTableInformation()
    elif targeted_database == "imdb":
        TableInformation = IMDBTableInformation()
    elif targeted_database == "tpch5":
        TableInformation = TPCHTableInformation5()
    elif targeted_database == "stats":
        TableInformation = STATSTableInformation()

    ### Method info.


    if targeted_enum_method == "XL":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_XL import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_XL import EnumerationAlgorithm")

    elif targeted_enum_method == "COST":
        from ltr_db_optimizer.enumeration_algorithm.archiv.enumeration_algorithm_COST import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_COST import EnumerationAlgorithm")

    elif targeted_enum_method == "HM":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm")

    elif targeted_enum_method == "RD":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_Random import EnumerationAlgorithm

        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_Random import EnumerationAlgorithm")



    ### trained ranking model to rank plans in enumeration process
    # model = f"/home/xliq/Documents/LTR_DP/ltr_db_optimizer/model/saved_models/{targeted_model_name}/best_avg_ndcg.pth"

    # model = f"/home/xliq/Documents/LTR_DP/ltr_db_optimizer/model/saved_models/{targeted_model_name}/min_avg_valid_loss.pth"

    model = None

    nr_of_workloads = args.iter
    ### training queries info.

    if targeted_test_query == "job-o":
        query_folder_name = "job"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)
    elif targeted_test_query == 'train':
        # input_job_folder = nr_of_workloads.split('_')[0]
        # query_path = f"/home/xliq/Documents/LTR_DP/Data/output_jobs_{input_job_folder}/"

        # query_path = f"/home/xliq/Documents/LTR_DP/Data/output_jobs_depth20joins12jobs2000/"

        query_path = f"/home/xliq/Documents/LTR_DP/Data/output_jobs/"

        train_query_files = os.listdir(query_path)
        testquery = []
        for train_query in train_query_files:
            if train_query.endswith(".txt"):
                testquery.append(train_query.split(".")[0])

        random.shuffle(testquery)
        # nr_testquery = int(nr_of_workloads.split('_')[1])
        nr_testquery = 2000
        testquery = testquery[:nr_testquery]
    elif targeted_test_query == 'stats':
        query_folder_name = "stats_train2" # stats1000reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".txt"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'imdb-t':
        query_folder_name = "imdb_train"  #imdb1000
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".txt"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'imdb-t2':
        query_folder_name = "imdb_train2"  # imdb39reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'imdb-t2':
        query_folder_name = "imdb_train2"  # imdb39reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'imdb-bs1':
        query_folder_name = "job_base_query_split_1"  # imdb39reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/test_flat/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'job-l':
        query_folder_name = "job-light"  # imdb39reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'imdb-los1':
        query_folder_name = "job_leave_one_out_split_1"  # imdb39reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/test_flat/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == 'jobc-train':
        query_folder_name = "job-c-train"  # imdb39reshuff
        query_path = f"/home/xliq/Documents/LTR_DP/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])


    k = args.topk

    root_save_path = f"/home/xliq/Documents/LTR_DP/Data/subplans_{nr_of_workloads}"
    if os.path.exists(root_save_path):
        os.system(f"rm -rf {root_save_path}")
        print(f'{root_save_path} already exists and thus removed first!')
    if not os.path.exists(root_save_path):
        os.system(f"mkdir -p {root_save_path}")



    enum_plan_time = []

    all_cost_labels = []

    testquery = list(sorted(testquery))

    if args.do_enum:
        for job_index, job in enumerate(testquery):
            try:

                if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3"]:
                    continue

                if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                                   'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0', 'Job1325v3',
                                                                   'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0', 'Job1252v1',
                                                                   'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4', 'Job1028v2',
                                                                   'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:

                    continue

                # if targeted_test_query == "job-o" and job in ["7a", "7c"]:
                #     continue

                # if targeted_test_query == "job-o" and job[-1] == "a":
                #     continue

                # if targeted_test_query == "job-o" and (not job.startswith("29")):
                #     continue

                # if targeted_test_query == "imdb-t" and job not in ["tq119"]:
                #     continue

                print('enumerate job: ', job_index, job)

                ###load query
                # if targeted_test_query == 'train':
                #     with open(f"{query_path}/{job}.txt", "r") as f:
                #         sql_full = f.read()
                #     with open(f"{query_path}/{job}.pickle", "rb") as f:
                #         d = pickle.load(f)
                #     sql_full = sql_full.replace("tcph", "tpch")
                #
                #     sql_full = clean_sql_from_datafarm(sql_full)
                #     d = clear_query_element(d)
                #     sql_dict, alias_dict = create_query_element(d)

                if targeted_test_query in ['job-o', 'imdb-t2', 'job-l', 'imdb-bs1', 'imdb-los1', 'jobc-train']:
                    with open(f"{query_path}/{job}.sql", "r") as f:
                        sql_full = f.read()

                    sql_dict, alias_dict = SQLParser.from_sql(sql_full,temp_table_info=TableInformation)

                elif targeted_test_query in ['train', 'stats', 'imdb-t']:
                    with open(f"{query_path}/{job}.txt", "r") as f:
                        sql_full = f.read()
                    if targeted_test_query == 'train':
                        sql_full = sql_full.replace("tcph", "tpch")
                    sql_dict, alias_dict = SQLParser.from_sql(sql_full,temp_table_info=TableInformation)


                path_out = f"{root_save_path}/{job}/"

                ###enumeration
                enum = EnumerationAlgorithm(sql_dict,
                                            TableInformation,
                                            model,
                                            sql_full,
                                            k,
                                            job_name=job,
                                            alias_dict=alias_dict,
                                            nr_workload=nr_of_workloads)

                start_time = datetime.now()

                best_plans = enum.find_best_plan()

                enum_end_time = datetime.now()
                enum_delta_time = (enum_end_time - start_time).total_seconds()

                # enum_plan_time.append(enum_delta_time)/planning time


                print(f"{job}'s planning time:", enum_delta_time) #planning time: 0.11666666666666667

                print('the number of returned best plans', len(best_plans))


                # os.mkdir(path_out)
                os.system(f"mkdir -p {path_out}")
                # pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
                cost_labels = enum.cost_labels
                if len(cost_labels)>0:
                    cost_labels_df = pd.DataFrame(cost_labels)
                    cost_labels_df.columns = ['sp_name', 'cost']
                    cost_labels_df.to_csv(f"{path_out}/cost_labels_{job}.csv", index=True, header=True)


                all_cost_labels.extend(cost_labels)

            except ModuleNotFoundError:
                print("IGNORED: Subquery in " + path_out)
            # except Exception as e:
            #     print("Problem with " + path_out)
            #     print(e)
            #     continue
    all_cost_labels_df = pd.DataFrame(all_cost_labels)
    all_cost_labels_df.columns = ['sp_name', 'cost']
    all_cost_labels_df.to_csv(f"{root_save_path}/all_cost_labels.csv", index=True, header=True)



    print('end time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    End_time = datetime.now()

    Elapsed_time = round((End_time - Start_time).total_seconds()/60, 2)
    print('Total Elapsed Time: ', Elapsed_time)




"""

Don't forget to active python virtual environments!!!




bash LTR_enumerate_plans_VM.sh RD train tpch None PruneThre_1

bash LTR_enumerate_plans_VM.sh RD train tpch None PruneThre_1_wl5000

bash LTR_enumerate_plans_VM.sh RD job-o imdb None PruneThre_1_job

bash LTR_enumerate_plans_VM.sh RD job-o imdb None PruneThre_1_joball

bash LTR_enumerate_plans_VM_Random.sh RD job-o imdb None PruneThre1JobAll


bash LTR_enumerate_plans_VM_Random.sh RD job-o imdb None TopK20

bash LTR_enumerate_plans_VM_Random.sh RD job-o imdb None TopK20


bash LTR_enumerate_plans_VM_Random.sh RD job-o imdb None TopK20AlmostAll
bash LTR_enumerate_plans_VM_Random.sh RD train tpch None depth20joins12jobs2000_2000 


bash LTR_enumerate_plans_VM_Random.sh RD stats stats None stats1000reshuff


bash LTR_enumerate_plans_VM_Random.sh RD imdb-t imdb None imdb1000


bash LTR_enumerate_plans_VM_Random.sh RD job-o imdb None TestOutlier7a7c


bash LTR_enumerate_plans_VM_Random.sh RD job-o imdb None TestOutlier29


bash LTR_enumerate_plans_VM_Random.sh RD imdb-t2 imdb None imdb39reshuff

bash LTR_enumerate_plans_VM_Random.sh RD train tpch None tpch1000

bash LTR_enumerate_plans_VM_Random.sh RD imdb-t2 imdb None imdb39reshuff50

bash LTR_enumerate_plans_VM_Random.sh RD stats stats None stats1000reshuff50


bash LTR_enumerate_plans_VM_Random.sh RD imdb-t2 imdb None imdb39reshuff


bash LTR_enumerate_plans_VM_Random.sh RD job-l imdb None joblight

bash LTR_enumerate_plans_VM_Random.sh RD imdb-bs1 imdb None imdbbasesplit

bash LTR_enumerate_plans_VM_Random.sh RD imdb-los1 imdb None imdbleaveonesplit

bash LTR_enumerate_plans_VM_Random.sh RD jobc-train imdb None jobcomplex











"""


"""
the order to test trained model: tpch-o -> imdb-o -> tpch-d

xliq@gpu.itu.dk:/home/xliq/Documents/LTR_DP

xliq@gpu.itu.dk:/home/xliq/Documents/LTR_DP/ltr_db_optimizer/model/model_structures

 xliq@gpu.itu.dk:/home/xliq/Documents/LTR_DP/ltr_db_optimizer/allrank/models
 
 xliq@gpu.itu.dk:/home/xliq/Documents/LTR_DP/ltr_db_optimizer/model

"""





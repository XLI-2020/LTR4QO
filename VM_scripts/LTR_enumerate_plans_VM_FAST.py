# from __future__ import print_function  # `__future__` has to be before other imports

from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_5 import TPCHTableInformation5

from ltr_db_optimizer.enumeration_algorithm.table_info_imdb import IMDBTableInformation

from ltr_db_optimizer.enumeration_algorithm.table_info_stats import STATSTableInformation

from pyodbc import ProgrammingError, OperationalError
import os
import pandas as pd
from datetime import datetime
import time
from ltr_db_optimizer.parser import SQLParser
from argparse import ArgumentParser

from ltr_db_optimizer.extra_utils_FAST import set_DBMS, enumerate_by_SQL_Server_FAST, enumerate_by_SQL_Server_FAST_without_hint

from hint_sets import HintSet

import json




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

    parser.add_argument('--iter', type=str, default='None')

    parser.add_argument("--mn", type=str, default='special_50_97_MODEL_LTRankModel1_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False')


    print('start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    Start_time = datetime.now()

    args = parser.parse_args()

    targeted_enum_method = args.emd

    targeted_database = args.db

    targeted_test_query = args.tq

    targeted_model_name = args.mn

    iter = args.iter

    k = args.topk



    if args.emd not in ["DB", 'COST', 'FAST']:
        train_wk = targeted_model_name.split('TWL_')[1].split('_')[0]
        train_wk_info =  targeted_model_name.split('TWL_')[1].split('_')[1]

        tree_high, tree_low = train_wk_info.split('Tree')[1].split('%')
        tree_high = int(tree_high)
        tree_low = int(tree_low)

        print('Training workload: ', train_wk)
        print('Tree info: ', tree_high, tree_low)

    print('Argumments: ', args)



    ### dataset info.
    if targeted_database == "tpch":
        TableInformation = TPCHTableInformation()
    elif targeted_database == "imdb":
        TableInformation = IMDBTableInformation()
    elif targeted_database == "tpch5":
        TableInformation = TPCHTableInformation5()
    elif targeted_database == "stats":
        TableInformation = STATSTableInformation()


    ### enum method info.

    if targeted_enum_method == "Ada":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_XL import EnumerationAlgorithm")

    elif targeted_enum_method == "COST":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_COST import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_COST import EnumerationAlgorithm")
        k = 3
        model = None
        train_wk = None
        tree_high = None
        tree_low = None

    elif targeted_enum_method == "HM":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_HM import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm")

    elif targeted_enum_method == "lero":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_lero import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm_lero import EnumerationAlgorithm")


    ### trained ranking model to rank plans in enumeration process
    # model = f"./LTR4QO/ltr_db_optimizer/model/saved_models/{targeted_model_name}/best_avg_ndcg.pth"
    print('targeted_model_name:', targeted_model_name)
    if targeted_model_name != None and targeted_model_name != "None":
        model_archi_name = targeted_model_name.split("MODEL_")[1].split("_")[0]

        print('used Model Architecture: ', model_archi_name)
        if model_archi_name  == "HM":
            model = f"./LTR4QO/ltr_db_optimizer/model/saved_models/{targeted_model_name}/avg_k.pth"
        else:
            model = f"./LTR4QO/ltr_db_optimizer/model/saved_models/{targeted_model_name}/min_avg_valid_loss.pth"


    ### test queries info.
    if targeted_test_query == "tpch-o":
        query_folder_name = "tpch_queries"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        testquery = ['1', '3', '5', '6', '10', '12', '14'] #'7', '8', '9' has subquery


    elif targeted_test_query in ["tpch-d", "tpch-l"]:
        df = pd.read_csv('./LTR4QO/Data/testing_data/HM_TPCH_test_results.csv', header=0)  ### 136 TPCH queries
        df = df[df['Time'] != 0]  # filter those bad queries
        testquery = df['Job'].values.tolist()
        query_folder_name = "HM_TPCH_test_queries"  # tpch_query
        query_path = f'./LTR4QO/Data/testing_data/{query_folder_name}/'


    elif targeted_test_query == "tpch-s": #tpch short queries
        df = pd.read_csv('./LTR4QO/results/dataset_tpch_workload_tpch-s_enum_DB_iter_tpchdThre50sNoBitMapCurrentCompaLevelCardEstimateCL140_runtime.csv', header=0)  ### 136 TPCH queries
        df = df[(df['Time'] != 0)]  # filter those bad queries
        print('total number of tpch datafarm queries: ', len(df))
        df = df[(df['Time'] < 50000)]
        print('the number of tpch datafarm queries less than 50 seconds: ', len(df))
        testquery = df['Job'].values.tolist()
        query_folder_name = "HM_TPCH_test_queries"  # tpch_query
        query_path = f'./LTR4QO/Data/testing_data/{query_folder_name}/'


    elif targeted_test_query == "imdb-o":
        query_folder_name = "imdb_queries"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".txt"):
                testquery.append(imdb_query.split(".")[0])

    elif targeted_test_query == "job-o":
        query_folder_name = "job"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    elif targeted_test_query == "stats-o":
        query_folder_name = "stats_queries"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".txt") and stats_query.startswith("qt"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == "stats-r":
        query_folder_name = "stats_test2"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".txt") and (stats_query.startswith("qt") or stats_query.startswith("tq")):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == "imdb-r":
        query_folder_name = "imdb_test2"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    elif targeted_test_query == "job-t3":
        query_folder_name = "imdb_test3"
        query_path = f"./LTR4QO/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    elif targeted_test_query == "job-c":
        query_folder_name = "job-complex"
        query_path = f"./LTR4QO/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    elif targeted_test_query == 'jobc-test':
        query_folder_name = "job-c-test"  # imdb39reshuff
        query_path = f"./LTR4QO/Data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".sql"):
                testquery.append(stats_query.split(".")[0])


    enum_plan_time = []

    ### read hints from FAST results

    postfix =  "_jobc"#""#"_jobc"

    final_prediction = json.loads(open('./LTR4QO/FASTgres-PVLDBv16/results/initial_predictions{postfix}.json'.format(postfix=postfix)).read())

    if args.do_enum:
        all_inference_times = []

        for job_index, job in enumerate(testquery):

            if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3"]:
                continue

            if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                               'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0', 'Job1325v3',
                                                               'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0', 'Job1252v1',
                                                               'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4', 'Job1028v2',
                                                               'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:


                continue

            if targeted_test_query == "tpch-s" and job in ["Job1300v2"]:
                continue

            # if targeted_test_query == "job-o" and job not in ["7a", "7c"]:
            #     continue

            # if targeted_test_query == "job-o" and (not job.startswith("29")):
            #     continue

            # sort_merge_join_jobs = ["8a", "9a", "9c", "9d", "17c", "17d", '17e', '17f', '20b', '22d']
            # if targeted_test_query == "job-o" and job not in sort_merge_join_jobs:
            #     continue

            # prune_at_end_test_query = ["32a", "23a", "5a", "6a", "2a", "11a", "12a", "13a", "4a", "1a"]
            # if targeted_test_query == "imdb-o" and job not in prune_at_end_test_query:
            #     continue

            # if targeted_test_query == "stats-o" and job not in ["q141", "q63", "q58", "q46", "q140", "q65", "q142", "q64", "q139", "q66", "q49"]: # select *
            #     continue

            # if targeted_test_query == "stats-o" and job not in ["q58"]:  # select *
            #     continue

            # if targeted_test_query == "job-c" and job in ["16", "10", "20"]:
            #     continue

            print('enumerate job: ', job_index, job)

            ###load query
            if os.path.exists(f"{query_path}/{job}.txt"):
                with open(f"{query_path}/{job}.txt", "r") as f:
                    sql_full = f.read()
            else:
                with open(f"{query_path}/{job}.sql", "r") as f:
                    sql_full = f.read()

            sql_full = sql_full.replace("tcph", "tpch")
            # sql_full = sql_full.replace("tpch5", "tpch")


            if targeted_enum_method == "FAST":
                hint_num = final_prediction["sq"+ job+".sql"]  #for job-c

                # hint_num = final_prediction[job+".sql"]

                hash_join = HintSet(int(hint_num)).hash_join
                merge_join = HintSet(int(hint_num)).merge_join
                loop_join = HintSet(int(hint_num)).nested_join
                index_scan = HintSet(int(hint_num)).index_scan
                sep_scan = HintSet(int(hint_num)).seq_scan
                join_list = []
                if hash_join:
                    join_list.append(" HASH JOIN")
                if merge_join:
                    join_list.append(" MERGE JOIN")
                if loop_join:
                    join_list.append(" LOOP JOIN")

                print('hint index scan:', HintSet(int(hint_num)).index_scan)
                print('hint table scan:', HintSet(int(hint_num)).seq_scan)

                join_hint = ','.join(join_list)
                print('job hint', job, join_hint)

                if not index_scan:
                    cursor = set_DBMS(db="imdb2")
                else:
                    cursor = set_DBMS(db="imdb")
                print('targeted database after FAST hint mapping:', targeted_database)

                # if hint_num == 63:
                #     enum_time = enumerate_by_SQL_Server_FAST_without_hint(job, sql_full, query_folder_name, cursor, iter)
                # else:
                enum_time = enumerate_by_SQL_Server_FAST(job, sql_full, query_folder_name, cursor, iter, hint=join_hint)

                all_inference_times.append(['DB', job, enum_time, 'ms'])


                continue

            path_out = f"./LTR4QO/results/enumerated_plans_{targeted_enum_method}_{targeted_model_name}/{query_folder_name}/iter{iter}/{job}/"

            if targeted_test_query in ["job-o", "stats-o"] and os.path.exists(path_out):
                print(f'{job} in {targeted_test_query} alreay enumerated and thus ignored!')
                continue


            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=TableInformation)
            # print('sql_dict', sql_dict)
            # print('alias_dict', alias_dict)



            ###enumeration
            enum = EnumerationAlgorithm(sql_dict,
                                        TableInformation,
                                        model,
                                        sql_full,
                                        k,
                                        alias_dict=alias_dict, train_wk=train_wk, tree_high=tree_high, tree_low=tree_low)

            start_time = datetime.now()

            best_plans = enum.find_best_plan()

            enum_end_time = datetime.now()
            enum_delta_time = (enum_end_time - start_time).total_seconds()

            # enum_plan_time.append(enum_delta_time)/planning time


            print(f"{job}'s planning time:", enum_delta_time) #planning time: 0.11666666666666667

            print('the number of returned best plans', len(best_plans))

            # path_out = f"./LTR4QO/results/enumerated_plans_{targeted_enum_method}_{targeted_model_name}/{query_folder_name}/iter{iter}/{job}/"

            # os.mkdir(path_out)
            os.system(f"mkdir -p {path_out}")
            # pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

            for idx, plan in enumerate(best_plans):
                xml = enum.to_xml(plan)

                with open(path_out + str(idx) + ".txt", "w") as f:
                    f.write(xml)
                # with open(path_out + str(idx) + ".pickle", "wb") as f:
                #     pickle.dump(plan, f)

    if targeted_enum_method == "FAST":
        all_inference_time_df = pd.DataFrame(all_inference_times, columns=["enum", "job", "enum_time",'unit'])  # here DB's inference is just enumeration time
        all_df_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_enumtime.csv"

        inference_times_descr_df = all_inference_time_df["enum_time"].describe()
        # inference_times_descr_df.columns = [targeted_enum_method]
        inference_times_descr_df = inference_times_descr_df.round(1)
        inference_times_descr_df_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_enumtime_descr.csv"

        all_inference_time_df.to_csv(all_df_path, header=True)

        inference_times_descr_df.to_csv(inference_times_descr_df_path, index=True, header=True)


    if args.do_runtime:
        ### get runtime of plans
        print('Enumeration done! Continue to obtain runtime of enumerated plans !!!')
        print('Current time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        all_runtime_list = []
        for job_index, job in enumerate(testquery):


            if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3", "Job128v3"]:
                continue

            if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                               'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0',
                                                               'Job1325v3',
                                                               'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0',
                                                               'Job1252v1',
                                                               'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4',
                                                               'Job1028v2',
                                                               'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:
                continue

            # if targeted_test_query == "job-c" and job in ["16", "10", "20"]:
            #     continue

            # if targeted_test_query == "job-o" and job not in ["7a", "7c"]:
            #     continue

            # if targeted_test_query == "job-o" and (not job.startswith("29")):
            #     continue


            # if targeted_test_query == "stats-o" and job not in ["q141", "q63", "q58", "q46", "q140", "q65", "q142", "q64", "q139", "q66", "q49"]:  # select *, q66 newly added
            #     continue


            # if targeted_test_query == "stats-o" and job in ["qo57", "qo47", "qo132"]: # select count(*)
            #     continue

            # sort_merge_join_jobs = ["8a", "9a", "9c", "9d", "17c", "17d", '17e', '17f', '20b', '22d']
            # if targeted_test_query == "job-o" and job not in sort_merge_join_jobs:
            #     continue

            # prune_at_end_test_query = ["32a", "23a", "5a", "6a", "2a", "11a", "12a", "13a", "4a", "1a"]
            # if targeted_test_query == "imdb-o" and job not in prune_at_end_test_query:
            #     continue

            print('run job: ', job_index, job)
            print('Job Start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            methods_runtime_list = []

            if targeted_enum_method == "FAST":
                plan_path = f"./LTR4QO/results/enumerated_plans_{targeted_enum_method}/{query_folder_name}/iter{iter}/{job}/"
            else:
                plan_folder_postfix = "_".join([targeted_enum_method, targeted_model_name])
                plan_path = f"./LTR4QO/results/enumerated_plans_{plan_folder_postfix}/{query_folder_name}/iter{iter}/{job}/"


            with open(plan_path + str(0) + ".txt", "r") as f:
                plan_sql = f.read()
            try:
                cursor.execute("SET STATISTICS TIME ON")
                cursor.execute(plan_sql)
                while (cursor.nextset()):
                    mess = cursor.messages
                    # print('results:', mess)
                    if len(mess[0][1].split(",")) == 3:
                        cpu = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                        cpu_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                        time = int(mess[0][1].split(",")[2].split("=")[1][:-3])
                        time_unit = mess[0][1].split(",")[2].split("=")[1][-3:]
                    elif len(mess[0][1].split(",")) == 2:
                        cpu = int(mess[0][1].split(",")[0].split("=")[1][:-3])
                        cpu_unit = mess[0][1].split(",")[0].split("=")[1][-3:]
                        time = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                        time_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                    else:
                        print('mess', mess)
                        continue
                result = [job, cpu, time, cpu + time, cpu_unit, time_unit]

            except OperationalError as err:
                print("Timeout")
                print(job, targeted_enum_method, targeted_model_name, err)
                result = [job, -1, -1, -1, 'ms', 'ms']
                print("----")

            except ProgrammingError as err:
                print('WeridError')
                print(job, targeted_enum_method, targeted_model_name, err)
                if targeted_enum_method != "FAST":
                    result = [job, -2, -2, -2, 'ms', 'ms']
                    print("----")
                else:
                    try:
                        pure_sql = plan_sql.split("OPTION")[0] + " OPTION (MAXDOP 1, USE HINT ('QUERY_OPTIMIZER_COMPATIBILITY_LEVEL_140', 'FORCE_DEFAULT_CARDINALITY_ESTIMATION')) "
                        ### rerun the pure sql instead of plan
                        cursor.execute("SET STATISTICS TIME ON")
                        cursor.execute(pure_sql)
                        while (cursor.nextset()):
                            mess = cursor.messages
                            # print('results:', mess)
                            if len(mess[0][1].split(",")) == 3:
                                cpu = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                                cpu_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                                time = int(mess[0][1].split(",")[2].split("=")[1][:-3])
                                time_unit = mess[0][1].split(",")[2].split("=")[1][-3:]
                            elif len(mess[0][1].split(",")) == 2:
                                cpu = int(mess[0][1].split(",")[0].split("=")[1][:-3])
                                cpu_unit = mess[0][1].split(",")[0].split("=")[1][-3:]
                                time = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                                time_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                            else:
                                print('mess', mess)
                                continue
                        result = [job, cpu, time, cpu + time, cpu_unit, time_unit]
                    except OperationalError as err:
                        print("Timeout After Syntax Error!")
                        print(job, targeted_enum_method, targeted_model_name, err)
                        result = [job, -1, -1, -1, 'ms', 'ms']
                        print("----")

            except Exception as exce:
                print('Exception:', Exception)
                continue

            print('statistic:', result)
            print('Job End time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            methods_runtime_list.append(result)

            df = pd.DataFrame(methods_runtime_list, index=[targeted_enum_method])
            print('df', df)
            df.columns = "Job,CPU time,Time,Sum,CPU unit,Time unit".split(',')
            all_runtime_list.append(df)

            all_runtime_df = pd.concat(all_runtime_list, axis=0)
            if targeted_enum_method == "FAST":
                all_df_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_runtime.csv"
            else:
                all_df_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_runtime.csv"
                # all_df_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_runtime.csv"

            all_runtime_df.to_csv(all_df_path, index=True, header=True)

        ### describe the runtime distributions of test queries
        runtime_desc_list = []
        for method in [targeted_enum_method]:
            method_runtime_df = all_runtime_df[all_runtime_df.index == method]
            method_runtime_desc_df = method_runtime_df['Time'].describe()
            runtime_desc_list.append(method_runtime_desc_df)

        runtime_desc_df = pd.concat(runtime_desc_list, axis=1)
        runtime_desc_df.columns = [targeted_enum_method]
        runtime_desc_df = runtime_desc_df.round(1)

        if targeted_enum_method == "FAST":
            runtime_desc_file_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_runtime_descr.csv"
        else:
            runtime_desc_file_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_runtime_descr.csv"
            # runtime_desc_file_path = f"./LTR4QO/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_runtime_descr.csv"

        runtime_desc_df.to_csv(runtime_desc_file_path, index=True, header=True)


    print('end time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    End_time = datetime.now()

    Elapsed_time = round((End_time - Start_time).total_seconds()/60, 2)
    print('Total Elapsed Time: ', Elapsed_time)




"""

Don't forget to active python virtual environments!!!



bash LTR_enumerate_plans_VM.sh HM imdb-o imdb linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_Db1kFullPlanSize187thModelSampleTrain50%FixedValid20%SubplanFixedSampleSize20 None

bash LTR_enumerate_plans_VM.sh DB imdb-o imdb None Job26NoBitMapLegacyCardEstimate


bash LTR_enumerate_plans_VM.sh DB imdb-o imdb None IMDB26NoBitMapCurrentCompaLevelCardEstimate


bash LTR_enumerate_plans_VM.sh DB imdb-o imdb None IMDB26NoBitMapCurrentCompaLevelCardEstimateCL150

bash LTR_enumerate_plans_VM.sh DB imdb-o imdb None IMDB26NoBitMapCurrentCompaLevelCardEstimateCL140





bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_job77%%273thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler Countq

bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_job77%%273thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler Countbq


bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_Db1kFullPlanSize187thModelSampleTrain50%FixedValid20%SubplanFixedSampleSize20 T100

bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_Db1kFullPlanSize187thModelSampleTrain50%FixedValid20%SubplanFixedSampleSize20 2T100


bash LTR_enumerate_plans_VM.sh DB stats-r stats None statsTop100NoBitMapCurrentCompaLevelCardEstimateCL140


bash LTR_enumerate_plans_VM.sh HM imdb-o imdb linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_tpch1k%%278thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler FixK

bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_tpch1k%%278thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler FixK

bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_tpch1k%%279thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler FixK


bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_tpch1k%%278thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler Elbow1

bash LTR_enumerate_plans_VM.sh HM stats-o stats linearR_50_MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_tpch1k%%279thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler Elbow1


bash LTR_enumerate_plans_VM.sh HM imdb-o imdb linearR_50_MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_tower_data_iter_tpch1k%%278thD10TreeHigh53SampleTrain50%FixedValid20%SampleSize20NoScheduler Elb50

bash LTR_enumerate_plans_VM.sh HM imdb-o imdb MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_stats1000_10D90_linearR_ITER_283thSamTrain50%FixValid20%SamSize20NoScheduler None

bash LTR_enumerate_plans_VM.sh HM stats-o stats MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_stats1000_10D95P_linearR_ITER_292thSamTrain50%FixValid20%SamSize20NoScheduler/ None


bash LTR_enumerate_plans_VM.sh HM imdb-o imdb  MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_job77_6Q10D95pTree14835700%1_linearR_ITER_309thSamTrain50%FixValid20%SamSize20NoScheduler None



bash LTR_enumerate_plans_VM.sh DB tpch-s tpch None tpchdThre50sNoBitMapCurrentCompaLevelCardEstimateCL140


bash LTR_enumerate_plans_VM.sh HM job-o imdb   MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_job77_6Q10D98pTree36244300%1_linearR_ITER_308thSamTrain50%FixValid20%SamSize20NoScheduler None


bash LTR_enumerate_plans_VM.sh HM job-t3 imdb   MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_job77_6Q10D98pTree36244300%1_linearR_ITER_308thSamTrain50%FixValid20%SamSize20NoScheduler 29Only


bash LTR_enumerate_plans_VM.sh DB job-t3 imdb None jobt3NoBitMapCurrentCompaLevelCardEstimateCL140


bash LTR_enumerate_plans_VM.sh HM imdb-o imdb  MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_job37_Q6D10P95Tree22255900%1_linearR_ITER_316thSamTrain50%FixValid20%SamSize20NoScheduler None


bash LTR_enumerate_plans_VM.sh HM imdb-r imdb   MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_imdb39reshuff_Q6D10P50Tree129798%1_linearR_ITER_319thSamTrain50%FixValid20%SamSize20NoScheduler None


bash LTR_enumerate_plans_VM.sh DB imdb-r imdb None imdb39reshuffNoBitMapCurrentCompaLevelCardEstimateCL140

 bash LTR_enumerate_plans_VM.sh HM imdb-r imdb MODEL_lero_ranknet_TWL_imdb39reshuff_Q6D10P98Tree36244300%1_linearR_ITER_325thSamTrain50%FixValid20%SamSize20NoScheduler None 

bash LTR_enumerate_plans_VM.sh lero imdb-r imdb MODEL_lero_ranknet_TWL_imdb39reshuff_Q6D10P98Tree36244300%1_linearR_ITER_327thSamTrain50%FixValid20%SamSize20NoScheduler None 

bash LTR_enumerate_plans_VM.sh HM tpch-s tpch  MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_tpch1000_Q6D10PrevTree53435600%1_linearR_ITER_330thSamTrain50%FixValid20%SamSize20NoScheduler None


bash LTR_enumerate_plans_VM.sh COST tpch-s tpch None None

bash LTR_enumerate_plans_VM.sh COST imdb-r imdb None None

bash LTR_enumerate_plans_VM.sh COST stats-r stats None None


bash LTR_enumerate_plans_VM.sh HM imdb-r imdb MODEL_HM_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_imdb39reshuff_Q6D10PrevTree53435600%1_special_50_ITER_334thSamTrain50%FixValid20%SamSize20NoScheduler None


bash LTR_enumerate_plans_VM.sh COST tpch-s tpch None k3


bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None Test5

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb2 None Test6

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb2 None Test7


bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None Test8


bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None Test9

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None Test10

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None Test11Initial


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobComplex2

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None imdbR

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobComplex3

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None imdbR2

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobComplex4

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None imdbR3

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobComplex5AllHintsOn

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None imdbR4AllHintsOn

bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None imdbR4

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobComplex5


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobC140hintOnly

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobC130hintOnly

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobC110hintOnly


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobC140OnlyShortQueries

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobC140Without10and20


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobCOnlyDefaultCardHint

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None JobCFASTHintFirst


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None OnlyAtrainingData


bash LTR_enumerate_plans_VM_FAST.sh FAST jobc-test imdb None jobcTestAllTraining


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None jobcTestAllTraining


bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None imdbSubqueryAsTrainData


bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None imdbSubqueryAsTrainDataJOBQuery


bash LTR_enumerate_plans_VM_FAST.sh FAST imdb-r imdb None EnumTime

bash LTR_enumerate_plans_VM_FAST.sh FAST job-c imdb None EnumTime

































"""







import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class FeatureExtractor:
    # We used the following SQL parts:
    # Sort
    # Join: Merge Join, Nested Loop Join, Hash Join
    # Aggregate: Stream Aggregate, Hash Aggregate
    # Scan: Index Scan, Table Scan
    type_vector = ["sort", "stream_aggregate", "hash_aggregate", "merge_join", "nested_loop_join", "hash_join", "index_scan", "table_scan"]
    
    def __init__(self, with_cost = True, small_version=False):
        self.vector_length = len(self.type_vector)
        self.with_cost = with_cost
        if with_cost:
            self.vector_length += 2 # for estimated subtree cost and estimated rows
        self.small_version = small_version
        
    def featurize_plan(self, plan):
        operation = plan["operator"] 
        none_child = np.zeros(self.vector_length)
        
        if operation in self.type_vector:
            this_vector = np.zeros(self.vector_length)
            this_vector[self.type_vector.index(operation)] = 1
        else:
            # If it is not in type vector, there should also be only one child
            return self.featurize_plan(plan["children"][0])
        if self.with_cost:
            # insert cost here
            this_vector[-2] = plan["EstimateRows"]
            this_vector[-1] = plan["EstimatedTotalSubtreeCost"]
        # test length of children (2,1,0)
        if len(plan["children"]) == 2:
            left_child = self.featurize_plan(plan["children"][0])
            right_child = self.featurize_plan(plan["children"][1])
        elif len(plan["children"]) == 1:
            left_child = self.featurize_plan(plan["children"][0])
            right_child = (none_child, None, None)
        else:
            return (this_vector, None, None)
        return (this_vector, left_child, right_child)
    
    def match_cost_plan(self, execution_plan, cost_plan):
        cost_parts = cost_plan.split("<")
        parts_cost = []
        
        for part_num, part in enumerate(cost_parts):
            if part.startswith("RelOp"):
                sub_parts = part.split('"')
                sub_parts_cost = []
                for idx, sub in enumerate(sub_parts):
                    if "PhysicalOp" in sub:
                        sub_parts_cost.append(("PhysicalOp", sub_parts[idx+1]))
                    if "EstimateRows" in sub:
                        sub_parts_cost.append(("EstimateRows",float(sub_parts[idx+1])))
                    if "EstimatedTotalSubtreeCost" in sub:
                        sub_parts_cost.append(("EstimatedTotalSubtreeCost",float(sub_parts[idx+1])))
                parts_cost.append(sub_parts_cost)

        extended_plan, _ = self.append_features(execution_plan, parts_cost)
        return extended_plan
    
    def append_features(self, execution_plan, label_parts):
        # Ugly, I will change that later
        if self.small_version:
            full_execution_plan = {"operator": execution_plan[0], "children": []}
            curr_part = label_parts[0]
            for c in curr_part:
                if c[0] == "PhysicalOp":
                    continue
                full_execution_plan[c[0]] = c[1]
                if execution_plan[0] not in ["index_scan", "table_scan"]:
                    for child in execution_plan[1]:
                        temp_sub_plan, _ = self.append_features(child, label_parts) 
                        full_execution_plan["children"].append(temp_sub_plan)
            return full_execution_plan, label_parts
        # Ugly edge-case where top and a sort are merged to "top sort"
        if not(execution_plan["operator"] == "top" and execution_plan["children"][0]["operator"] == "sort"):
            curr_part = label_parts[0]
            not_compute_scalar = True
            for c in curr_part:
                if execution_plan["operator"] == "compute_scalar" and c[0] == "PhysicalOp" and not c[1] == "ComputeScalar":
                    not_compute_scalar = False
                    break
                elif c[0] == "PhysicalOp":
                    continue
                execution_plan[c[0]] = c[1]
            if not_compute_scalar:
                label_parts.pop(0)
        for idx, child in enumerate(execution_plan["children"]):
            execution_plan["children"][idx], label_parts = self.append_features(child, label_parts)
        return execution_plan, label_parts

    
def get_features_with_cost_from_folder(plans_folder, cost_folder, return_featurized=True):
    feature_ext = FeatureExtractor()
    
    featurized_plans = {}
    
    for file in os.listdir(cost_folder):
        if file.endswith(".txt"):
            file_name = file.split(".")[0]
            job_nr = file.split("_")[0]
            version_nr = file_name.split("_")[1]
            
            cost_plan = ""
            with open(cost_folder+"/"+file, "r") as f:
                for line in f:
                    cost_plan += line
            try:
                with open(plans_folder+"/"+job_nr+"/"+version_nr+".pickle", "rb") as d:
                    execution_plan = pickle.load(d)
            except:
                #print(f"Problems finding plan for {file}")
                continue
            try:
                full_execution_plan = feature_ext.match_cost_plan(execution_plan, cost_plan)
            except:
                print(file)
                continue
            if return_featurized:
                featurized_plan = feature_ext.featurize_plan(full_execution_plan)
                featurized_plans[file_name] = featurized_plan
            else:
                featurized_plans[file_name] = full_execution_plan
    return featurized_plans

def featurize_with_labels(plans_folder, cost_folder, label_csv, max_score = 5, score_function = "linear", extra_for_min = True):
    featurized_plans = get_features_with_cost_from_folder(plans_folder, cost_folder)
    label_dict = {}
    
    df = pd.read_csv(label_csv, index_col = 0)
    df["Job_nr"] = df["Unnamed: 0.1"].apply(lambda x: x.split("_")[0])
    
    for job in pd.unique(df["Job_nr"]):
        temp_df = df[df["Job_nr"]==job].copy()
        a = np.array(temp_df["Sum"])
        if score_function == "linear":
            a[a==-2] = max(a)*2 # Necessary, otherwise min(a) == -2
            temp_df["scores"] = calculate_linear_scores(a, n = max_score)
        elif score_function == "histogram":
            temp_df["scores"] = calculate_histogram_score(a, nr_bins = max_score, extra_bin_for_min = extra_for_min)
        elif score_function == "agglomerative":
            temp_df["scores"] = calculate_agglomerative_score(a, nr_clusters = max_score, extra_bin_for_min = extra_for_min)
        
        for idx, row in temp_df.iterrows():
            label_dict[idx] = float(row["scores"])
    
    return featurized_plans, label_dict
        
def calculate_linear_scores(scores, n = 5):
    best = min(scores)
    ten_best = best*n
    if not ten_best:
        ten_best = n    
    # apply linear scores:
    m = -n/(ten_best - best)
    b = -1*m*(ten_best)
    
    scores = m*scores+b
    return scores

def calculate_histogram_score(labels, nr_bins = 10, extra_bin_for_min = True):
    labels_copy = labels[labels != -2]
    hist, edges = np.histogram(labels_copy, nr_bins)
    edges_inv = edges[::-1]
    result = np.digitize(labels,edges_inv)
    result[labels == -2] = -1
    if extra_bin_for_min:
        result[labels == min(labels_copy)] = nr_bins+1
    return result

def calculate_agglomerative_score(labels, nr_clusters=10, extra_bin_for_min = True):    
    # Todo: This won't work yet because there is e.g. a max score of 3 for list of length 3 
    
    labels_copy = labels[labels != -2].reshape(-1, 1)
    if len(labels_copy) < nr_clusters:
        nr_clusters = len(labels_copy)
    
    clustering = AgglomerativeClustering(n_clusters = nr_clusters).fit_predict(labels_copy)
    maxima = [np.max(labels_copy[np.where(clustering == i)]) for i in range(nr_clusters)]
    sort = np.concatenate((np.sort(maxima)[::-1],np.array([0])))
    result = np.digitize(labels,sort)
    result[labels == -2] = -1
    if extra_bin_for_min:
        result[labels == min(labels_copy)] = nr_clusters+1
    return result


def get_left_child(node):
    if len(node) != 3:
        return None
    return node[1]

def get_right_child(node):
    if len(node) != 3:
        return None
    return node[2]
    
def get_features(node):
    return node[0]   
    
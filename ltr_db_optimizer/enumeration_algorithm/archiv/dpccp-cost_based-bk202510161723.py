import copy

import networkx as nx
import random
import torch

import ltr_db_optimizer.enumeration_algorithm.utils as utils 
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes
from ltr_db_optimizer.model.model_structures.comparison_net2 import LTRComparisonNet

from ltr_db_optimizer.model.model_structures.proposals_by_XL import *
from ltr_db_optimizer.model.model_structures.comparison_model import LTRComparisonNet as LTRComparisonNet_comp


class DPccp:
    
    def __init__(self, model = None, graph = None, joiner = None, top_k = 1, comparison = False):
        if model:
            if not comparison:
                model_archi_name = model.split("MODEL_")[1].split("_")[0]

                if model_archi_name == "HM":
                    self.model = LTRComparisonNet(10, 6)
                    print('load LTRComparisonNet HM!!!')

                else:
                    self.model = eval(model_archi_name)(10, 6)
                    print(f'load model:{model_archi_name}!!!')
                # self.model.load_state_dict(torch.load(model), strict=False)
                self.model.load_state_dict(torch.load(model))

            else:
                self.model = LTRComparisonNet_comp(10,6)
                self.model.load_state_dict(torch.load(model))                
        else:
            self.model = None
        self.graph = graph
        self.joiner = joiner
        self.top_k = top_k

    def recurse_all_the_child_nodes(self, plan):
        print('plan detail: ', plan.name, plan.contained_tables, plan.id)

        plan = copy.copy(plan)
        if plan.has_left_child():
            print('plan.left_child.name', plan.left_child.name)
            print('plan.left_child.contained_tables', plan.left_child.contained_tables)
        else:
            print('plan has no left child')

        if plan.has_right_child():
            print('plan.right_child.name', plan.right_child.name)
            print('plan.right_child.contained_tables', plan.right_child.contained_tables)
        else:
            print('plan has no right child')

        if plan.has_left_child():
            print('enter left child traverse')
            self.recurse_all_the_child_nodes(plan.left_child)

        if plan.has_right_child():
            print('enter right child traverse')
            self.recurse_all_the_child_nodes(plan.right_child)

        print("######")

    def enumerate(self, name_in_data=False):
        """
        :param graph: networkx Graph in BFS
        """
        best_parts = {}
        full = ""
        
        for i in list(self.graph.nodes):
            print('graph i: ', i)
            dict_name = str(i)
            i_name = dict_name if not name_in_data else nx.get_node_attributes(self.graph, "old_name")[i]
            print('graph node i_name: ', i_name)
            best_parts[dict_name] = self.joiner.get_scan(i_name)

        # print('csg_cmp_pairs:', list(self.get_csg_cmp_pairs()))
        # cnt = 0
        for csg, cmp in self.get_csg_cmp_pairs():
            csg_name = self.to_name(csg)
            cmp_name = self.to_name(cmp)
            full_name = self.to_name(csg, cmp)
            print('csg cmp csg_name cmp_name full_name', csg, cmp, csg_name, cmp_name, full_name)
            assert csg_name in best_parts
            assert cmp_name in best_parts
            
            # wenn nicht länge 1
            # filtern für beste gefundene Lösung von csg und cmp
            # alle möglichen Varianten (verschiedene Join Types und cmp-csg vs. csg-cmp) einfügen in best_parts[full_name]
            if csg_name in best_parts:
                print('inner#csg:', csg_name, len(best_parts[csg_name]))
                if len(best_parts[csg_name]) > self.top_k:
                    print('csg_name > topk')
                    best_parts[csg_name] = self.reduce(best_parts[csg_name])

                left = best_parts[csg_name]

            for lef_index, lef in enumerate(left):
                print('lef: ', lef_index, lef.name, lef.contained_tables, lef.id)

            for lef_index, lef in enumerate(left):
                print('!!!lef: ', lef_index, lef.name, lef.contained_tables, lef.id)
                self.recurse_all_the_child_nodes(lef)




            if cmp_name in best_parts:
                print('inner#cmp:', cmp_name, len(best_parts[cmp_name]))
                if len(best_parts[cmp_name]) > self.top_k:
                    print('cmp_name > topk')
                    best_parts[cmp_name] = self.reduce(best_parts[cmp_name])
                
                right = best_parts[cmp_name]

            for rit_index, rit in enumerate(right):
                print('rig: ', rit_index, rit.name, rit.contained_tables, rit.id)

            for rit_index, rit in enumerate(right):
                print('!!!rig: ', rit_index, rit.name, rit.contained_tables, rit.id)
                self.recurse_all_the_child_nodes(rit)

            possible_joins = self.joiner.get_join_possibilities(left, right)
            
            if full_name not in best_parts.keys():
                best_parts[full_name] = possible_joins
                print('aaa', full_name, len(possible_joins))
            else:
                best_parts[full_name].extend(possible_joins)
                print('bbb', full_name, len(possible_joins))

            full = full_name


        # gib besten Subplan zurück
        print('full name at last: ', full)
        print('accumulated full:', len(best_parts[full_name]))
        if len(best_parts[full]) > self.top_k:
            print('ccc')
            return self.reduce(best_parts[full])#[0]
        else:
            print('ddd')
            return best_parts[full]
    
    def reduce(self, plans, last = False):
        # hier noch model einfügen
        if self.model is None:
            print('hhhhhhh')
            return random.sample(plans, self.top_k)
        else:
            # should be the same sql for all
            prepared_plans = self.prepare_plans(plans, last)
            assert len(prepared_plans) == 2
            # if len(prepared_plans[1]) == 1:
            #     return prepared_plans[1]
            # predictions = self.model.predict_all(*prepared_plans).t()
            # get top_k plans
            plans_costs = prepared_plans[1]
            sorted_plans_costs = sorted(plans_costs, key=lambda x:x[1], reverse=False)

            print('sorted_plans_costs: ', list(sorted_plans_costs))

            k = self.top_k if not last else 1
            selected_plans = [item[0] for item in list(sorted_plans_costs)[:k]]
            return selected_plans
                
    
    def prepare_plans(self, plans):
        return plans
        
    def get_csg_cmp_pairs(self):
        for csg in self.enumerate_csg():
            for cmp in self.enumerate_cmp(csg):
                yield csg, cmp
    
    def enumerate_csg(self):
        # For all nodes i in reversed BFS
        for i in reversed(list(self.graph.nodes)):
            yield from self.yield_enumerate_csg(i)
    
    def yield_enumerate_csg(self, i: int):
        def filter_smaller(n):
            return n <= i
        yield [i]
        yield from self.enumerate_csg_rec([i], nx.subgraph_view(self.graph, filter_node=filter_smaller).nodes)
        
    def get_union_nodes(self, S, subset):
        try:
            subset = list(subset)
            S = list(S)
        except:
            raise Exception("subset should be convertible to list")
            
        def filter_with_subset(n):
            return n in S or n in subset
        
        return nx.subgraph_view(self.graph, filter_node=filter_with_subset).nodes
        
    def enumerate_csg_rec(self, S, X):
        N = self.get_neighbors_of_subgraph(S, X)
        for subset in utils.powerset(N):
            yield self.get_union_nodes(S, subset)
            
        for subset in utils.powerset(N):
            yield from self.enumerate_csg_rec(self.get_union_nodes(S, subset), 
                                              self.get_union_nodes(X, N))
        
    def get_neighbors_of_subgraph(self, S, X):
        assert all([node in X for node in S])
        N = []
        for node in S:
            N.extend(self.graph.neighbors(node)-X)
        return set(N)
        
    def enumerate_cmp(self, S_1):
        X = self.get_union_nodes(S_1, self.get_b_min(S_1))
        N = self.get_neighbors_of_subgraph(S_1, X)
        for node in reversed(list(N)):
            yield [node]
            yield from self.enumerate_csg_rec([node], self.get_union_nodes(X, N))
                           
    def get_b_min(self, S_1: list):
        minimum = min(S_1)
        def filter_higher(n):
            return n <= minimum
        return nx.subgraph_view(self.graph, filter_node=filter_higher).nodes
        
    def to_name(self, nodes_1, nodes_2 = None):
        nodes_1 = list(nodes_1)
        if nodes_2:
            nodes_1 += list(nodes_2)
        return "".join(str(i) for i in sorted(nodes_1))



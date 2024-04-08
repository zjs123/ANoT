from graph import Graph
from model import Model
from temporal_model import Temporal_Model
from evaluator import Evaluator
import argparse
import numpy as np
import itertools
import functools
import random
from math import log2 as log
import heapq
import copy
from numba import jit
from tqdm import tqdm
import multiprocessing as mp
from memory_profiler import profile

class Searcher:
    '''
    A class which searches for the optimal model.
    '''
    def __init__(self, graph):
        self.graph = graph
        self.evaluator = Evaluator(graph)
        self.candidates = list(self.graph.candidates.keys())
        self.null_val = self.evaluator.length_graph_with_model_no_time(Model(self.graph))[0]
    
    
    def get_chain_pairs(self):
        max_rule_size = 10000000
        candidate_rule_pairs = dict()
        rule_pair_to_times = dict()
        pair_2_rtd = self.graph.e_pair_to_rtd
        for key in tqdm(pair_2_rtd.keys()):
            if len(candidate_rule_pairs.keys()) >= max_rule_size:
                return candidate_rule_pairs, rule_pair_to_times
            sub = key[0]
            obj = key[1]

            r_squence = [i[0] for i in pair_2_rtd[key]]
            t_squence = [i[1] for i in pair_2_rtd[key]]
            d_squence = [i[2] for i in pair_2_rtd[key]]

            for index in range(len(r_squence)-1):
                pre_pred = r_squence[index]
                aft_pred = r_squence[index+1]

                pre_dir = d_squence[index]
                aft_dir = d_squence[index+1]

                pre_time = t_squence[index]
                aft_time = t_squence[index+1]

                candidate_pre_rule = self.graph.triple_to_rule[(sub, pre_pred, obj, pre_dir)]
                candidate_aft_rule = self.graph.triple_to_rule[(sub, aft_pred, obj, aft_dir)]

                for pre_rule in candidate_pre_rule:
                    sub_type, obj_type = pre_rule[0], pre_rule[2]
                    for aft_rule in candidate_aft_rule:
                        if aft_rule[0] != sub_type or aft_rule[2] != obj_type:
                            continue
                        if (pre_rule, aft_rule) not in candidate_rule_pairs.keys():
                            candidate_rule_pairs[(pre_rule, aft_rule)] = set()
                            rule_pair_to_times[(pre_rule, aft_rule)] = list()
                        if aft_dir == 'in':
                            candidate_rule_pairs[(pre_rule, aft_rule)].add((obj, aft_pred, sub, aft_time))
                            rule_pair_to_times[(pre_rule, aft_rule)].append(aft_time - pre_time)
                            if (obj, aft_pred, sub, aft_time) not in self.graph.fact_list:
                                print("error_2")
                        if aft_dir == 'out':
                            candidate_rule_pairs[(pre_rule, aft_rule)].add((sub, aft_pred, obj, aft_time))
                            rule_pair_to_times[(pre_rule, aft_rule)].append(aft_time - pre_time)
                            if (sub, aft_pred, obj, aft_time) not in self.graph.fact_list:
                                print("error_2")
                        
        return candidate_rule_pairs, rule_pair_to_times
    

    def get_triangles_helper(self, time_target_triple):
        triple_sample_size = 500 # 500
        rule_sample_size = 10 #10
        covered_set = set()
        triangle_list = []

        #print('1')
        target_triple, time = time_target_triple[0], time_target_triple[1]
        sub = target_triple[0]
        pred = target_triple[1]
        obj = target_triple[2]
        target_dir = target_triple[-1]
        pre_time_list = [time]

        #print('2')
        if time in self.graph.e_2_t_list[sub].keys():
            pre_time_list = pre_time_list+self.graph.e_2_t_list[sub][time]
        if time in self.graph.e_2_t_list[obj].keys():
            pre_time_list = pre_time_list+self.graph.e_2_t_list[obj][time]

        for in_time in pre_time_list[:10]:
            triple_list_2rd = set()
            delete_set = set()
            #print('3')
            if sub in self.graph.t_e_2_triple[in_time].keys():
                triple_list_2rd = triple_list_2rd | self.graph.t_e_2_triple[in_time][sub] 
                delete_set = self.graph.t_e_2_triple[in_time][sub]
            if obj in self.graph.t_e_2_triple[in_time].keys():
                triple_list_2rd = triple_list_2rd | self.graph.t_e_2_triple[in_time][obj]
                delete_set = delete_set & self.graph.t_e_2_triple[in_time][obj]
            
            triple_list_2rd = triple_list_2rd - delete_set

            for triple_2rd in triple_list_2rd:
                sub_2 = triple_2rd[0]
                pred_2 = triple_2rd[1]
                obj_2 = triple_2rd[2]
                
                #print('4')
                co_e = list(set([sub, obj]) & set([sub_2, obj_2]))[0]
                candidate_e_1 = sub if obj == co_e else obj
                candidate_e_2 = sub_2 if obj_2 == co_e else obj_2
                
                pre_time_list = [in_time]
                if in_time in self.graph.e_2_t_list[candidate_e_1].keys():
                    pre_time_list = pre_time_list+self.graph.e_2_t_list[candidate_e_1][in_time]
                if in_time in self.graph.e_2_t_list[candidate_e_2].keys():
                    pre_time_list = pre_time_list+self.graph.e_2_t_list[candidate_e_2][in_time]

                for close_time in pre_time_list[:10]:
                    #print('5')
                    triple_list_3rd = set()
                    if (candidate_e_1, candidate_e_2) in self.graph.t_pair_2_triple[close_time].keys():
                        triple_list_3rd = triple_list_3rd | self.graph.t_pair_2_triple[close_time][(candidate_e_1, candidate_e_2)]
                    if (candidate_e_2, candidate_e_1) in self.graph.t_pair_2_triple[close_time].keys():
                        triple_list_3rd = triple_list_3rd | self.graph.t_pair_2_triple[close_time][(candidate_e_2, candidate_e_1)]
                    
                    for triple_3rd in triple_list_3rd:
                        candidate_1_rule = self.graph.triple_to_rule[target_triple][:rule_sample_size]
                        candidate_2_rule = self.graph.triple_to_rule[triple_2rd][:rule_sample_size]
                        candidate_3_rule = self.graph.triple_to_rule[triple_3rd][:rule_sample_size]
                        dir = triple_3rd[-1]
                        
                        #print('6')
                        rule_1_2 = list(itertools.product(candidate_1_rule, candidate_2_rule))
                        for rule_1, rule_2 in rule_1_2:
                            #print('7')
                            co_e_list = list(set([rule_1[0], rule_1[2]]) & set([rule_2[0], rule_2[2]]))
                            if len(co_e_list) != 1:
                                continue
                            co_e = co_e_list[0]
                            rule_3_e_1 = rule_1[0] if rule_1[2] == co_e else rule_1[2]
                            rule_3_e_2 = rule_2[0] if rule_2[2] == co_e else rule_2[2]
                            combined_rules = [(rule_3_e_1, triple_3rd[1], rule_3_e_2, 'out'), (rule_3_e_2, triple_3rd[1], rule_3_e_1, 'out'), (rule_3_e_1, triple_3rd[1], rule_3_e_2, 'in'), (rule_3_e_2, triple_3rd[1], rule_3_e_1, 'in')]
                            for rule_3 in combined_rules:
                                if rule_3 in candidate_3_rule:
                                    if target_dir == 'out':
                                        triangle_list.append([(rule_2, rule_3, rule_1), (sub, pred, obj, time), max(time - in_time, time - close_time)])
                                        covered_set.add((sub, pred, obj, time))
                                        
                                    if target_dir == 'in':
                                        triangle_list.append([(rule_2, rule_3, rule_1),(obj, pred, sub, time), max(time - in_time, time - close_time)])
                                        covered_set.add((obj, pred, sub, time))
        return triangle_list

    
    
    def get_triangles_mp(self):
        triangle_list = dict()
        triangle_to_times = dict()
        triple_sample_size = 500 # 500
        rule_sample_size = 10 #10

        p = mp.Pool(4)
        time_list = list(self.graph.t_2_triple.keys())
        #print('a')
        for t in tqdm(range(len(time_list))):
            target_triple_list = list(self.graph.t_2_triple[time_list[t]])[:triple_sample_size]
            tmp_list = [[target_triple_list[i], time_list[t]] for i in range(len(target_triple_list))]
            #print('b')
            time_2_triangles = p.map(self.get_triangles_helper, tmp_list)
            #print('c')
            for triangles in time_2_triangles:
                rule_comb, fact, span = triangles[0], triangles[1], triangles[2]
                if rule_comb not in triangle_list.keys():
                    triangle_list[rule_comb] = set()
                if rule_comb not in triangle_to_times.keys():
                    triangle_to_times[rule_comb] = []
                triangle_list[rule_comb].add(fact)
                triangle_to_times[rule_comb].append(span)
                                                
        return triangle_list, triangle_to_times


    def get_triangles(self):
        triple_sample_size = 500 # 500
        rule_sample_size = 5 # 5
        max_rule_size = 10000000

        triangle_list = dict()
        triangle_to_times = dict()
        covered_set = set()

        for time in tqdm(sorted(list(self.graph.t_2_triple.keys()), reverse=True)):
            target_triple_list = list(self.graph.t_2_triple[time])
            if len(triangle_list.keys()) >= max_rule_size:
                return triangle_list, triangle_to_times
            for target_triple in target_triple_list[:triple_sample_size]:
                sub = target_triple[0]
                pred = target_triple[1]
                obj = target_triple[2]
                target_dir = target_triple[-1]

                pre_time_list = [time]
                if time in self.graph.e_2_t_list[sub].keys():
                    pre_time_list = pre_time_list+self.graph.e_2_t_list[sub][time]
                if time in self.graph.e_2_t_list[obj].keys():
                    pre_time_list = pre_time_list+self.graph.e_2_t_list[obj][time]

                for in_time in pre_time_list[:10]:
                    triple_list_2rd = set()
                    delete_set = set()
                    if sub in self.graph.t_e_2_triple[in_time].keys():
                        triple_list_2rd = triple_list_2rd | self.graph.t_e_2_triple[in_time][sub] 
                        delete_set = self.graph.t_e_2_triple[in_time][sub]
                    if obj in self.graph.t_e_2_triple[in_time].keys():
                        triple_list_2rd = triple_list_2rd | self.graph.t_e_2_triple[in_time][obj]
                        delete_set = delete_set & self.graph.t_e_2_triple[in_time][obj]
                    
                    triple_list_2rd = list(triple_list_2rd - delete_set)

                    for triple_2rd in triple_list_2rd:
                        sub_2 = triple_2rd[0]
                        pred_2 = triple_2rd[1]
                        obj_2 = triple_2rd[2]
                        
                        co_e = list(set([sub, obj]) & set([sub_2, obj_2]))[0]
                        candidate_e_1 = sub if obj == co_e else obj
                        candidate_e_2 = sub_2 if obj_2 == co_e else obj_2
                        
                        pre_time_list = [in_time]
                        if in_time in self.graph.e_2_t_list[candidate_e_1].keys():
                            pre_time_list = pre_time_list+self.graph.e_2_t_list[candidate_e_1][in_time]
                        if in_time in self.graph.e_2_t_list[candidate_e_2].keys():
                            pre_time_list = pre_time_list+self.graph.e_2_t_list[candidate_e_2][in_time]

                        for close_time in pre_time_list[:10]:
                            triple_list_3rd = set()
                            if (candidate_e_1, candidate_e_2) in self.graph.t_pair_2_triple[close_time].keys():
                                triple_list_3rd = triple_list_3rd | self.graph.t_pair_2_triple[close_time][(candidate_e_1, candidate_e_2)]
                            if (candidate_e_2, candidate_e_1) in self.graph.t_pair_2_triple[close_time].keys():
                                triple_list_3rd = triple_list_3rd | self.graph.t_pair_2_triple[close_time][(candidate_e_2, candidate_e_1)]
                            
                            for triple_3rd in list(triple_list_3rd):
                                candidate_1_rule = self.graph.triple_to_rule[target_triple][:rule_sample_size]
                                candidate_2_rule = self.graph.triple_to_rule[triple_2rd][:rule_sample_size]
                                candidate_3_rule = self.graph.triple_to_rule[triple_3rd][:rule_sample_size]
                                dir = triple_3rd[-1]
                                
                                rule_1_2 = list(itertools.product(candidate_1_rule, candidate_2_rule))
                                for rule_1, rule_2 in rule_1_2:
                                    co_e_list = list(set([rule_1[0], rule_1[2]]) & set([rule_2[0], rule_2[2]]))
                                    if len(co_e_list) != 1:
                                        continue
                                    co_e = co_e_list[0]
                                    rule_3_e_1 = rule_1[0] if rule_1[2] == co_e else rule_1[2]
                                    rule_3_e_2 = rule_2[0] if rule_2[2] == co_e else rule_2[2]
                                    combined_rules = [(rule_3_e_1, triple_3rd[1], rule_3_e_2, 'out'), (rule_3_e_2, triple_3rd[1], rule_3_e_1, 'out'), (rule_3_e_1, triple_3rd[1], rule_3_e_2, 'in'), (rule_3_e_2, triple_3rd[1], rule_3_e_1, 'in')]
                                    for rule_3 in combined_rules:
                                        if rule_3 in candidate_3_rule:
                                            if (rule_2, rule_3, rule_1) not in triangle_list.keys():
                                                triangle_list[(rule_2, rule_3, rule_1)] = set()
                                                triangle_to_times[(rule_2, rule_3, rule_1)] = list()
                                            if target_dir == 'out':
                                                sorted_pre = sorted([rule_2, rule_3], key= lambda g: (g[0], g[1], g[2]))
                                                if (sorted_pre[0], sorted_pre[1], rule_1) not in triangle_list.keys():
                                                    triangle_list[(sorted_pre[0], sorted_pre[1], rule_1)] = set()
                                                    triangle_to_times[(sorted_pre[0], sorted_pre[1], rule_1)] = []
                                                triangle_list[(sorted_pre[0], sorted_pre[1], rule_1)].add((sub, pred, obj, time))
                                                triangle_to_times[(sorted_pre[0], sorted_pre[1], rule_1)].append(max(time - in_time, time - close_time))
                                                if (sub, pred, obj, time) not in self.graph.fact_list:
                                                    print("error_triple_3")
                                            if target_dir == 'in':
                                                sorted_pre = sorted([rule_2, rule_3], key= lambda g: (g[0], g[1], g[2]))
                                                if (sorted_pre[0], sorted_pre[1], rule_1) not in triangle_list.keys():
                                                    triangle_list[(sorted_pre[0], sorted_pre[1], rule_1)] = set()
                                                    triangle_to_times[(sorted_pre[0], sorted_pre[1], rule_1)] = []
                                                triangle_list[(sorted_pre[0], sorted_pre[1], rule_1)].add((obj, pred, sub, time))
                                                triangle_to_times[(sorted_pre[0], sorted_pre[1], rule_1)].append(max(time - in_time, time - close_time))
                                                if (obj, pred, sub, time) not in self.graph.fact_list:
                                                    print("error_triple_3")
                                                
        return triangle_list, triangle_to_times
    
    def get_rule_triangle(self, target_triple, sub_triple, obj_triple, triangle_list, covered_set, time):
        tmp_set = set()
        tar_sub, tar_pred, tar_obj, tar_dir = target_triple[0], target_triple[1], target_triple[2], target_triple[3]
        co_e_list = list(set([sub_triple[0], sub_triple[2]]) & set([obj_triple[0], obj_triple[2]]))
        co_e = co_e_list[0]
        if sub_triple[0] == co_e:
            sub_triple_type = 'co_sub'
        else:
            sub_triple_type = 'sub_co'
        sub_pred, sub_dir = sub_triple[1], sub_triple[3]

        if obj_triple[0] == co_e:
            obj_triple_type = 'co_obj'
        else:
            obj_triple_type = 'obj_co'
        obj_pred, obj_dir = obj_triple[1], obj_triple[3]

        for type_1 in self.graph.node_to_labels[tar_sub][:5]:
            for type_2 in self.graph.node_to_labels[tar_obj][:5]:
                target_rules = (type_1, tar_pred, type_2, tar_dir)
                for type_3 in self.graph.node_to_labels[co_e][:5]:
                    
                    if sub_triple_type == 'co_sub':
                        sub_rules = (type_3, sub_pred, type_1, sub_dir)
                    else:
                        sub_rules = (type_1, sub_pred, type_3, sub_dir)

                    if obj_triple_type == 'co_obj':
                        obj_rules = (type_3, obj_pred, type_1, obj_dir)
                    else:
                        obj_rules = (type_1, obj_pred, type_3, obj_dir)

                    if sub_rules in self.graph.candidates.keys() and obj_rules in self.graph.candidates.keys() and target_rules in self.graph.candidates.keys():
                        rule_triangle = (sub_rules, obj_rules, target_rules)
                        if rule_triangle not in triangle_list.keys():
                            triangle_list[rule_triangle] = set()
                        if target_triple[-1] == 'out':
                            covered_set.add((target_triple[0], target_triple[1], target_triple[2], time))
                            triangle_list[rule_triangle].add((target_triple[0], target_triple[1], target_triple[2], time))
                        if target_triple[-1] == 'in':
                            covered_set.add((target_triple[2], target_triple[1], target_triple[0], time))
                            triangle_list[rule_triangle].add((target_triple[2], target_triple[1], target_triple[0], time))

    def get_triangles_new(self):
        triple_sample_size = 1000
        rule_sample_size = 5
        covered_set = set()
        triangle_list = dict()

        for time in tqdm(sorted(list(self.graph.t_2_triple.keys()), reverse=True)):
            target_triple_list = list(self.graph.t_2_triple[time])
            for target_triple in target_triple_list:
                if target_triple in covered_set:
                    continue
                sub = target_triple[0]
                pred = target_triple[1]
                obj = target_triple[2]
                pre_time_list_sub = set([time])
                pre_time_list_obj = set([time])
                if time in self.graph.e_2_t_list[sub].keys():
                    pre_time_list_sub = list(pre_time_list_sub | set(self.graph.e_2_t_list[sub][time]))[:10]
                if time in self.graph.e_2_t_list[obj].keys():
                    pre_time_list_obj = list(pre_time_list_obj | set(self.graph.e_2_t_list[obj][time]))[:10]

                triple_list_sub = set()
                triple_list_obj = set()
                delete_set = set()
                for in_time in pre_time_list_sub:
                    triple_list_sub = triple_list_sub | self.graph.t_e_2_triple[in_time][sub]
                    delete_set = self.graph.t_e_2_triple[in_time][sub]
                for in_time in pre_time_list_obj:
                    triple_list_obj = triple_list_obj | self.graph.t_e_2_triple[in_time][obj]
                    delete_set = delete_set & self.graph.t_e_2_triple[in_time][obj]
                
                triple_list_sub = list(triple_list_sub - delete_set)
                triple_list_obj = list(triple_list_obj - delete_set)
                
                for sub_triple in triple_list_sub[:triple_sample_size]:
                    for obj_triple in triple_list_obj[:triple_sample_size]:
                        if len(set([sub_triple[0], sub_triple[2], target_triple[0], target_triple[2]])) != 3:
                            continue
                        if len(set([obj_triple[0], obj_triple[2], target_triple[0], target_triple[2]])) != 3:
                            continue
                        if len(set([sub_triple[0], sub_triple[2], obj_triple[0], obj_triple[2]])) != 3:
                            continue
                        self.get_rule_triangle(target_triple, sub_triple, obj_triple, triangle_list, covered_set, time)
        return triangle_list
    
    def reduction_in_error(self, g):
            
            #:g: a rule
            _model = Model(self.graph)
            _model.add_rule(g)
            # compute L(G|M_0) - L(G|M \cup {g}) 即增加该规则可以减少多少不可覆盖的知识 （error negative）
            red_in_err = self.null_val - self.evaluator.length_graph_with_model_no_time(_model)[0]
            return red_in_err
    
    def rank_rules(self):
        # L(G|M_0)
        
        self.candidates = sorted(self.candidates,
                                 reverse=True,
                                 key=lambda g: (self.reduction_in_error(g), # reduction in error
                                                len(self.graph.candidates[g]['ca_to_size']), # number of correct assertions
                                                g[0])) # rule root labels
        
        return self.candidates
    
    class BoundedMinHeap:
        '''
        A Min Heap that only allows :bound: items.
        If :bound: is << len(items), then this can be more efficient for finding the top k than sorting the whole list.
        '''
        def __init__(self, bound, key=lambda it: it):
            self.bound = bound
            self.key = key
            self._data = list()

        def push(self, it):
            if len(self._data) < self.bound:
                heapq.heappush(self._data, (self.key(it), it))
            else:
                heapq.heappushpop(self._data, (self.key(it), it))

        def get_reversed(self):
            temp = list()
            while len(self._data) > 0:
                temp.append(heapq.heappop(self._data)[1])
            return list(reversed(temp))

    def build_model_top_k_freq(self, k):
        '''
        Build a model containing the k rules with the most correct assertions.
        '''
        model = Model(self.graph)
        heap = Searcher.BoundedMinHeap(bound=k, key=lambda rule: len(self.graph.candidates[rule]['ca_to_size']))
        for rule in self.candidates:
            heap.push(rule)
        for rule in heap.get_reversed():
            model.add_rule(rule)
        return model

    def build_model_top_k_coverage(self, k):
        '''
        Build a model containing the k rules that explain the most edges.
        '''
        model = Model(self.graph)
        heap = Searcher.BoundedMinHeap(bound=k, key=lambda rule: sum(list((self.graph.candidates[rule]['ca_to_size']).values())))
        for rule in self.candidates:
            heap.push(rule)
        for rule in heap.get_reversed():
            model.add_rule(rule)
        return model

    def check_qualify(self, evaluator, verbosity, rule_and_new_labels):
        '''
        Check whether adding more labels to a rule (qualifying it) leads to improvements in MDL terms.

        :evaluator: an Evaluator object with which to compute MDL scores
        :verbosity: how much to print to the log
        :rule_and_new_labels: (rule, the rule's new labels if qualified)

        :return: True if the rule is qualified, False if not.
        '''
        rule, new_labels = rule_and_new_labels
        old_rule = rule
        root, children = rule
        rule = (new_labels, children)
        # create M_0
        model = Model(self.graph)
        # create M_0 \cup {g without qualification}
        model.add_rule(old_rule)
        # compute L(G, M_0 \cup {g without qualification})
        # (Note: this computation works because the newly added labels are contained by all correct assertion starting points, so L(G|M) does not change)
        cost_without_qualification = evaluator.length_rule(old_rule) + evaluator.length_rule_assertions(old_rule, model)
        # replace the rule with the qualified version to make M_0 \cup {g with qualification}
        model.rules[rule] = model.rules[old_rule]
        del model.rules[old_rule]
        # compute L(G, M_0 \cup {g with qualification})
        cost_with_qualification = evaluator.length_rule(rule) + evaluator.length_rule_assertions(rule, model)

        qualified = False
        # if the cost went down, then keep the qualifiers
        if cost_with_qualification < cost_without_qualification:
            # update the data structures so that the new, qualified rules has the same correct assertions as the old, unqualified rule
            self.graph.candidates[rule] = self.graph.candidates[old_rule]
            # delete the data on the old, unqualified rule
            del self.graph.candidates[old_rule]
            return True
        return False

    def label_qualify(self, verbosity):
        '''
        Qualify rules where appropriate (Section 4.1.1).

        :verbosity: How often to print progress.
        '''
        num_qualified = 0
        rule_to_new_labels = dict()
        if verbosity > 0:
            print('Qualifying candidate rules (Section 4.1.2).')
        n = len(self.candidates)
        for i, rule in enumerate(self.candidates):
            root = rule[0]
            heads = list(self.graph.candidates[rule]['ca_to_size'].keys())
            # if all correct assertions share more labels than the current root label, then we can add this label
            shared_by_all = set(self.graph.labels(heads[0]))
            for head in heads[1:]:
                shared_by_all = shared_by_all.intersection(self.graph.labels(head))
                if shared_by_all == {root}:
                    break
            if shared_by_all != {root}:
                rule_to_new_labels[rule] = tuple(sorted(shared_by_all))
            if verbosity > 0 and i > 0 and i % verbosity == 0:
                print('{}% of candidates processed.'.format(round(i / n * 100, 2)))
        if verbosity > 0:
            print('{}% of candidates processed.'.format(round(i / n * 100, 2)))

        evaluator = Evaluator(self.graph)

        n = len(rule_to_new_labels.items())
        for i, rule_and_new_labels in enumerate(rule_to_new_labels.items()):
            # check whether the qualifier leads to improvements
            qualified = self.check_qualify(evaluator, verbosity, rule_and_new_labels)
            if qualified:
                num_qualified += 1
            if verbosity > 0 and i > 0 and i % verbosity == 0:
                print('{}'.format(i / n))

        self.candidates = list(self.graph.candidates.keys())
        if verbosity > 0:
            print('{}% of candidates qualified.'.format(round(num_qualified / n * 100, 2)))

    #@profile
    def build_temporal_model(self, raw_model, passes=1, verbosity=1000000):
        # get candidate chain combinations
        print("generating chain combinations....")
        candidate_rule_pairs, pair_2_times = self.get_chain_pairs()
        print("get candidate rule pairs: " + str(len(candidate_rule_pairs)))

        # get candidate chain combinations
        print("generating riangle combinations....")
        candidate_rule_triangles, triangle_2_times = self.get_triangles()
        ranked_rule_triangles = sorted(list(candidate_rule_triangles.keys()), reverse=True, key=lambda g: (
                                                 # number of correct assertions
                                                len(self.graph.candidates[g[-1]]['triples']),
                                                len(self.graph.candidates[g[0]]['triples']),
                                                len(candidate_rule_triangles[g]),
                                                g[-1], g[0], g[1]))
        new_candidate_rule_triangles = dict()
        for triangles in ranked_rule_triangles[:7000000]:
            new_candidate_rule_triangles[triangles] = candidate_rule_triangles[triangles]
        print("get candidate rule riangle: " + str(len(candidate_rule_triangles)))

        # select rule pairs
        candidate_rule_pairs.update(new_candidate_rule_triangles)
        pair_2_times.update(triangle_2_times)
        selected_rule_pairs = []
        selected_rule_triangles = []
        tmp_temporal_model = Temporal_Model(raw_model)
        best_val, best_model_length, best_neg_edge, best_neg_node = self.evaluator.evaluate_temporal(tmp_temporal_model, with_lengths=True)
        if verbosity > 0:
            print('Null encoding cost: {}'.format(round(best_val, 4)))
        ranked_rule_pairs = sorted(list(candidate_rule_pairs.keys()), reverse=True, key=lambda g: (
                                                # number of correct assertions
                                                len(self.graph.candidates[g[-1]]['triples']),
                                                len(self.graph.candidates[g[0]]['triples']),
                                                len(candidate_rule_pairs[g]),
                                                g[-1], g[0], g[1]))
        
        for _pass in range(1, passes + 1):
            if verbosity > 0:
                print('Starting pass {}.'.format(_pass))
            _pass += 1
            for i, rule in tqdm(enumerate(ranked_rule_pairs)):
                if rule in tmp_temporal_model.edges:
                    continue

                tmp_temporal_model.add_edge(rule, candidate_rule_pairs, pair_2_times)
                val, model_length, neg_edge, neg_node = self.evaluator.evaluate_change_temporal(tmp_temporal_model, rule, best_model_length)
                tmp_temporal_model.remove_edge(rule, candidate_rule_pairs, pair_2_times)

                # if the cost didn't go down, remove the rule
                if val < best_val:
                    assert(val < best_val)
                    tmp_temporal_model.add_edge(rule, candidate_rule_pairs, pair_2_times, 'final')
                    if len(rule) == 2:
                        selected_rule_pairs.append(rule)
                    if len(rule) == 3:
                        selected_rule_triangles.append(rule)
                    best_val = val
                    best_model_length = model_length
            if verbosity > 0:
                print('Final number of combined_rules after pass {}: {}'.format(_pass, len(tmp_temporal_model.edges)))
        
        tmp_temporal_model.post_process()
        return tmp_temporal_model, selected_rule_pairs, selected_rule_triangles


    #@profile
    def build_model(self, rank='metric', order=['mdl_err', 'coverage', 'lex'], passes=3, label_qualify=True, verbosity=1000000):
        '''
        The core of the algorithm. (Sections 4.1.2-4.2.1)
        '''
        '''
        if label_qualify:
            self.label_qualify(verbosity=verbosity)
        '''
        print('ranking rules....')
        ranked_rules = self.rank_rules()
        # starts null
        print('constructing raw model...')
        model = Model(self.graph)

        best_val, best_model_length, best_neg_edge, best_neg_node = self.evaluator.evaluate(model, with_lengths=True)
        null_cost = best_val
        if verbosity > 0:
            print('Null encoding cost: {}'.format(round(best_val, 4)))
        num_cans = len(ranked_rules)
        tried = set()

        for _pass in range(1, passes + 1):
            if verbosity > 0:
                print('Starting pass {}.'.format(_pass))
            _pass += 1
            for i, rule in tqdm(enumerate(ranked_rules)):
                # build reverse
                sub, pred, obj, dr = rule
                reverse_rule = (obj, pred, sub, 'in' if dr == 'out' else 'out')
                if reverse_rule in self.graph.candidates:
                    
                    if rule in model.rules or reverse_rule in model.rules:
                        continue
                    
                    model.add_rule(rule)
                    val, model_length, neg_edge, neg_node = self.evaluator.evaluate_change(model, rule, best_model_length)
                    model.remove_rule(rule)
                    model.add_rule(reverse_rule)
                    rev_val, rev_model_length, rev_neg_edge, rev_neg_node = self.evaluator.evaluate_change(model, reverse_rule, best_model_length)
                    model.remove_rule(reverse_rule)
                    # if the cost didn't go down, remove the rule
                    if val <= rev_val < best_val:
                        assert(val <= rev_val)
                        assert(val < best_val)
                        model.add_rule(rule)
                        best_val = val
                        best_model_length = model_length
                    elif rev_val < best_val:
                        assert(rev_val < best_val)
                        assert(rev_val < val)
                        model.add_rule(reverse_rule)
                        best_val = rev_val
                        best_model_length = rev_model_length
                else:
                    if rule in model.rules:
                        continue

                    model.add_rule(rule)
                    val, model_length, neg_edge, neg_node = self.evaluator.evaluate_change(model, rule, best_model_length)

                    if val < best_val:
                        best_val = val
                        best_model_length = model_length
                    else: # if the cost didn't go down, remove the rule
                        model.remove_rule(rule)

                if verbosity > 0 and i > 0 and i % verbosity == 0:
                    print('Percent tried: {}. Num rules: {}. New encoding cost: {}. Percent saved: {}.'.format((i / num_cans) * 100, len(model.rules), best_val, ((null_cost - best_val) / null_cost) * 100))
            if verbosity > 0:
                print('Final number of rules after pass {}: {}'.format(_pass, len(model.rules)))

        # M*
        model.post_process()
        return model

if __name__ == '__main__':
    '''
    Entry point for the program.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',
                        '-g',
                        type=str,
                        required=True,
                        help='the name of the graph to summarize')
    args = parser.parse_args()
    graph = Graph(args.graph)

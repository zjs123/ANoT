from math import log2 as log
from scipy.special import gammaln
from collections import defaultdict
import numpy as np
from itertools import chain, combinations
from model_updater import ModelUpdater
import heapq
import warnings

warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, temporal_model):
        self.idify = False
        self.temporal_model = temporal_model

        self.binomial_map = dict()
        self.updater = ModelUpdater(self.temporal_model)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    def re_fresh(self):
        self.temporal_model = self.updater.raw_temporal_model
    
    def length_binomial(self, n, k):
        '''
        :n: n in (n choose k)
        :k: k in (n choose k)

        :return: log_2(n choose k)
        '''
        if (n, k) in self.binomial_map:
            return self.binomial_map[(n, k)]

        length = (gammaln(n + 1) - gammaln(k + 1) - gammaln((n + 1) - k)) / np.log(2)
        self.binomial_map[(n, k)] = length
        return length

    def score_blame_edge(self, node, pred):
        '''
        The number of bits describing the node as an exception to rules.
        '''
        score = 0
        rules = set()
        for labels in self.temporal_model.model.graph.labels(node):
            if labels in self.temporal_model.model.subject_to_rules:
                rules.update(self.temporal_model.model.subject_to_rules[labels]) # 先得到待检测事实的头实体所包含的类别，进而筛选包含这些类别的规则

        def has_pred(rule, pred):
            return rule[1] == pred

        rules = set(filter(lambda rule: has_pred(rule, pred), rules)) # 在上述规则中进而筛选有目标关系的规则

        for rule in rules:
            cas = set(self.temporal_model.model.graph.candidates[rule]['ca_to_size'].keys())
            if node not in cas:
                num_assertions = self.temporal_model.model.graph.nodes_with_type(rule[0])
                num_correct = len(self.temporal_model.model.graph.candidates[rule]['ca_to_size'])
                num_exceptions = num_assertions - num_correct
                assert(num_assertions > 0)
                assert(num_exceptions > 0)
                score += (1 / num_exceptions) * self.length_binomial(num_assertions, num_exceptions)
        return score

    def conceptual_score(self, node, pred, node_2, time, dir, hop = 2, max_span = 100, and_or = 'and', time_specific = False):
        score = 0
        rules = set()
        cover_rules = set()
        for labels in self.temporal_model.model.graph.labels(node):
            if labels in self.temporal_model.model.subject_to_rules:
                rules.update(self.temporal_model.model.subject_to_rules[labels]) # 先得到待检测事实的头实体所包含的类别，进而筛选包含这些类别的规则

        def has_pred(rule, pred):
            return rule[1] == pred

        rules = set(filter(lambda rule: has_pred(rule, pred), rules)) # 在上述规则中进而筛选有目标关系的规则
        
        num_cover_rules = 0
        for rule in rules:
            node_type = rule[2]
            if time_specific:
                if node_2 in self.temporal_model.model.graph.nodes_with_type(node_type, False) and self.temporal_model.model.graph.nodes_shortese_path_time(node, node_2, time) <= hop:
                    time_list_1 = abs(np.array(list(self.temporal_model.model.graph.e_2_t_list[node].keys())).astype('float64') - int(time))
                    time_list_2 = abs(np.array(list(self.temporal_model.model.graph.e_2_t_list[node_2].keys())).astype('float64') - int(time))
                    
                    if and_or == 'and':
                        signal = (len(time_list_1) >0 and min(time_list_1) < max_span) and (len(time_list_2) >0 and min(time_list_2) < max_span)
                    if and_or == 'or':
                        signal = (len(time_list_1) >0 and min(time_list_1) < max_span) or (len(time_list_2) >0 and min(time_list_2) < max_span)

                    if signal:
                        num_cover_rules += len(self.temporal_model.model.graph.candidates[rule]['triples'])  #1
                        cover_rules.add(rule)
            else:
                try:
                    tmp = self.temporal_model.model.graph.nodes_shortese_path(node, node_2) <= hop
                except:
                    cover_rules.add(rule)
                    continue
                
                if node_2 in self.temporal_model.model.graph.nodes_with_type(node_type, False) and self.temporal_model.model.graph.nodes_shortese_path(node, node_2) <= hop:
                    time_list_1 = abs(np.array(list(self.temporal_model.model.graph.e_2_t_list[node].keys())).astype('float64') - int(time))
                    time_list_2 = abs(np.array(list(self.temporal_model.model.graph.e_2_t_list[node_2].keys())).astype('float64') - int(time))
                    
                    if and_or == 'and':
                        signal = (len(time_list_1) >0 and min(time_list_1) < max_span) and (len(time_list_2) >0 and min(time_list_2) < max_span)
                    if and_or == 'or':
                        signal = (len(time_list_1) >0 and min(time_list_1) < max_span) or (len(time_list_2) >0 and min(time_list_2) < max_span)

                    if signal:
                        num_cover_rules += len(self.temporal_model.model.graph.candidates[rule]['triples'])  #1
                        cover_rules.add(rule)
        score += 1.0/float(num_cover_rules+1)
        
        return score, cover_rules
    

    def temporal_score(self, node, pred, node_2, time, dir, project_rules, max_span, max_steps, dsa, max_rule):
        score = 0
        filtered_rules = []
        proj_rules = dict()
        for rule in project_rules:
            if rule in self.temporal_model.nodes:
                filtered_rules.append(rule)
        filtered_rules = sorted(filtered_rules, reverse=True, key= lambda g: len(self.temporal_model.model.graph.candidates[g]['ca_to_size']))
        filtered_rules = filtered_rules[:max_rule]

        def detect_one_rule(triple, new_rule, old_rule, step):
            final_score = 0
            if step >= max_steps:
                return 0
            max_1_index = -1
            if (new_rule, old_rule) in self.temporal_model.rule_to_time.keys():
                pass
                example_time_line = self.temporal_model.rule_to_time[(new_rule, old_rule)]
                hist, bin_edges = np.histogram(example_time_line)
                hist = self.normalization(hist)
                max_1_index, max_2_index, max_3_index = heapq.nlargest(3, range(len(hist)), hist.take)

            if (triple[0], new_rule[1], triple[2], triple[-1]) in self.temporal_model.model.graph.triple_2_t.keys():
                time_line = np.array(list(self.temporal_model.model.graph.triple_2_t[(triple[0], new_rule[1], triple[2], triple[-1])])) - triple[3]
                time_line[time_line > 0] = 0
                in_score = 0
                if abs(max(time_line)) <= max_span:
                    if dsa == False:
                        return 1
                    if max_1_index == -1:
                        return 1
                    if step == 0 and len(new_rule) == 4:
                        if (new_rule, old_rule) not in proj_rules.keys() and len(proj_rules.keys()) <= 5:
                            proj_rules[(new_rule, old_rule)] = []
                        try:
                            proj_rules[(new_rule, old_rule)].append(abs(max(time_line)))
                        except:
                            pass
                    time_line = triple[3] - np.array(list(self.temporal_model.model.graph.triple_2_t[(triple[0], new_rule[1], triple[2], triple[-1])]))
                    max_1_l = time_line - bin_edges[max_1_index]
                    max_1_r = time_line - bin_edges[max_1_index+1]
                    max_1_mul = max_1_l*max_1_r
                    if (max_1_mul<0).sum() < 0:
                        return 1+hist[max_1_index]
                    else:
                        max_2_l = time_line - bin_edges[max_2_index]
                        max_2_r = time_line - bin_edges[max_2_index+1]
                        max_2_mul = max_2_l*max_2_r
                        if (max_2_mul<0).sum() < 0:
                            return 1+hist[max_2_index]
                        else:
                            return 1+hist[max_3_index]
                else:
                    if new_rule in self.temporal_model.aft_to_pre.keys():
                        num = 0
                        pre_rules = self.temporal_model.aft_to_pre[new_rule][:max_rule]
                        for pre_rule in pre_rules:
                            if len(pre_rule) == 4:
                                if pre_rule in self.temporal_model.nodes:
                                    in_score += detect_one_rule((node, new_rule[1], node_2, triple[3], dir), pre_rule, new_rule, step+1)
                                    num+=1
                            if len(pre_rule) == 2:
                                if pre_rule[0] in self.temporal_model.nodes and pre_rule[1] in self.temporal_model.nodes:
                                    in_score += detect_two_rule((node, new_rule[1], node_2, triple[3], dir), pre_rule[0], pre_rule[1], new_rule, step+1)
                                    num+=1
                        return float(in_score)/(float(num)+1)
                    else:
                        return float(final_score)
            else:
                return 0
            
                
        def detect_two_rule(triple, rule_1, rule_2, old_rule, step):
            #return 0
            final_score = 0
            if step == max_steps:
                return final_score
            node_1, node_2, time = triple[0], triple[2], triple[3]
            et_dict_1, et_dict_2 = dict(), dict()
            if (node_1, rule_1[1]) in self.temporal_model.model.graph.er_2_e_t.keys() and (node_2, rule_2[1]) in self.temporal_model.model.graph.er_2_e_t.keys():
                et_dict_1 = self.temporal_model.model.graph.er_2_e_t[(node_1, rule_1[1])]
                et_dict_2 = self.temporal_model.model.graph.er_2_e_t[(node_2, rule_2[1])]
            else:
                if (node_1, rule_2[1]) in self.temporal_model.model.graph.er_2_e_t.keys() and (node_2, rule_1[1]) in self.temporal_model.model.graph.er_2_e_t.keys():
                    et_dict_1 = self.temporal_model.model.graph.er_2_e_t[(node_1, rule_2[1])]
                    et_dict_2 = self.temporal_model.model.graph.er_2_e_t[(node_2, rule_1[1])]
            if len(et_dict_1) == 0 or len(et_dict_2) == 0:
                # 暂时不做二维规则的递归
                return final_score
            co_e_set = set(list(et_dict_1.keys())) & set(list(et_dict_2.keys()))
            if len(co_e_set) == 0:
                return final_score
            for co_e in co_e_set:
                t_array_1, t_array_2 = np.array(list(et_dict_1[co_e])) - time, np.array(list(et_dict_2[co_e])) - time
                t_array_1[t_array_1 > 0] = 0
                t_array_2[t_array_2 > 0] = 0
                if abs(max(t_array_1)) <= max_span and abs(max(t_array_2)) <= max_span:
                    final_score += 1.0
            return float(final_score)/(len(co_e_set)+1.0)
        
        next_step_rules = set()
        for rule in filtered_rules:
            if rule in self.temporal_model.aft_to_pre.keys():
                pre_rules = self.temporal_model.aft_to_pre[rule][:max_rule]
            else:
                continue
            #print("1.5--->" + str(pre_rules))
            for pre_rule in pre_rules:
                if pre_rule in self.temporal_model.nodes and len(pre_rule) == 4:
                    next_step_rules.add((pre_rule, rule))

        #print("2--->" + str(next_step_rules))
        score = 1
        next_step_rules_list = list(next_step_rules)
        for rule_pair in next_step_rules_list[:max_rule]:
            if rule_pair[0] in self.temporal_model.nodes:
                if rule_pair[1] in self.temporal_model.nodes:
                    score_1 = detect_one_rule((node, pred, node_2, time, dir), rule_pair[0], rule_pair[1], 0) # 查找是否有前置规则
                    score += score_1
        
        return 1.0/float(score), proj_rules
    
    def error_score(self, node, pred, node_2, time, dir, project_rules):
        max_rule = 10000
        filtered_rules = []
        for rule in project_rules:
            if rule in self.temporal_model.nodes:
                filtered_rules.append(rule)

        next_step_facts = set()
        for rule in filtered_rules:
            if rule in self.temporal_model.pre_to_aft.keys():
                aft_rules = self.temporal_model.pre_to_aft[rule][:max_rule]
            else:
                continue
            for aft_rule in aft_rules:
                if aft_rule in self.temporal_model.nodes:
                    next_step_facts.add(((node, aft_rule[1], node_2, dir), rule, aft_rule))
        
        num = 0
        count = 1
        next_step_facts = list(next_step_facts)
        for i in range(len(next_step_facts)):
            fact = next_step_facts[i][0]
            if fact in self.temporal_model.model.graph.triple_2_t.keys():
                time_line = np.array(list(self.temporal_model.model.graph.triple_2_t[fact])) - time
                time_line[time_line > 0] = 0
                count += 1
                if sum(time_line) < 0:
                    num += 1
        
        return float(num) / float(count)
        
    
    def score_edge(self, edge, hop = 2, max_span = 100, and_or = 'and', time_specific = False, blame_edge=True):
        sub, pred, obj, time = int(edge[0]), int(edge[1]), int(edge[2]), int(edge[3])
        score = 0
        if sub not in self.temporal_model.model.graph.node_list or obj not in self.temporal_model.model.graph.node_list:
            return 0

        # get conceptual scores
        if (sub, pred, obj) not in self.temporal_model.model.graph.triple_list: # edges not in the model are not explained
            # number of bits describing the unexplained edge
            score_1, _ = self.conceptual_score(sub, pred, obj, time, 'out', hop, max_span, and_or, time_specific)
            score_2, _ = self.conceptual_score(obj, pred, sub, time, 'in', hop, max_span, and_or, time_specific)

            score = score_1 + score_2
        
        return score

    def score_edge_temporal(self, edge, hop = 2, max_span = 100, and_or = 'and', max_step = 2, aux_score = True, dsa = True, max_rule = 10000, blame_edge=True, file_type='test'):
        sub, pred, obj, time = edge
        score = 0
        proj_dict = dict()
        if sub not in self.temporal_model.model.graph.node_list or obj not in self.temporal_model.model.graph.node_list:
            return score, proj_dict

        # get conceptual scores
        #if (sub, pred, obj) not in self.temporal_model.model.graph.triple_list: # edges not in the model are not explained
            # number of bits describing the unexplained edge
        _, project_1 = self.conceptual_score(sub, pred, obj, time, 'out', hop = hop, max_span = max_span, and_or=and_or)
        _, project_2 = self.conceptual_score(obj, pred, sub, time, 'in', hop = hop, max_span = max_span, and_or=and_or)

        # get causal scores
        if (sub, pred, obj, time) not in self.temporal_model.model.graph.fact_list: # edges not in the model are not explained
            # edges not in the model are not explained
            #print("1----》" + str((sub, pred, obj, time)))
            score_1, proj_dict = self.temporal_score(sub, pred, obj, time, 'out', project_1, max_span = max_span, max_steps = max_step, dsa = dsa, max_rule = max_rule)
            score_2, proj_2 = self.temporal_score(obj, pred, sub, time, 'in', project_2, max_span = max_span, max_steps = max_step, dsa = dsa, max_rule = max_rule)
            score += score_1+score_2
            proj_dict.update(proj_2)
            if aux_score:
                score += self.error_score(sub, pred, obj, time, 'out', project_1)# + self.error_score(obj, pred, sub, time, 'in', project_2)

        return score, proj_dict
    
    def score_edge_missing(self, edge, hop_c = 2, max_span_c = 100, max_span_t = 100, and_or = 'and', max_step = 2, aux_score = True, dsa = True, max_rule = 10000, blame_edge=True, file_type='test'):
        sub, pred, obj, time = edge
        score = 0
        if sub not in self.temporal_model.model.graph.node_list or obj not in self.temporal_model.model.graph.node_list:
            return 0

        # get conceptual scores
        # number of bits describing the unexplained edge
        score_1, project_1 = self.conceptual_score(sub, pred, obj, time, 'out', hop = hop_c, max_span = max_span_c, and_or = and_or)
        score_2, project_2 = self.conceptual_score(obj, pred, sub, time, 'in', hop = hop_c, max_span = max_span_c, and_or = and_or)

        if (sub, pred, obj) not in self.temporal_model.model.graph.triple_list: # edges not in the model are not explained
            score = (score_1 + score_2)/2

        # get causal scores
        #if (sub, pred, obj, time) not in self.temporal_model.model.graph.fact_list: # edges not in the model are not explained
        # edges not in the model are not explaisned
        #print("1----》" + str((sub, pred, obj, time)))
        score_1, _ = self.temporal_score(sub, pred, obj, time, 'out', project_1, max_span = max_span_t, max_steps = max_step, dsa = dsa, max_rule = max_rule)
        score_2, _ = self.temporal_score(obj, pred, sub, time, 'in', project_2, max_span = max_span_t, max_steps = max_step, dsa = dsa, max_rule = max_rule)
        score += (score_1+score_2)/2
        if aux_score:
            score += self.error_score(sub, pred, obj, time, 'out', project_1)# + self.error_score(obj, pred, sub, time, 'in', project_2)

        return score
    
    def update(self, edge, rules, sample_size, span):
        sub, pred, obj, time = edge

        if sub not in self.temporal_model.model.graph.node_list:
            self.updater.new_e_handeler(self.temporal_model, edge, sub, sample_size)
        if obj not in self.temporal_model.model.graph.node_list:
            self.updater.new_e_handeler(self.temporal_model, edge, obj, sample_size)
        
        self.updater.update_model(self.temporal_model, edge, sample_size, span)
        
        for rule_pair in rules.keys():
            self.updater.update_temporal_model(self.temporal_model, rule_pair[0], rule_pair[1], rules[rule_pair], sample_size)
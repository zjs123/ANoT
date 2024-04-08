from collections import defaultdict
from evaluator import Evaluator
from itertools import chain, combinations
import os
import sys
import json
import networkx as nx
from rule import Rule
from correct_assertion import CorrectAssertion
from copy import copy, deepcopy
import _pickle as pickle
import random

class Temporal_Model:
    '''
    A model consisting of rules which explain a knowledge graph.
    '''
    def __init__(self, model):
        '''
        :graph: the knowledge graph being modeled
        '''
        self.model = model
        
        # creat nodes for rule graph
        self.nodes = list(self.model.rules.keys())
        self.id_2_rule_dict = {}
        self.rule_2_id_dict = {}

        # creat empty edge list
        self.edges = dict()

        for i in range(len(self.nodes)):
            self.id_2_rule_dict[i] = self.nodes[i]
            self.rule_2_id_dict[self.nodes[i]] = i

        self.num_nodes = len(self.id_2_rule_dict)
        self.tensor = set() # (s, r, o, t) 具有前驱的事实集合
        self.label_matrix = set() # (triple, rule) 每个三元组和其能映射到的规则（同样只计算具有前驱的事实）

        self.aft_to_pre = dict()
        self.pre_to_aft = dict()
        self.rule_to_time = dict()
        self.pair_to_pre_rule = dict()

    def iscomplete(self):
        print([len(self.tensor_no_time), len(self.model.graph.triple_list)])
        if len(self.tensor_no_time & self.model.graph.triple_list) != len(self.model.graph.triple_list):
            return True
        return False
    
    def post_process(self):
        for rule in self.pre_to_aft.keys():
            if len(rule) == 4:
                pair = (rule[0], rule[1])
                if pair not in self.pair_to_pre_rule.keys():
                    self.pair_to_pre_rule[pair] = set()
                self.pair_to_pre_rule[pair].add(rule)
            rule_list = self.pre_to_aft[rule]
            rule_list = sorted(rule_list, reverse=True, key= lambda g: (len(g), len(self.model.graph.candidates[g]['ca_to_size']) if len(g) == 4 else len(self.model.graph.candidates[g[0]]['ca_to_size'])))
            self.pre_to_aft[rule] = rule_list
        
        for rule in self.aft_to_pre.keys():
            rule_list = self.aft_to_pre[rule]
            rule_list = sorted(rule_list, reverse=True, key= lambda g: (len(g), len(self.model.graph.candidates[g]['ca_to_size']) if len(g) == 4 else len(self.model.graph.candidates[g[0]]['ca_to_size'])))
            self.aft_to_pre[rule] = rule_list
        
        for pair in self.pair_to_pre_rule.keys():
            rule_list = list(self.pair_to_pre_rule[pair])
            rule_list = sorted(rule_list, reverse=True, key= lambda g: (len(g), len(self.model.graph.candidates[g]['ca_to_size'])))
            self.pair_to_pre_rule[pair] = rule_list

    
    def add_edge(self, edge, input_dict, time_dict, type = 'test'):
        if edge in self.edges:
            print('Already added')
            return

        self.edges[edge] = list(input_dict[edge])

        aft_rule = edge[-1]
        if type == 'final':
            #self.model.add_rule(aft_rule)
            if aft_rule not in self.nodes:
                self.nodes.append(edge[-1])
                self.id_2_rule_dict[len(self.nodes)] = aft_rule
                self.rule_2_id_dict[aft_rule] = len(self.nodes)
        if len(edge) == 2:
            pre_rule = edge[0]
            if type == 'final':
                #self.model.add_rule(pre_rule)
                if pre_rule not in self.nodes:
                    self.nodes.append(edge[0])
                    self.id_2_rule_dict[len(self.nodes)] = pre_rule
                    self.rule_2_id_dict[pre_rule] = len(self.nodes)
                if (pre_rule, aft_rule) not in self.rule_to_time.keys():
                    self.rule_to_time[(pre_rule, aft_rule)] = time_dict[edge]
        if len(edge) == 3:
            pre_rule = (edge[0], edge[1])
            if type == 'final':
                if edge[0] not in self.nodes:
                    self.nodes.append(edge[0])
                    self.id_2_rule_dict[len(self.nodes)] = edge[0]
                    self.rule_2_id_dict[edge[0]] = len(self.nodes)
                if edge[1] not in self.nodes:
                    self.nodes.append(edge[1])
                    self.id_2_rule_dict[len(self.nodes)] = edge[1]
                    self.rule_2_id_dict[edge[1]] = len(self.nodes)
                if (pre_rule, aft_rule) not in self.rule_to_time.keys():
                    self.rule_to_time[(pre_rule, aft_rule)] = time_dict[edge]
        
        if aft_rule not in self.aft_to_pre.keys():
            self.aft_to_pre[aft_rule] = set()
        self.aft_to_pre[aft_rule].add(pre_rule)

        if pre_rule not in self.pre_to_aft.keys():
            self.pre_to_aft[pre_rule] = set()
        self.pre_to_aft[pre_rule].add(aft_rule)

        self.make_assertions(edge, input_dict)

    def remove_edge(self, edge, input_dict, time_dict):
        if edge != self.cache['last_updated_edge']:
            print('We can only remove the last added rule.')
            return
        if edge not in self.edges: # make sure the rule is actually there
            return
        # remove rule
        del self.edges[edge]
        if len(edge) == 2:
            self.aft_to_pre[edge[-1]].remove(edge[0])
        if len(edge) == 3:
            self.aft_to_pre[edge[-1]].remove((edge[0], edge[1]))
        if len(self.aft_to_pre[edge[-1]]) == 0:
            del self.aft_to_pre[edge[-1]]
        
        self.undo_assertions(edge, input_dict)

    def make_assertions(self, edge, input_dict):
        '''
        Fills in model's tensor and node label map with assertions of a rule

        :rule: a rule
        '''
        # reset cache
        self.cache = {'last_updated_edge': edge}

        # update cache
        self.cache['new_triples'] = input_dict[edge].difference(self.tensor)
        rule_triple_combine = set([(edge[-1], triple) for triple in input_dict[edge]])
        self.cache['new_rules'] = rule_triple_combine.difference(self.label_matrix)

        self.tensor.update(self.cache['new_triples'])
        self.label_matrix.update(self.cache['new_rules'])

    def undo_assertions(self, egde, input_dict):
        '''
        Removes things from model's tensor and node label map

        :rule: a rule
        '''
        self.tensor.difference_update(self.cache['new_triples'])
        self.label_matrix.difference_update(self.cache['new_rules'])

    def print_stats(self, temporal_model):
        evaluator = Evaluator(self.model.graph)
        val = evaluator.evaluate_temporal(temporal_model)
        print('----- Model stats -----')
        print('L(G,M) = {}'.format(round(val, 2)))
        null_val = evaluator.evaluate_temporal(Temporal_Model(self.model))
        print([val, null_val])
        print('% Bits needed: {}'.format(round((val / null_val) * 100, 2)))
        print('# Rules: {}'.format(len(self.edges)))
        print('% Edges Explained: {}'.format(round(len(self.tensor & self.model.graph.fact_list) / len(self.model.graph.fact_list) * 100, 2)))
        print('-----------------------')
    
    def update(self, rule_pair, span):
        if rule_pair in self.rule_to_time.keys():
            self.rule_to_time[rule_pair] += span
    
    def add_new_rules(self, new_rules, span):
        for new_rule in new_rules:
            if new_rule in self.aft_to_pre.keys():
                continue
            else:
                if (new_rule[0], new_rule[2]) in self.pair_to_pre_rule.keys():
                    pre_rules = self.pair_to_pre_rule[(new_rule[0], new_rule[2])]
                    self.aft_to_pre[new_rule] = pre_rules
                    for pre_rule in pre_rules:
                        if new_rule not in self.pre_to_aft[pre_rule]:
                            self.pre_to_aft[pre_rule].append(new_rule)
                        if (pre_rule, new_rule) not in self.rule_to_time.keys():
                            self.rule_to_time[(pre_rule, new_rule)] = []
                        self.rule_to_time[(pre_rule, new_rule)].append(span)


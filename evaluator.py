from math import log2 as log
from scipy.special import comb
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from scipy.special import gammaln
from rule import Rule

class Evaluator:
    '''
    Evaluates a model or a rule.
    '''
    def __init__(self, graph):
        self.graph = graph
        self.log_V = log(self.graph.n)
        self.log_E = log(self.graph.m)
        self.log_E_plus_1 = log(self.graph.m + 1)
        self.log_labels_plus_1 = log(self.graph.total_num_labels + 1)
        self.log_LV = log(self.graph.num_node_labels)
        # a cache for L_N since this can be slow to do many times
        self.length_natural_number_map = dict()
        # a cache for log of binomials
        self.binomial_map = dict()
        self.rule_to_length = dict()
        self.edge_to_length = dict()

    def evaluate(self, model, with_lengths=False):
        '''
        L(M) + L(G|M) - (Section 3.2)
        '''
        length_model = self.length_model_no_time(model)
        length_error, neg_edge, neg_node = self.length_graph_with_model_no_time(model)
        val = length_model + length_error
        if with_lengths:
            return val, length_model, neg_edge, neg_node
        return val
    
    def evaluate_temporal(self, temporal_model, with_lengths=False):
        length_model = self.length_model_time(temporal_model)
        length_error, neg_edge, neg_node = self.length_graph_with_model_time(temporal_model)
        val = length_model + length_error
        if with_lengths:
            return val, length_model, neg_edge, neg_node
        return val

    def evaluate_change(self, model, rule, prev_model_len):
        '''
        L(M union {r}) + L(G|M union {r})

        :return: the MDL objective after inserting a new rule to M
        '''
        # update score
        neg_edge = self.length_negative_edge_error_no_time(model)
        neg_node = self.length_negative_label_error_no_time(model)
        length_error_with_rule = neg_edge + neg_node
        length_model_with_rule = self.length_model_new_rule(model, rule, prev_model_len)
        val = length_model_with_rule + length_error_with_rule

        return val, length_model_with_rule, neg_edge, neg_node
    
    def evaluate_change_temporal(self, temporal_model, edge, prev_model_len):
         # update score
        neg_edge = self.length_negative_edge_error_time(temporal_model)
        neg_node = self.length_negative_label_error_time(temporal_model)
        length_error_with_rule = neg_edge + neg_node
        length_model_with_rule = self.length_model_new_rule_time(temporal_model, edge, prev_model_len)
        val = length_model_with_rule + length_error_with_rule

        return val, length_model_with_rule, neg_edge, neg_node

    def length_model_new_rule(self, model, rule, length_model):
        '''
        L(M union {r})

        :return: the length of the model with a new rule added
        '''
        # old length
        length = length_model
        length += self.length_rule(rule) + self.length_rule_assertions(rule, model)

        return length
    
    def length_model_new_rule_time(self, temporal_model, edge, length_model):
        # old length
        length = length_model
        length += self.length_combined_rule(edge, temporal_model) + self.length_combined_rule_assertions(edge, temporal_model)

        if self.length_combined_rule(edge, temporal_model) < 0:
            print('error_1')
        if self.length_combined_rule_assertions(edge, temporal_model) < 0:
            print('error_2')

        return length

    def length_natural_number(self, n):
        '''
        :n: a number

        :return: the number of bits required to transmit the number
        '''
        if n <= 0:
            return 0
        if n in self.length_natural_number_map:
            return self.length_natural_number_map[n]
        c = log(2.865064)
        i = log(n)
        while i > 0:
            c = c + i
            i = log(i)
        self.length_natural_number_map[n] = c # cache the value
        return c

    def length_binomial(self, n, k):
        '''
        :n: n in (n choose k)
        :k: k in (n choose k)

        :return: log_2(n choose k)
        '''
        # we cache computations that we've already made for speedups
        if (n, k) in self.binomial_map:
            return self.binomial_map[(n, k)]

        # gammln computation of log_e(n choose k) with change of base to log_2
        length = (gammaln(n + 1) - gammaln(k + 1) - gammaln((n + 1) - k)) / np.log(2)
        # cache the result for future use
        self.binomial_map[(n, k)] = length
        return length

    def length_model_no_time(self, model):
        rules = model.rules.keys()
        # num rules
        length = log(2 * self.graph.num_node_labels * self.graph.num_edge_labels * self.graph.num_node_labels + 1)
        # rules
        length += sum(self.length_rule(rule) + self.length_rule_assertions(rule, model) for rule in rules)

        return length
    
    def length_model_time(self, temporal_model):
        nodes = temporal_model.rule_2_id_dict.keys()
        edges = temporal_model.edges.keys()
        # num combined rules
        length = log(2 * len(nodes) * len(nodes) + 1) + log(len(nodes) * len(nodes) * len(nodes) + 1)
        # rules
        length += sum(self.length_combined_rule(edge, temporal_model) + self.length_combined_rule_assertions(edge, temporal_model) for edge in edges)

        return length

    def length_rule(self, rule):
        '''
        L(g) - (Section 3.2.1)

        :rule: (parent, children)
            - recusrive case
                - parent: a set of node labels
                - children: a set of elements like (edge_type, dir, rule)
            - base case (parent is a leaf)
                - parent: a set of node labels
                - children: an empty set

        :return: the number of bits required to transmit a rule
        '''
        sub, pred, obj, dr = rule
        
        length = self.log_LV
        length += sum(-log(self.graph.node_label_counts[label] / self.graph.n) for label in [sub, obj])
        length += -log(self.graph.edge_label_counts[pred] / len(self.graph.triple_list))
        
        return length
    
    def length_combined_rule(self, edge, temporal_model):
        length = 0 # should be log(len(rules)), but simplified as 0 here.

        if len(edge) == 2:
            pre_rule, aft_rule = edge
            length += sum(-log(len(self.graph.candidates[rule]['facts']) / len(self.graph.fact_list)) for rule in [pre_rule, aft_rule])
        else:
            pre_rule, aux_rule, aft_rule = edge
            length += sum(-log(len(self.graph.candidates[rule]['facts']) / len(self.graph.fact_list)) for rule in [pre_rule, aux_rule, aft_rule])
        
        return length

    
    def length_rule_assertions(self, rule, model, correct_assertions=None, info=False):
        '''
        L(alpha(g)) - (Section 3.2.1)
        '''

        if rule in self.rule_to_length:
            return self.rule_to_length[rule]
        
        # label correct assertions
        correct_assertions = model.rules[rule] if model else correct_assertions # 该规则可覆盖的知识对应的头实体
        # num assertions
        num_assertions = self.graph.nodes_with_type(rule[0]) # 包含该规则头类别的实体
        # num exceptions
        num_correct = len(correct_assertions)
        num_exceptions = num_assertions - num_correct # 有该规则的头类别，但没有包含该规则实例的实体
        length = log(num_assertions)
        # exception ids
        length += self.length_binomial(num_assertions, num_exceptions)
        
        return length
    
    
    def length_combined_rule_assertions(self, edge, temporal_model, correct_assertions=None, info=False):
        '''
        L(alpha(g)) - (Section 3.2.1)
        '''

        if edge in self.edge_to_length:
            return self.edge_to_length[edge]
        
        # label correct assertions
        correct_assertions = temporal_model.edges[edge] if temporal_model else correct_assertions # 该规则组合可为多少知识提供前驱
        # num assertions
        num_assertions = len(self.graph.candidates[edge[-1]]['facts']) # 可以映射为末尾规则的事实个数
        # num exceptions
        num_correct = len(correct_assertions)
        num_exceptions = num_assertions - num_correct # 有该规则的头类别，但没有包含该规则实例的实体
        length = log(num_assertions)
        # exception ids
        length += self.length_binomial(num_assertions, num_exceptions)

        if num_assertions < 0:
            print('in_error_1')
        if num_exceptions < 0:
            print('in_error_2')
        
        return length

    
    def length_graph_with_model_no_time(self, model):
        '''
        L(G|M) - (Section 3.2.2)
        '''
        negative_edge_error = self.length_negative_edge_error_no_time(model)
        negative_node_error = self.length_negative_label_error_no_time(model)
        length = negative_edge_error + negative_node_error
        return length, negative_edge_error, negative_node_error
    
    
    def length_graph_with_model_time(self, temporal_model):
        
        #L(G|M) - (Section 3.2.2)
        
        negative_edge_error = self.length_negative_edge_error_time(temporal_model)
        negative_node_error = self.length_negative_label_error_time(temporal_model)
        length = negative_edge_error + negative_node_error
        return length, negative_edge_error, negative_node_error
    

    def length_negative_edge_error_no_time(self, model):
        '''
        L(A-) - (Section 3.2.2)
        '''
        # number of ones modeled
        num_modeled = len(model.tensor_no_time)
        # num ones
        num_unexplained = len(self.graph.triple_list) - num_modeled
        # ones
        if num_unexplained >= 0:
            length = self.length_binomial((self.graph.n ** 2) * self.graph.num_edge_labels - num_modeled, num_unexplained)
        else:
            length = 0
        return length
    
    def length_negative_edge_error_time(self, temporal_model):
        '''
        L(A-) - (Section 3.2.2)
        '''
        # number of ones modeled
        num_modeled = len(temporal_model.tensor)
        # num ones
        num_unexplained = self.graph.fact_num - num_modeled
        # ones
        if num_unexplained >= 0:
            length = self.length_binomial((self.graph.n ** 2) * self.graph.num_edge_labels * len(self.graph.time_list) - num_modeled, num_unexplained)
        else:
            length = 0
        return length

    def length_negative_label_error_no_time(self, model):
        '''
        L(L-) - (Section 3.2.2)
        '''
        # number of ones modeled
        num_modeled = len(model.label_matrix)
        # num ones
        num_unexplained = self.graph.total_num_labels - num_modeled
        # ones
        if num_unexplained >= 0:
            length = self.length_binomial(self.graph.num_node_labels * self.graph.n - num_modeled, num_unexplained)
        else:
            length = 0
        return length
    
    def length_negative_label_error_time(self, temporal_model):
        '''
        L(L-) - (Section 3.2.2)
        '''
        # number of ones modeled
        num_modeled = len(temporal_model.label_matrix)
        # num ones
        num_unexplained = self.graph.total_num_triples - num_modeled
        # ones
        if num_unexplained >= 0:
            length = self.length_binomial(temporal_model.num_nodes * len(self.graph.triple_list) - num_modeled, num_unexplained)
        else:
            length = 0
        
        return length
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
import itertools
import random

class Model:
    '''
    A model consisting of rules which explain a knowledge graph.
    '''
    def __init__(self, graph):
        '''
        :graph: the knowledge graph being modeled
        '''
        self.graph = graph
        # rules stored mapping to matches
        self.rules = dict()
        self.node_label_counts = defaultdict(int)
        self.edge_label_counts = defaultdict(int)
        self.tensor_no_time = set()
        self.tensor = set()
        self.label_matrix = set()
        self.cache = {'last_updated_rule': None}
        self.rule_graph = None
        self.shared_root_rule_dependency_graph = None
        self.subject_to_rules = dict()
        self.object_to_rules = dict()
        self.pred_to_rules = dict()

        self.subject_to_triples = dict()
        self.object_to_triples = dict()
        self.pred_to_triples = dict()

        self.new_candidate_rules = dict()
        self.min_ca_to_size = 1000000

    
    def post_process(self):
        for rule in self.rules.keys():
            ca_to_size = len(self.graph.candidates[rule]['ca_to_size'])
            if ca_to_size < self.min_ca_to_size:
                self.min_ca_to_size = ca_to_size
    
    def iscomplete(self):
        print([len(self.tensor_no_time), len(self.graph.triple_list)])
        if len(self.tensor_no_time & self.graph.triple_list) != len(self.graph.triple_list):
            return True
        return False
    
    def add_rule(self, rule):
        '''
        :rule: the rule to be added.
        '''
        # store rule
        if rule in self.rules:
            print('Already added')
            return

        self.rules[rule] = list(self.graph.candidates[rule]['ca_to_size'].values())
        
        if rule[0] not in self.subject_to_rules:
            self.subject_to_rules[rule[0]] = set()
        self.subject_to_rules[rule[0]].add(rule)

        if rule[1] not in self.pred_to_rules:
            self.pred_to_rules[rule[1]] = set()
        self.pred_to_rules[rule[1]].add(rule)

        if rule[2] not in self.object_to_rules:
            self.object_to_rules[rule[2]] = set()
        self.object_to_rules[rule[2]].add(rule)
        
        self.make_assertions(rule)

    def remove_rule(self, rule):
        '''
        :rule: the rule to be removed
        '''
        if rule != self.cache['last_updated_rule']:
            print('We can only remove the last added rule.')
            return
        if rule not in self.rules: # make sure the rule is actually there
            return
        # remove rule
        del self.rules[rule]
        self.subject_to_rules[rule[0]].remove(rule)
        if len(self.subject_to_rules[rule[0]]) == 0:
            del self.subject_to_rules[rule[0]]
        
        self.object_to_rules[rule[2]].remove(rule)
        if len(self.object_to_rules[rule[2]]) == 0:
            del self.object_to_rules[rule[2]]
        
        self.pred_to_rules[rule[1]].remove(rule)
        if len(self.pred_to_rules[rule[1]]) == 0:
            del self.pred_to_rules[rule[1]]

        self.undo_assertions(rule)

    def make_assertions(self, rule):
        '''
        Fills in model's tensor and node label map with assertions of a rule

        :rule: a rule
        '''
        # reset cache
        self.cache = {'last_updated_rule': rule}

        # update cache
        self.cache['new_edges'] = self.graph.candidates[rule]['edges'].difference(self.tensor)
        self.cache['new_triples'] = self.graph.candidates[rule]['triples'].difference(self.tensor_no_time)
        self.cache['new_labels'] = self.graph.candidates[rule]['label_coverage'].difference(self.label_matrix)

        self.tensor.update(self.cache['new_edges'])
        self.tensor_no_time.update(self.cache['new_triples'])
        self.label_matrix.update(self.cache['new_labels'])

    def undo_assertions(self, rule):
        '''
        Removes things from model's tensor and node label map

        :rule: a rule
        '''
        self.tensor.difference_update(self.cache['new_edges'])
        self.tensor_no_time.difference_update(self.cache['new_triples'])
        self.label_matrix.difference_update(self.cache['new_labels'])

    def build_rule_graph(self, build_prime=False, force=False):
        '''
        Builds a dependency graph.
            - Nodes are rules
            - (r1, r2) means that r1's tail is r2's head and denotes a possible composition

        :return: a dependency graph of rules connected by their possible compositions
        '''
        if not force and self.rule_graph:
            return

        root_to_rule = defaultdict(list)
        # find dependencies and store as edges for a dependency graph
        edges = list()
        tree_rules = list()
        for rule in self.rules:
            if type(rule) is tuple:
                parent, children = rule
                rule = Rule(parent, children)
            self.plant_forest(rule)
            root_to_rule[rule.root].append(rule)
            tree_rules.append(rule)

        if not build_prime:
            # build graph
            for rule in tree_rules:
                # if leaf matches other rules' roots
                if len(set(rule.get_leaves()).intersection(set(root_to_rule.keys()))) > 0:
                    # get the matching rules
                    matching_rules = set()
                    for leaf in rule.get_leaves():
                        if leaf in root_to_rule:
                            matching_rules.update(root_to_rule[leaf])
                    for other_rule in matching_rules:
                        if rule.root in other_rule.get_leaves(): # don't allow loops
                            continue
                        edges.append((rule, other_rule))

            self.rule_graph = nx.DiGraph(edges)

        if build_prime:
            def jaccard_sim(r1, r2):
                a = set(real.root for real in r1.correct_assertions)
                b = set(real.root for real in r2.correct_assertions)
                return len(a.intersection(b)) / len(a.union(b)) if len(a.union(b)) > 0 else 0
            # build rule graph prime, which encodes dependencies of shared root types
            edges = list()
            for rule in tree_rules:
                for other_rule in root_to_rule[rule.root]:
                    if rule != other_rule: #(other_rule, rule) not in edges:
                        self.plant_forest(rule)
                        self.plant_forest(other_rule)
                        if jaccard_sim(rule, other_rule) == 1.0:
                            edges.append((rule, other_rule))
            self.shared_root_rule_dependency_graph = nx.Graph(edges)

    def pickle_copy(self, obj):
        name = '{}'.format(random.randint(0, 10000000))
        with open('{}.pickle'.format(name), 'wb') as handle:
            pickle.dump(obj, handle)

        with open('{}.pickle'.format(name), 'rb') as handle:
            new_obj = pickle.load(handle)

        os.remove('{}.pickle'.format(name))

        return new_obj

    def save(self, fname):
        '''
        Saves the model in json format.

        :path: path to a json file where model should be saved
        '''
        rules = list()
        for rule in self.rules:
            if type(rule) is tuple:
                parent, children = rule
                rule = Rule(parent, children)
            self.plant_forest(rule)
            rules.append(rule)

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        pickle_outfile = os.path.join(ROOT_DIR, fname + '.pickle')
        with open(pickle_outfile, 'wb') as handle:
            pickle.dump(self, handle)

        rule_outfile = os.path.join(ROOT_DIR, fname + '.rules')
        with open(rule_outfile, 'w') as f:
            for rule in rules:
                if self.graph.idify:
                    f.write('{}\n'.format(rule.tuplify(id_to_node=self.graph.id_to_node, id_to_pred=self.graph.id_to_pred)))
                else:
                    f.write('{}\n'.format(rule.tuplify()))

    def percent_improved(self):
        evaluator = Evaluator(self.graph)
        null_val = evaluator.evaluate(Model(self.graph))
        val = evaluator.evaluate(self)
        # the difference is what percent of the original?
        return ((null_val - val) / null_val) * 100

    def print_stats(self):
        evaluator = Evaluator(self.graph)
        val = evaluator.evaluate(self)
        print('----- Model stats -----')
        print('L(G,M) = {}'.format(round(val, 2)))
        null_val = evaluator.evaluate(Model(self.graph))
        print('% Bits needed: {}'.format(round((val / null_val) * 100, 2)))
        print('# Rules: {}'.format(len(self.rules)))
        print('% Edges Explained: {}'.format(round(len(self.tensor_no_time) / len(self.graph.triple_list) * 100, 2)))
        print('-----------------------')
    
    
    def update(self, fact, sample_size):
        new_rules = []
        all_candidate_rules = []
        sub, pred, obj, time = fact[0], fact[1], fact[2], fact[3]
        sub_labels = self.graph.node_to_labels[sub]
        obj_labels = self.graph.node_to_labels[obj]
        if (sub, pred, obj) in self.graph.triple_list:
            return new_rules
        sls_ols = list(itertools.product(sub_labels, obj_labels))
        for sl, ol in sls_ols:
            if sl == ol:
                continue
            candidate_rule_1 = (sl, pred, ol, 'out')
            candidate_rule_2 = (ol, pred, sl, 'in')
            if candidate_rule_1 in self.graph.candidates.keys():
                all_candidate_rules.append(candidate_rule_1)
            if candidate_rule_2 in self.graph.candidates.keys():
                all_candidate_rules.append(candidate_rule_2)
        #sorted_candidate_rules = sorted(all_candidate_rules, reverse=True, key= lambda g: len(self.graph.candidates[g]['ca_to_size']))
        sorted_candidate_rules = sorted(all_candidate_rules, reverse=True, key= lambda g: len(self.graph.candidates[g]['triples']))
        for rule in sorted_candidate_rules[:sample_size]:
            if rule not in self.rules.keys():
                self.new_candidate_rules[rule] = len(self.graph.candidates[rule]['ca_to_size'])+1
                self.add_rule(rule)
                new_rules.append(rule)

        return new_rules
    
    '''
    def update(self, fact, sample_size):
        new_rules = []
        sub, pred, obj, time = fact[0], fact[1], fact[2], fact[3]
        sub_labels = self.graph.node_to_labels[sub]
        obj_labels = self.graph.node_to_labels[obj]
        if pred in self.graph.e2r[sub] and pred in self.graph.e2r[obj]:
            sls_ols = list(itertools.product(sub_labels[:5], obj_labels[:5]))
            for sl, ol in sls_ols:
                if sl == ol:
                    continue
                candidate_rule_1 = (sl, pred, ol, 'out')
                candidate_rule_2 = (ol, pred, sl, 'in')
                if candidate_rule_1 not in self.rules.keys() and candidate_rule_1 in self.graph.candidates.keys():
                    if candidate_rule_1 not in self.new_candidate_rules.keys():
                        self.new_candidate_rules[candidate_rule_1] = len(self.graph.candidates[candidate_rule_1]['ca_to_size'])+1
                    if self.new_candidate_rules[candidate_rule_1] > self.min_ca_to_size:
                        self.add_rule(candidate_rule_1)
                        new_rules.append(candidate_rule_1)
                if candidate_rule_2 not in self.rules.keys() and candidate_rule_2 in self.graph.candidates.keys():
                    if candidate_rule_2 not in self.new_candidate_rules.keys():
                        self.new_candidate_rules[candidate_rule_2] = len(self.graph.candidates[candidate_rule_2]['ca_to_size'])+1
                    if self.new_candidate_rules[candidate_rule_2] > self.min_ca_to_size:
                        self.add_rule(candidate_rule_2)
                        new_rules.append(candidate_rule_2)
        return new_rules
        '''

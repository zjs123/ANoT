#typelist = open(os.path.join(ROOT_DIR, '../data/{}/types_bad.{}'.format(name, ext)), 'r')
#label_map = open(os.path.join(ROOT_DIR, '../data/{}/e2types_bad.{}'.format(name, ext)), 'r')

from tqdm import tqdm
import numpy as np
from collections import defaultdict
import os
from itertools import chain, combinations
from scipy.sparse import lil_matrix
import itertools
import sys
import networkx as nx
import _pickle as pickle

class Graph:
    '''
    A Temporal Knowledge Graph representation which seeks to make the operations needed
    for this method as efficient as possible.
    '''
    def __init__(self, name, ext='txt', delimiter='	', idify=False, verbose=True, load_candidates_from_disk=False):
        '''
        :name: the name of the graph to load.
        :ext: the file extension for the graph files.
        :delimiter: the delimiter for the graph files.
        :idify: if true, converts strings to ids
        :verbose: whether or not to print simple stats (e.g., num nodes)

        Assumes:
        1) ../data/name/train.txt edgelist.
        2) ../data/name/e2types.txt node label mapping.
        '''
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        edgelist = open(os.path.join(ROOT_DIR, '../data/{}/train.{}'.format(name, ext)), 'r')
        typelist = open(os.path.join(ROOT_DIR, '../data/{}/types_bad.{}'.format(name, ext)), 'r')
        label_map = open(os.path.join(ROOT_DIR, '../data/{}/e2types_bad.{}'.format(name, ext)), 'r')
        self.name = name
        self.node_list = set()
        self.time_list = set()
        self.triple_list = set()
        self.fact_list = set()
        self.new_e_set = set()
        self.labels_to_ids = dict()
        self.label_r_to_ids = dict()
        self.node_to_labels = dict()
        self.t_2_triple = dict()
        self.triple_2_t = dict()
        self.e_pair_to_rtd = dict() # (e_1, e_2): (relation, time, dir)
        self.t_e_2_triple = dict() # time:[e1: (triple1, triple2, ...), e2:...]
        self.t_pair_2_triple = dict()
        self.e_2_t_list = dict()
        self.er_2_e_t = dict()
        self.e2r = dict()
        self.label_to_rule = defaultdict(set) # 只在最初选规则时用到,不用更新
        self.triple_to_rule = defaultdict(list) # 只在最初选规则时用到,不用更新
        self.label_pair_to_rule = defaultdict(set)
        self.node_label_counts = defaultdict(int)
        self.edge_label_counts = defaultdict(int)
        self.idify = idify
        # load the graph
        self.graph = nx.Graph()
        self.t_2_graph = dict()
        self.load(typelist, edgelist, label_map, delimiter, verbose, load_candidates_from_disk)
        self.num_node_labels = len(self.label_matrix)
        if verbose:
            print('|V| = {}'.format(self.n))
            print('|E| = {}'.format(self.m))
            print('|L_V| = {}'.format(self.num_node_labels))
            print('|L_E| = {}'.format(self.num_edge_labels))
            print('|Rule| = {}'.format(len(self.candidates.keys())))

    def load(self, typelist, edgelist, label_map, delimiter, verbose, load_candidates_from_disk):
        '''
        Loads a knowledge graph.

        :edgelist: a path to the graph edgelist.
        :label_map: a path to the node label mapping.
        :delimiter: the delimiter used in the above files.
        '''
        
        # read label list
        for line in typelist:
            line = line.strip().split(delimiter)
            label_description = eval(line[0])
            id = int(line[1])
            self.labels_to_ids[label_description] = id
            for r in label_description:
                if r not in self.label_r_to_ids.keys():
                    self.label_r_to_ids[r] = set()
                self.label_r_to_ids[r].add(id)
        self.ids_to_labels = dict(zip(self.labels_to_ids.values(), self.labels_to_ids.keys()))
        # read node label mapping
        self.label_matrix = dict()
        for line in label_map:
            line = line.strip().split(delimiter)
            node = int(line[0])
            self.node_list.add(node)
            labels = eval(line[1])
            self.node_to_labels[node] = labels
            for label in labels:
                self.node_label_counts[label] += 1
                if label not in self.label_matrix:
                    self.label_matrix[label] = set()
                self.label_matrix[label].add(node)
        
        for key in self.label_r_to_ids.keys():
            tmp_list = list(self.label_r_to_ids[key])
            tmp_list_sort = sorted(tmp_list, reverse=True, key=lambda g: len(self.label_matrix[g]))
            self.label_r_to_ids[key] = tmp_list_sort

        # candidates map to matches
        self.candidates = dict()
        self.m = 0
        self.id_to_edge = dict()
        # read edgelist
        edge_labels = set()
        tmp_e_2_t = dict()
        self.tensor = set()
        for line in tqdm(edgelist):
            sub, pred, obj, time = line.strip().split(delimiter)[:4]
            sub, pred, obj, time = int(sub), int(pred), int(obj), int(time)
            edge_labels.add(pred)
            self.triple_list.add((sub, pred, obj))
            self.fact_list.add((sub, pred, obj, time))
            self.graph.add_edge(sub, obj)
            if time not in self.t_2_graph.keys():
                self.t_2_graph[time] = nx.Graph()
            self.t_2_graph[time].add_edge(sub, obj)
            if sub not in self.e2r.keys():
                self.e2r[sub] = set()
            self.e2r[sub].add(pred)
            if obj not in self.e2r.keys():
                self.e2r[obj] = set()
            self.e2r[obj].add(pred)

            if (sub, pred) not in self.er_2_e_t.keys():
                self.er_2_e_t[(sub, pred)] = dict()
                self.er_2_e_t[(sub, pred)][obj] = set()
            else:
                if obj not in self.er_2_e_t[(sub, pred)].keys():
                    self.er_2_e_t[(sub, pred)][obj] = set()
            self.er_2_e_t[(sub, pred)][obj].add(time)

            if (obj, pred) not in self.er_2_e_t.keys():
                self.er_2_e_t[(obj, pred)] = dict()
                self.er_2_e_t[(obj, pred)][sub] = set()
            else:
                if sub not in self.er_2_e_t[(obj, pred)].keys():
                    self.er_2_e_t[(obj, pred)][sub] = set()
            self.er_2_e_t[(obj, pred)][sub].add(time)

            if sub not in tmp_e_2_t.keys():
                tmp_e_2_t[sub] = []
            if time not in tmp_e_2_t[sub]:
                tmp_e_2_t[sub].append(time)

            if obj not in tmp_e_2_t.keys():
                tmp_e_2_t[obj] = []
            if time not in tmp_e_2_t[obj]:
                tmp_e_2_t[obj].append(time)

            if time not in self.t_2_triple.keys():
                self.t_2_triple[time] = set()
            self.t_2_triple[time].add((sub, pred, obj, 'out'))
            self.t_2_triple[time].add((obj, pred, sub, 'in'))

            if (sub, pred, obj, 'out') not in self.triple_2_t.keys():
                self.triple_2_t[(sub, pred, obj, 'out')] = set()
            self.triple_2_t[(sub, pred, obj, 'out')].add(time)

            if (obj, pred, sub, 'in') not in self.triple_2_t.keys():
                self.triple_2_t[(obj, pred, sub, 'in')] = set()
            self.triple_2_t[(obj, pred, sub, 'in')].add(time)

            if (sub, obj) not in self.e_pair_to_rtd.keys():
                self.e_pair_to_rtd[(sub, obj)] = []
            if (pred, time, 'out') not in self.e_pair_to_rtd[(sub, obj)]:
                self.e_pair_to_rtd[(sub, obj)].append((pred, time, 'out'))

            if (obj, sub) not in self.e_pair_to_rtd.keys():
                self.e_pair_to_rtd[(obj, sub)] = []
            if (pred, time, 'in') not in self.e_pair_to_rtd[(obj, sub)]:
                self.e_pair_to_rtd[(obj, sub)].append((pred, time, 'in'))

            if time not in self.t_e_2_triple.keys():
                self.t_e_2_triple[time] = dict()
            
            if sub not in self.t_e_2_triple[time].keys():
                self.t_e_2_triple[time][sub] = set()
            self.t_e_2_triple[time][sub].add((sub, pred, obj, 'out'))

            if obj not in self.t_e_2_triple[time].keys():
                self.t_e_2_triple[time][obj] = set()
            self.t_e_2_triple[time][obj].add((obj, pred, sub, 'in'))

            if time not in self.t_pair_2_triple.keys():
                self.t_pair_2_triple[time] = dict()

            if (sub, obj) not in self.t_pair_2_triple[time].keys():
                self.t_pair_2_triple[time][(sub, obj)] = set()
            self.t_pair_2_triple[time][(sub, obj)].add((sub, pred, obj, 'out'))

            if (obj, sub) not in self.t_pair_2_triple[time].keys():
                self.t_pair_2_triple[time][(obj, sub)] = set()
            self.t_pair_2_triple[time][(obj, sub)].add((obj, pred, sub, 'in'))
            
            self.edge_label_counts[pred] += 1
            self.node_list.add(sub)
            self.node_list.add(obj)
            self.time_list.add(time)
            self.tensor.add(self.m)
            if not load_candidates_from_disk:
                # generate rule candidates

                # sub labels
                sub_labels = self.labels(sub)
                # obj labels
                obj_labels = self.labels(obj)
                sls_ols = list(itertools.product(sub_labels, obj_labels))
                for sl, ol in sls_ols:
                    if sl == ol:
                        continue
                    if (sl, pred, ol, 'out') not in self.candidates:
                        self.candidates[(sl, pred, ol, 'out')] = {'label_coverage': set(),
                                                                    'edges': set(),
                                                                    'triples':set(),
                                                                    'facts':set(),
                                                                    'ca_to_size': defaultdict(int)}
                    self.candidates[(sl, pred, ol, 'out')]['label_coverage'].add((ol, obj))
                    self.candidates[(sl, pred, ol, 'out')]['edges'].add(self.m)
                    self.candidates[(sl, pred, ol, 'out')]['triples'].add((sub, pred, obj))
                    self.candidates[(sl, pred, ol, 'out')]['facts'].add((sub, pred, obj, time))
                    self.candidates[(sl, pred, ol, 'out')]['ca_to_size'][sub] += 1

                    if (ol, pred, sl, 'in') not in self.candidates:
                        self.candidates[(ol, pred, sl, 'in')] = {'label_coverage': set(),
                                                                    'edges': set(),
                                                                    'triples':set(),
                                                                    'facts':set(),
                                                                    'ca_to_size': defaultdict(int)}
                    self.candidates[(ol, pred, sl, 'in')]['label_coverage'].add((sl, sub))
                    self.candidates[(ol, pred, sl, 'in')]['edges'].add(self.m)
                    self.candidates[(ol, pred, sl, 'in')]['triples'].add((sub, pred, obj))
                    self.candidates[(ol, pred, sl, 'in')]['facts'].add((sub, pred, obj, time))
                    self.candidates[(ol, pred, sl, 'in')]['ca_to_size'][obj] += 1

                    if (sl, pred, ol, 'out') not in self.triple_to_rule[(sub, pred, obj, 'out')]:
                        self.triple_to_rule[(sub, pred, obj, 'out')].append((sl, pred, ol, 'out'))
                    if (ol, pred, sl, 'in') not in self.triple_to_rule[(obj, pred, sub, 'in')]:
                        self.triple_to_rule[(obj, pred, sub, 'in')].append((ol, pred, sl, 'in'))

                    self.label_to_rule[sl].add((sl, pred, ol, 'out'))
                    self.label_to_rule[ol].add((ol, pred, sl, 'in'))

                    self.label_pair_to_rule[(sl, ol)].add((sl, pred, ol, 'out'))
                    self.label_pair_to_rule[(ol, sl)].add((ol, pred, sl, 'in'))
            
            self.id_to_edge[self.m] = (sub, pred, obj, time)
            self.m += 1
        
        self.node_list = list(self.node_list)
        self.num_edge_labels = len(edge_labels)
        self.fact_num = len(self.fact_list)
        self.total_num_labels = 0
        for label, nodes in self.label_matrix.items():
            self.total_num_labels += len(nodes)

        self.total_num_triples = 0
        for rule in self.candidates.keys():
            self.total_num_triples += len(self.candidates[rule]['triples'])

        self.n = len(self.node_list)

        for e in tmp_e_2_t.keys():
            time_list = tmp_e_2_t[e]
            time_list = sorted(time_list)
            self.e_2_t_list[e] = dict()
            for i in range(len(time_list) - 1):
                pre_time = time_list[i]
                aft_time = time_list[i+1]
                self.e_2_t_list[e][aft_time] = time_list[:i+1] #pre_time
        
        for key in self.e_pair_to_rtd.keys():
            rtd_sequence = self.e_pair_to_rtd[key]
            rtd_sequence = sorted(rtd_sequence, reverse=False, key = lambda x: (x[1], x[0], x[2]))
            r_squence_sorted = [(rtd[0], rtd[1], rtd[2]) for rtd in rtd_sequence]
            self.e_pair_to_rtd[key] = r_squence_sorted

        for key in self.t_2_triple.keys():
            triple_sequence = list(self.t_2_triple[key])
            triple_sequence = sorted(triple_sequence, reverse=False, key = lambda x: (x[0], x[1], x[2]))
            self.t_2_triple[key] = triple_sequence

        for key in self.triple_to_rule.keys():
            rule_sequence = self.triple_to_rule[key]
            rule_sequence = sorted(rule_sequence, reverse=True, key = lambda g: (len(self.candidates[g]['triples']), len(self.candidates[g]['ca_to_size']), g[0], g[1]))
            self.triple_to_rule[key] = rule_sequence


    def nodes(self):
        return self.node_list

    def labels(self, node):
        return self.node_to_labels[node]

    def nodes_with_type(self, typ, num_only=True): # 返回具有某一个类别的所有实体(数量)
        return len(self.label_matrix[typ]) if num_only else self.label_matrix[typ]
    
    def nodes_k_hop(self, node, hop): # 返回一个节点的k-hop内节点
        return list(nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys())
    
    def nodes_is_connected(self, node_1, node_2):
        return nx.has_path(self.graph, node_1,node_2)
    
    def nodes_shortese_path(self, node_1, node_2):
        if self.nodes_is_connected(node_1, node_2):
            return nx.shortest_path_length(self.graph, source=node_1, target=node_2, weight=None)
        else:
            return 0
    
    def nodes_shortese_path_time(self, node_1, node_2, time):

        time_list = sorted(list(self.t_2_graph.keys()), reverse=True)
        recent_graph = nx.compose_all([self.t_2_graph[i] for i in time_list[-10:]])
        
        if (node_1 in recent_graph.nodes and node_2 in recent_graph.nodes) and nx.has_path(recent_graph, node_1, node_2):
            return nx.shortest_path_length(recent_graph, source=node_1, target=node_2, weight=None)
        else:
            return 100000
        
    def tuplify(self, rule):
        if self.idify:
            return (tuple(self.id_to_label[label] for label in rule[0]), tuple((self.id_to_pred[child[0]], child[1], self.tuplify(child[2])) for child in rule[1]))
        return (rule[0], tuple((child[0], child[1], self.tuplify(child[2])) for child in rule[1]))
    
    def update_new_e(self, fact, new_e, sample_size):
        self.new_e_set.add(new_e)
        pred = fact[1]
        self.graph.add_edge(fact[0], fact[2])
        self.e2r[new_e] = set()
        self.e2r[new_e].add(pred)
        if pred in self.label_r_to_ids.keys():
            candidate_labels = self.label_r_to_ids[pred][:sample_size]
        else:
            candidate_labels = []
        self.node_to_labels[new_e] = candidate_labels
        self.e_2_t_list[new_e] = dict()
        self.e_2_t_list[new_e][fact[-1]] = [fact[-1]] #pre_time
        for label in candidate_labels:
            self.label_matrix[label].add(new_e)
            self.node_label_counts[label] += 1
    
    
    def update(self, fact, sample_size):
        sub, pred, obj, time = fact[0], fact[1], fact[2], fact[3]
        self.graph.add_edge(sub, obj)
        if time not in self.t_2_graph.keys():
            self.t_2_graph[time] = nx.Graph()
        self.t_2_graph[time].add_edge(sub, obj)
        
        # for sub label update
        if pred in self.e2r[sub]:
            pass
        else:
            
            old_labels = set(self.node_to_labels[sub])
            if pred in self.label_r_to_ids.keys():
                candidate_labels = set(self.label_r_to_ids[pred][:sample_size])
                new_labels = list(old_labels | candidate_labels)
                self.node_to_labels[sub] = new_labels
                for label in candidate_labels:
                    self.label_matrix[label].add(sub)
                    self.node_label_counts[label] += 1
        
        # for sub time line update
        try:
            max_old_time = max(list(self.e_2_t_list[sub].keys()))
            new_time_line = self.e_2_t_list[sub][max_old_time]
            new_time_line.append(max_old_time)
            self.e_2_t_list[sub][time] = new_time_line
        except:
            pass
        
        # for obj label update
        if pred in self.e2r[obj]:
            pass
        else:
                       
            old_labels = set(self.node_to_labels[obj])
            if pred in self.label_r_to_ids.keys():
                candidate_labels = set(self.label_r_to_ids[pred][:sample_size])
                new_labels = list(old_labels | candidate_labels)
                self.node_to_labels[obj] = new_labels
                for label in candidate_labels:
                    self.label_matrix[label].add(obj)
                    self.node_label_counts[label] += 1
            
        # for obj time line update
        try:
            max_old_time = max(list(self.e_2_t_list[obj].keys()))
            new_time_line = self.e_2_t_list[obj][max_old_time]
            new_time_line.append(max_old_time)
            self.e_2_t_list[obj][time] = new_time_line
        except:
            pass

        return
    
    
    
    '''
    def update(self, fact, sample_size):
        sub, pred, obj, time = fact[0], fact[1], fact[2], fact[3]
        self.graph.add_edge(sub, obj)
        if time not in self.t_2_graph.keys():
            self.t_2_graph[time] = nx.Graph()
        self.t_2_graph[time].add_edge(sub, obj)
        
        # for sub label update
        if sub not in self.e2r.keys():
            self.e2r[sub] = set()
            old_labels = set([])
            if pred in self.label_r_to_ids.keys():
                candidate_labels = set(self.label_r_to_ids[pred][:sample_size])
                new_labels = list(old_labels | candidate_labels)
                self.node_to_labels[sub] = new_labels[:sample_size*2]
                for label in candidate_labels:
                    self.label_matrix[label].add(sub)
                    self.node_label_counts[label] += 1
        else:
            if pred in self.e2r[sub]:
                pass
            else:
                
                #old_labels = set(self.node_to_labels[sub])
                #old_r_list = [self.ids_to_labels[i] for i in old_labels][:sample_size]
                #for r_list in old_r_list:
                #    tmp_r_list = tuple(set(r_list) | set([pred]))
                #    if tmp_r_list in self.labels_to_ids.keys():
                #        label = self.labels_to_ids[tmp_r_list]
                #        new_labels = list(old_labels | set([label]))
                #        self.node_to_labels[sub] = new_labels
                #        self.label_matrix[label].add(sub)
                #        self.node_label_counts[label] += 1
                
                old_labels = set(self.node_to_labels[sub])
                if pred in self.label_r_to_ids.keys():
                    candidate_labels = set(self.label_r_to_ids[pred][:sample_size])
                    new_labels = list(old_labels | candidate_labels)
                    self.node_to_labels[sub] = new_labels[:sample_size*2]
                    for label in candidate_labels:
                        self.label_matrix[label].add(sub)
                        self.node_label_counts[label] += 1
                
        # for sub time line update
        try:
            max_old_time = max(list(self.e_2_t_list[sub].keys()))
            new_time_line = self.e_2_t_list[sub][max_old_time]
            new_time_line.append(max_old_time)
            self.e_2_t_list[sub][time] = new_time_line
        except:
            self.e_2_t_list[sub] = {}
            self.e_2_t_list[sub][time] = [time]
            pass
        
        # for obj label update
        if obj not in self.e2r.keys():
            self.e2r[obj] = set()
            old_labels = set([])
            if pred in self.label_r_to_ids.keys():
                candidate_labels = set(self.label_r_to_ids[pred][:sample_size])
                new_labels = list(old_labels | candidate_labels)
                self.node_to_labels[object] = new_labels
                for label in candidate_labels:
                    self.label_matrix[label].add(obj)
                    self.node_label_counts[label] += 1
        else:
            if pred in self.e2r[obj]:
                pass
            else:

                #old_labels = set(self.node_to_labels[obj])
                #old_r_list = [self.ids_to_labels[i] for i in old_labels][:sample_size]
                #for r_list in old_r_list:
                #    tmp_r_list = tuple(set(r_list) | set([pred]))
                #    if tmp_r_list in self.labels_to_ids.keys():
                #        label = self.labels_to_ids[tmp_r_list]
                #        new_labels = list(old_labels | set([label]))
                #        self.node_to_labels[sub] = new_labels
                #        self.label_matrix[label].add(sub)
                #        self.node_label_counts[label] += 1

                old_labels = set(self.node_to_labels[obj])
                if pred in self.label_r_to_ids.keys():
                    candidate_labels = set(self.label_r_to_ids[pred][:sample_size])
                    new_labels = list(old_labels | candidate_labels)
                    self.node_to_labels[obj] = new_labels
                    for label in candidate_labels:
                        self.label_matrix[label].add(obj)
                        self.node_label_counts[label] += 1
                
            
        # for obj time line update
        try:
            max_old_time = max(list(self.e_2_t_list[obj].keys()))
            new_time_line = self.e_2_t_list[obj][max_old_time]
            new_time_line.append(max_old_time)
            self.e_2_t_list[obj][time] = new_time_line
        except:
            self.e_2_t_list[obj] = {}
            self.e_2_t_list[obj][time] = [time]
            pass
    '''
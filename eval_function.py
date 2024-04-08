import pickle
from anomaly_detector import AnomalyDetector
from model_updater import ModelUpdater
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.dummy import freeze_support, Manager
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score


class Eval_function:
    def __init__(self):
        # init parameters
        self.hop = 2 # ICEWS14 2; ICEWS05 2; YAGO 10
        self.span = 2000 # ICEWS14 1000; ICEWS05 2000; YAGO 50
        self.span_t = 2000 # ICEWS14 200; ICEWS05 2000; YAGO 50
        self.span_m = 2000 # ICEWS14 150; ICEWS05 2000; YAGO 50
        self.step = 1 # ICEWS14 2; ICEWS05 2; YAGO 1
        self.aux_score = True # ICEWS14 T; ICEWS05 T; YAGO F
        self.aux_score_m = True # ICEWS14 F; ICEWS05 F; YAGO F
        self.distribution_aware = True # ICEWS14 T; ICEWS05 F; YAGO F
        self.and_or = 'and' # ICEWS14 or; ICEWS05 and; YAGO or
        self.update_sample_size = 20 # ICEWS14 10; ICEWS05 10; YAGO 20
        self.update_span = 2000 # ICEWS14 200; ICEWS05 3000; YAGO 20

    def sigmoid(self, z):
            return 1/(1 + np.exp(-z))

    def get_f1(self, p, r):
        if (p+r) == 0:
            return 0
        else:
            return 2*(p*r)/(p+r)

    def get_f05(self, p, r):
        if (p+r) == 0:
            return 0
        else:
            return 1.25*(p*r)/(0.25*p+r)
        
    def get_metric(self, pred, y, anomaly_type):
        precision, recall, threshold = precision_recall_curve(y, pred)
        # find the best threshold by F1 score

        f1s = [self.get_f1(precision[i], recall[i]) for i in range(len(precision))]
        f05s = [self.get_f05(precision[i], recall[i]) for i in range(len(precision))]
        optimal_index = np.argmax(f05s)
        '''
        while precision[optimal_index] == 1.0:
            f05s[optimal_index] = 0
            optimal_index = np.argmax(f05s)
        '''

        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        

        optimal_P_by_fscore, optimal_R_by_fscore, optimal_F1_by_fscore, optimal_F05_by_fscore, optimal_T_by_fscore = precision[optimal_index], recall[optimal_index], f1s[optimal_index], f05s[optimal_index], threshold[optimal_index]
        optimal_ACC_by_fscore = accuracy_score(y, pred > optimal_T_by_fscore)

        print('----- Result stats -----' + anomaly_type)
        print('Number of test samples: ' + str(len(y)))
        print('Number of anomaly samples: ' + str(sum(y)))
        print('-----------------------')
        print('Best result by F1-score:')
        print('P: ' + str(optimal_P_by_fscore))
        print('F05: ' + str(optimal_F05_by_fscore))
        print('AUC: ' + str(auc))

    def update_helper_valid(self, m):
        raw_score, project_rules = self.detector.score_edge_temporal((int(m[0]), int(m[1]), int(m[2]), int(m[3])), file_type='valid')
        self.detector.update(m, project_rules, self.update_sample_size, self.update_span)

    def update_helper_test(self, x_y):
        m = x_y[0]
        project_rules = x_y[1]
        self.detector.update(m, project_rules, self.update_sample_size, self.update_span)

    def concept_detect_helper(self, fact_label_t):
        m = fact_label_t[0]
        label = fact_label_t[1]
        t = fact_label_t[2]
        raw_score = self.detector.score_edge((int(m[0]), int(m[1]), int(m[2]), int(t)), hop = self.hop, max_span = self.span, and_or = self.and_or)
        pred = self.sigmoid(raw_score)

        return [pred, label]

    def temporal_detect_helper(self, fact_label_t):
        m = fact_label_t[0]
        label = fact_label_t[1]
        t = fact_label_t[2]
        raw_score, project_rules = self.detector.score_edge_temporal((int(m[0]), int(m[1]), int(m[2]), int(t)), hop = self.hop, max_span = self.span_t, and_or = self.and_or, max_step = self.step, aux_score = self.aux_score, dsa = self.distribution_aware, file_type='test')
        pred = self.sigmoid(raw_score)

        return [pred, label, m, project_rules]

    def missing_detect_helper(fact_label_t):
        m = fact_label_t[0]
        label = fact_label_t[1]
        t = fact_label_t[2]
        raw_score = detector.score_edge_missing((int(m[0]), int(m[1]), int(m[2]), int(t)), hop_c = hop, max_span_c = span, max_span_t = span_m, and_or = and_or, max_step = step, aux_score = aux_score_m, dsa = distribution_aware)
        pred = sigmoid(-raw_score)

        return [pred, label]

def eval_model():
    # init model
    print("reading saved model...")
    graph_name = 'GDELT'
    root_path = '/home/zhangjs/expriment/TKGist/data/'
    dataset = graph_name+'/'

    graph = pickle.load(open(root_path + dataset + "graph.pickle", "rb"))
    model = pickle.load(open(root_path + dataset + "static_model.pickle", "rb"))
    temporal_model = pickle.load(open(root_path + dataset + "temporal_model.pickle", "rb"))
    print("dnoe")

    # init data
    print("reading anomalies data...")
    def read_file(input_file):
        raw_data = []
        for fact in input_file.readlines():
            s, r, o, t = fact.strip().split('	')[:4]
            s = int(s)
            r = int(r)
            o = int(o)
            t = int(t)
            raw_data.append((s, r, o, t))
        return raw_data

    valid_pos, test_pos = read_file(open(root_path + dataset + 'valid.txt', 'r')), read_file(open(root_path + dataset + 'test.txt', 'r'))
    valid_t_2_C_neg, test_t_2_C_neg = pickle.load(open(root_path + dataset + '/conceptual_errors.pickle', 'rb'))
    valid_t_2_T_neg, test_t_2_T_neg = pickle.load(open(root_path + dataset + '/time_errors.pickle', 'rb'))
    valid_t_2_M_neg, test_t_2_M_neg = pickle.load(open(root_path + dataset + '/missing_errors.pickle', 'rb'))

    valid_t_2_pos = {}
    test_t_2_pos = {}
    for sample in valid_pos:
        s, r, o, t = sample
        if t not in valid_t_2_pos.keys():
            valid_t_2_pos[t] = []
        valid_t_2_pos[t].append((int(s), int(r), int(o), int(t)))

    for sample in test_pos:
        s, r, o, t = sample
        if t not in test_t_2_pos.keys():
            test_t_2_pos[t] = []
        test_t_2_pos[t].append((int(s), int(r), int(o), int(t)))
    print("dnoe")
    
    # start eval
    detector = AnomalyDetector(temporal_model)
    # detect concept+time+missing anomaly

    pred_list_C = [] # for concept anomaly
    label_list_C = []

    pred_list_T = [] # for time anomaly
    label_list_T = []

    pred_list_M = [] # for missing anomaly
    label_list_M = []

    all_proj_rules = []

    p = mp.Pool(50)
    # update model in validate set
    for t in tqdm(valid_t_2_pos.keys()):
        pos_samples = valid_t_2_pos[t]
        #p.map(update_helper_valid, pos_samples)

    # detect anomalies in test set
    for t in tqdm(test_t_2_C_neg.keys()):
        pos_samples = test_t_2_pos[t]
        pos_missing = test_t_2_M_neg[1][t]

        concept_temporal_p = [[pos_samples[i], 0, t] for i in range(len(pos_samples))]
        missing_p = [[pos_missing[i], 1, t] for i in range(len(pos_missing))]
        concept_score_label_p = p.map(concept_detect_helper, concept_temporal_p)
        temporal_score_label_proj_p = p.map(temporal_detect_helper, concept_temporal_p)
        missing_score_label_p = p.map(missing_detect_helper, missing_p)
        for i in range(len(concept_score_label_p)):
            pred_list_C.append(concept_score_label_p[i][0])
            label_list_C.append(concept_score_label_p[i][1])
        
        for i in range(len(temporal_score_label_proj_p)):
            pred_list_T.append(temporal_score_label_proj_p[i][0])
            label_list_T.append(temporal_score_label_proj_p[i][1])
            all_proj_rules.append([temporal_score_label_proj_p[i][2], temporal_score_label_proj_p[i][3]])
        
        for i in range(len(missing_score_label_p)):
            pred_list_M.append(missing_score_label_p[i][0])
            label_list_M.append(missing_score_label_p[i][1])
        


        concept_n = [[test_t_2_C_neg[t][i], 1, t] for i in range(len(test_t_2_C_neg[t]))]
        temporal_n = [[test_t_2_T_neg[t][i], 1, t] for i in range(len(test_t_2_T_neg[t]))]
        missing_n = [[test_t_2_M_neg[0][t][i], 0, t] for i in range(len(test_t_2_M_neg[0][t]))]
        concept_score_label_n = p.map(concept_detect_helper, concept_n)
        temporal_score_label_proj_n = p.map(temporal_detect_helper, temporal_n)
        missing_score_label_n = p.map(missing_detect_helper, missing_n)
        
        for i in range(len(concept_score_label_n)):
            pred_list_C.append(concept_score_label_n[i][0])
            label_list_C.append(concept_score_label_n[i][1])
        
        for i in range(len(temporal_score_label_proj_n)):
            pred_list_T.append(temporal_score_label_proj_n[i][0])
            label_list_T.append(temporal_score_label_proj_n[i][1])
        
        for i in range(len(missing_score_label_n)):
            pred_list_M.append(missing_score_label_n[i][0])
            label_list_M.append(missing_score_label_n[i][1])
        
        # update model in test set
        #p.map(update_helper_test, all_proj_rules)
        all_proj_rules = []

    get_metric(np.array(pred_list_C), np.array(label_list_C), 'concept') # concept
    get_metric(np.array(pred_list_T), np.array(label_list_T), 'time') # time
    get_metric(np.array(pred_list_M), np.array(label_list_M), 'missing') # missing
    detector.re_fresh()
    p.close()




if __name__ == '__main__':
    eval_model()
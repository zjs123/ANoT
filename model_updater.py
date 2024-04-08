class ModelUpdater:
    def __init__(self, temporal_model):
        self.raw_temporal_model = temporal_model
    
    def update_model(self, temporal_model, fact, sample_size, span):
        temporal_model.model.graph.update(fact, sample_size)
        new_rules = temporal_model.model.update(fact, sample_size)
        if len(new_rules) != 0:
            temporal_model.add_new_rules(new_rules, span)
    
    def update_temporal_model(self, temporal_model, pre_rule, rule, span, sample_size):
        temporal_model.update((pre_rule, rule), span)
    
    def new_e_handeler(self, temporal_model, fact, new_entity, sample_size):
        temporal_model.model.graph.update_new_e(fact, new_entity, sample_size)
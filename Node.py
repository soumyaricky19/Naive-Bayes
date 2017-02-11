class Node:
    def __init__(self,split_att_index=0,split_att_value=2,possible_attributes_index=[],n_num=0,p_num=0,decision=2,att_to_split_on_index=0,false_child=None,true_child=None):
        self.split_att_index=split_att_index
        self.split_att_value=split_att_value
        self.possible_attributes_index=possible_attributes_index
        self.p_num=p_num
        self.n_num=n_num   
        self.decision=decision
        self.att_to_split_on_index=att_to_split_on_index    
        self.false_child=false_child
        self.true_child=true_child
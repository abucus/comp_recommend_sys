'''
Created on Feb 12, 2015

@author: tengmf
'''
import os.path as op
import os
from src.input.read_csv import read_in, generate_file, generate_PIMF_data

def prepare_general_validation_data():
    base_path = op.join("..","..","output","data2","validate")
    training_output = op.join(base_path, "training") 
    test_output = op.join(base_path, "test") 
    if not op.exists(training_output):
        os.makedirs(training_output)
    if not op.exists(test_output):
        os.makedirs(test_output) 
    
    training_data_ratio = .65
    
    training_data = read_in(source_path=op.join("..", "..", "output", "data2", "original", 'simpleB.csv'))
    test_data = {'event_types':training_data['event_types'], 'table':{}}
    test_table = test_data['table']
    training_table = training_data['table']
    for k,v in training_table.iteritems():
        test_table[k] = {'company':v['company'], 'events':[]}
        total = len(v['events'])
        split_idx = int(total*training_data_ratio)
        if split_idx == 0:
            split_idx = total -1
        test_table[k]['events'].extend(v['events'][split_idx+1 : total])
        del v['events'][split_idx+1:total]
        
    generate_file(training_data, training_output)
    generate_file(test_data, test_output)
    
def prepare_pimf_validation_data(source_path=op.join("..", "..", "output", "data2", "original", 'simpleB.csv'), 
                                 out_path = op.join("..","..","output","data2","validate","pimf")):
    
    training_output = op.join(out_path, "training") 
    test_output = op.join(out_path, "test") 
    if not op.exists(training_output):
        os.makedirs(training_output)
    if not op.exists(test_output):
        os.makedirs(test_output) 
    
    training_data_ratio = .65
    
    training_data = read_in(source_path)
    test_data = {'event_types':training_data['event_types'], 'table':{}}
    test_table = test_data['table']
    training_table = training_data['table']
    for k,v in training_table.iteritems():
        test_table[k] = {'company':v['company'], 'events':[]}
        total = len(v['events'])
        split_idx = int(total*training_data_ratio)
        if split_idx == 0:
            split_idx = total -1
        test_table[k]['events'].extend(v['events'][split_idx+1 : total])
        del v['events'][split_idx+1:total]
        
    generate_PIMF_data(training_data, training_output)
    generate_PIMF_data(test_data, test_output, True)
    
if __name__ == '__main__':
    prepare_pimf_validation_data(source_path=op.join("..", "..", "output", "data2", "original", 'simpleB.csv'), out_path = op.join("..","..","output","data2","validate","pimf"))
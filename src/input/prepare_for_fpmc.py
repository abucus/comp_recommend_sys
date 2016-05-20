from src.input.read_csv import read_in
import csv, pickle, os
import os.path as op
import numpy as np

def prepare_validation_data(raw_data, out_path, ratio=0.6):
	training_data = raw_data
	test_data = {'event_types':training_data['event_types'], 'table':{}, 'users':training_data['users']}
	test_table = test_data['table']
	training_table = training_data['table']
	for k, v in training_table.items():
	    test_table[k] = {'company':v['company'], 'events':[]}
	    total = len(v['events'])
	    split_idx = int(total * ratio)
	    if split_idx == 0:
	        split_idx = total - 1
	    test_table[k]['events'].extend(v['events'][split_idx + 1 : total])
	    del v['events'][split_idx + 1:total]
	    
	generate_file_for_data(training_data, op.join(out_path, 'training'))
	generate_file_for_data(test_data, op.join(out_path, 'test'))
	

def generate_file_for_data(raw_data, out_path):

	event_column_maps = dict()
	for i, e in enumerate(raw_data['event_types']):
		event_column_maps[e] = i
	pickle.dump(event_column_maps, open(op.join(out_path, 'event_column_maps'), 'w'))	

	user_row_maps = dict()
	i = 0
	for i, u in enumerate(raw_data['users']):
		user_row_maps[u] = i
	pickle.dump(user_row_maps, open(op.join(out_path, 'user_row_maps'), 'w'))

	a = np.zeros((len(user_row_maps), len(event_column_maps), len(event_column_maps)))
	data = dict()
	data['i_num'] = len(event_column_maps)
	data['u_num'] = len(user_row_maps)
	data['transactions'] = {}

	for k, v in raw_data['table'].items():
		events = [event_column_maps[e[1]] for e in v['events']]
		data['transactions'][user_row_maps[k]] = events
		update_matrix(events , a[user_row_maps[k]])
	pickle.dump(a, open(op.join(out_path, 'full_tensor'), 'w'))
	pickle.dump(data, open(op.join(out_path, 'data'), 'w'))


def update_matrix(events, matrix):
	
	pair_events = [(events[i], events[i + 1]) for i in range(len(events) - 1)]

	seperator = '#'
	pair_count = dict()
	last_count = dict()
	for p in pair_events:
		code = str(p[0]) + seperator + str(p[1])
		if code not in pair_count:
			pair_count[code] = 0
		pair_count[code] += 1
		if p[0] not in last_count:
			last_count[p[0]] = 0
		last_count[p[0]] += 1

	for p, count in pair_count.items():
		idxes = p.split(seperator)
		l = int(idxes[0])
		i = int(idxes[1])
		matrix[l, i] = 1.*count / last_count[l]

if __name__ == "__main__":
	generate_file_for_data(read_in(op.join('..', '..', 'output', 'data', 'original', 'simpleA2.csv')),
		out_path=op.join('..', '..', 'output', 'fpmc_data'))

	traing_data_dir = op.join('..', '..', 'output', 'fpmc_data', 'training')
	test_data_dir = op.join('..', '..', 'output', 'fpmc_data', 'test')
	if not op.exists(traing_data_dir):
		os.mkdir(traing_data_dir)

	if not op.exists(test_data_dir):
		os.mkdir(test_data_dir)
	prepare_validation_data(read_in(op.join('..', '..', 'output', 'data', 'original', 'simpleA2.csv')),
		op.join('..', '..', 'output', 'fpmc_data'))



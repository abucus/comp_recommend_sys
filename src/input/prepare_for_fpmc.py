from src.input.read_csv import read_in
import csv,pickle
import os.path as op
import numpy as np

def prepare(in_path, out_path):
	raw_data = read_in(in_path)

	event_column_maps=dict()
	for i,e in enumerate(raw_data['event_types']):
		event_column_maps[e] = i
	pickle.dump(event_column_maps, open(op.join(out_path, 'event_column_maps'), 'w'))	

	user_row_maps=dict()
	i = 0
	for i,u in enumerate(raw_data['table'].keys()):
		user_row_maps[u] = i
	pickle.dump(user_row_maps, open(op.join(out_path, 'user_row_maps'), 'w'))


	a = np.zeros((len(user_row_maps), len(event_column_maps), len(event_column_maps)))
	for k,v in raw_data['table'].iteritems():
		update_matrix(v['events'], event_column_maps, a[user_row_maps[k]])
	pickle.dump(a, open(op.join(out_path, 'full_tensor'), 'w'))



def update_matrix(events, event_column_maps, matrix):
	events = [event_column_maps[e[1]] for e in events]
	pair_events = [(events[i], events[i+1]) for i in range(len(events)-1)]

	seperator = '#'
	pair_count = dict()
	last_count = dict()
	for p in pair_events:
		code = str(p[0])+seperator+str(p[1])
		if code not in pair_count:
			pair_count[code] = 0
		pair_count[code] += 1
		if p[0] not in last_count:
			last_count[p[0]] = 0
		last_count[p[0]] += 1

	for p,count in pair_count.iteritems():
		idxes = p.split(seperator)
		l = int(idxes[0])
		i = int(idxes[1])
		matrix[l,i] = 1.*count/last_count[l]

if __name__ == "__main__":
	prepare(in_path = op.join('..','..','output','data2','original','simpleB.csv'), 
		out_path = op.join('..','..','output','fpmc_data'))
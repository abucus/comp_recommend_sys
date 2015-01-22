import csv
import numpy as np
import os.path as op
from datetime import datetime

base_file_path = '.'
source_path = op.join(base_file_path,'a.csv')
output_pure_matrix = op.join(base_file_path, 'pure_matrix.csv')
output_full_matrix = op.join(base_file_path, 'full_matrix.csv')
output_col_map = op.join(base_file_path, 'column_map.csv')
id_column = 0
time_column = 3
event_type_column = 7

# read in all data and organize in a map like
# {'id1':[(time1,event_type1),(time2,event_type2)]} the event sequence is ordered by date-time
with open(source_path) as cf:
    reader = csv.reader(cf, delimiter=',')
    table = {}
    event_types = []
    date_format = '%Y-%m-%d %H:%M:%S.0%f'
    for row in reader:
        cid = row[id_column]
        if(cid not in table):
            table[cid]=[]
        table[cid].append((datetime.strptime(row[time_column],date_format),row[event_type_column]))
        if(row[event_type_column] not in event_types):
            event_types.append(row[7])
    
    for v in table.itervalues():
        v.sort(key = lambda l:l[0])


# genearte the mapping between the column number and column name
# key: event_type_1+@+event_type_2
# value: column number
event_types.sort()
col_names = []
col_num_name_map = {}
col_num = 0
with open(output_col_map, 'w') as cmf:
    writer = csv.writer(cmf, delimiter=',')
    writer.writerow(['Column Number','Column Name'])
    for i in range(0,len(event_types)):
        for j in range(0, len(event_types)):
            col_name = event_types[i] + '@' + event_types[j]
            col_names.append(col_name)
            col_num_name_map[col_name] = col_num
            writer.writerow([col_num, col_name])
            col_num += 1

# generate pure matrix and write to csv
m = np.full((len(table.keys()),len(event_types)**2),-1)
row_names = []
r = 0
for k,v in table.iteritems():
    row_names.append(str(k))
    for i in range(0,len(v)):
        for j in range(i+1, len(v)):
            if(v[i][1] == v[j][1]):
                continue
            else:
                m[r,col_num_name_map[v[i][1]+'@'+v[j][1]]] = (v[j][0] - v[i][0]).total_seconds()
#                print "put ",(v[j][0] - v[i][0]).total_seconds()," in cell ",r," ",col_num_name_map[v[i][1]+'@'+v[j][1]]
    r += 1        
np.savetxt(output_pure_matrix, m, delimiter=",")

# generate well format csv
with open(output_full_matrix,'w') as out_file:
    writer = csv.writer(out_file, delimiter=',')
    writer.writerow(['id']+col_names)
    with open(output_pure_matrix, 'r') as in_file:
        reader = csv.reader(in_file, delimiter=',')
        row_num = 0
        for row in reader:
            writer.writerow([row_names[row_num]]+row)
            row_num += 1


# output column number and name mapping for pure matrix


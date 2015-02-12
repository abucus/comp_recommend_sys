import csv
import numpy as np
import os.path as op
import json
from datetime import datetime


def read_in(source_path = op.join("..","..","output","data", "original",'simpleA2.csv')):
    '''
    Read in all data and organize in a map like
    {'id1':[(time1,event_type1),(time2,event_type2)]} the event sequence is ordered by date-time
    '''
    id_column = 0
    time_column = 1
    company_column = 2
    event_type_column = 3
    with open(source_path) as cf:
        reader = csv.reader(cf, delimiter=',')
        reader.next()
        table = {}
        event_types = []
        date_format = '%Y-%m-%d %H:%M:%S.0%f'
        for row in reader:
            cid = row[id_column]
            if(cid not in table):
                company = row[company_column]
                table[cid] = {'company':company, 'events':[]}
            table[cid]['events'].append((datetime.strptime(row[time_column], date_format), row[event_type_column]))
            if(row[event_type_column] not in event_types):
                event_types.append(row[event_type_column])
        
        for v in table.itervalues():
            v['events'].sort(key=lambda l:l[0])
            
    print "event_types:#", len(event_types), "\n", event_types
    return {'event_types':event_types, 'table':table}

def generate_file(data, base_file_path = op.join("..","..","output","data")):
    table = data['table']
    event_types = data['event_types']
    
    output_pure_matrix = op.join(base_file_path, 'pure_matrix.csv')
    output_full_matrix = op.join(base_file_path, 'full_matrix.csv')
    output_col_map = op.join(base_file_path, 'column_map.csv')
    output_stat_path = op.join(base_file_path, "stat.txt")
    output_company_matrix = op.join(base_file_path, "company_matrix")
    # output the stat info
    total_days_interval = 0
    interval_count = 0
    for c in table.itervalues():
        v = c['events']
        for i in range(len(v) - 1):
            if v[i][1] == v[i+1][1]:
                interval_count += 1
                total_days_interval += (v[i+1][0] - v[i][0]).total_seconds()/3600.0/24.0
    delta = total_days_interval/interval_count
    json.dump({
               "total_days_interval":total_days_interval, 
               "interval_count":interval_count, 
               "delta":delta
               },open(output_stat_path,"w"))    
    
    # genearte the mapping between the column number and column name
    # key: event_type_1+'@'+event_type_2
    # value: column number
    event_types.sort()
    col_names = []
    col_num_name_map = {}
    col_num = 0
    with open(output_col_map, 'w') as cmf:
        writer = csv.writer(cmf, delimiter=',')
        writer.writerow(['Column Number', 'Column Name'])
        for i in range(0, len(event_types)):
            for j in range(0, len(event_types)):
                col_name = event_types[i] + '@' + event_types[j]
                col_names.append(col_name)
                col_num_name_map[col_name] = col_num
                writer.writerow([col_num, col_name])
                col_num += 1
    
    # generate pure matrix and write to csv
    m = np.full((len(table.keys()), len(event_types) ** 2), 0)
    row_names = []
    r = 0
    for cid, record in table.iteritems():
        row_names.append(str(cid))
        v = record['events']
        for i in range(0, len(v)):
            d = 0.
            for j in range(i + 1, len(v)):
                if(v[i][1] == v[j][1]):
                    m[r, col_num_name_map[v[i][1] + '@' + v[j][1]]] = 1
                    continue
                else:
                    d += np.exp(-(v[j][0] - v[j-1][0]).total_seconds()/3600.0/24.0/delta)
                    m[r, col_num_name_map[v[i][1] + '@' + v[j][1]]] = d
        r += 1        
    np.savetxt(output_pure_matrix, m, delimiter=",")
    
    # generate well format csv
    with open(output_full_matrix, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(['id'] + col_names)
        with open(output_pure_matrix, 'r') as in_file:
            reader = csv.reader(in_file, delimiter=',')
            row_num = 0
            for row in reader:
                writer.writerow([row_names[row_num]] + row)
                row_num += 1
                
    # generate company matrix
    company_matrix = np.zeros((len(table), len(table)))
    cur_row = 0
    for k,v in table.iteritems():
        cur_col = 0
        for k2,v2 in table.iteritems():
            if v['company'] == v2['company']:
                company_matrix[cur_row, cur_col] = 1
            else:
                company_matrix[cur_row, cur_col] = 0
            cur_col += 1
        cur_row += 1
    np.savetxt(output_company_matrix, company_matrix)            

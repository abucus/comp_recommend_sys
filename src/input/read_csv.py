from collections import namedtuple, Counter
from datetime import datetime
import os, json, pickle, csv, time, logging

import numpy as np
import os.path as op
from src.log import get_logger
# logging.basicConfig(filename='./read_csv.log',level=logging.DEBUG, 
#                     format='%(asctime)s %(message)s')

def read_in(source_path=op.join("..", "..", "output", "data2", "original", 'small.csv')):
    '''
    Read in all data and organize in a map like
    {'id1':[(time1,event_type1),(time2,event_type2)]} the event sequence is ordered by date-time
    '''
    logger = get_logger(__name__)
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
            
    # print "event_types:#", len(event_types), "\n", event_types
    return {'event_types':event_types, 'table':table, 'users':table.keys()} 

def generate_file(data, base_file_path=op.join("..", "..", "output", "data")):
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
            if v[i][1] != v[i + 1][1]:
                interval_count += 1
                total_days_interval += (v[i + 1][0] - v[i][0]).total_seconds() / 3600.0 / 24.0    
    r = total_days_interval / interval_count
    json.dump({
               "total_days_interval":total_days_interval,
               "interval_count":interval_count,
               "r":r
               }, open(output_stat_path, "w"))    
    
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
    delta = 90
    m = np.full((len(table.keys()), len(event_types) ** 2), 0)
    row_names = []
    k = 0
    for cid, record in table.iteritems():
        row_names.append(str(cid))
        v = record['events']
        v_len = len(v)
        for i in range(0, v_len):
            for j in range(i + 1, v_len):
                if(v[i][1] == v[j][1]):
                    m[k, col_num_name_map[v[i][1] + '@' + v[j][1]]] = 1. / v_len
                else:
                    period_in_day = (v[j][0] - v[j - 1][0]).total_seconds() / 3600.0 / 24.0
                    if period_in_day < delta:
                        m[k, col_num_name_map[v[i][1] + '@' + v[j][1]]] = np.exp(-period_in_day / r) / v_len
        k += 1        
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
    for k, v in table.iteritems():
        cur_col = 0
        for k2, v2 in table.iteritems():
            if v['company'] == v2['company']:
                company_matrix[cur_row, cur_col] = 1
            else:
                company_matrix[cur_row, cur_col] = 0
            cur_col += 1
        cur_row += 1
    np.savetxt(output_company_matrix, company_matrix) 
     
def generate_R(c_events, window=90):
    '''
    generate_R matrix for single user using algorithms 1 in Gang Zhao's Paper
    '''    
    if len(c_events) == 0:
        return []
    Event = namedtuple('Event', 'id t')
    Record = namedtuple('Record', 'i j x c')
    Q, R, RW, RWP = [], [], [], []
    for original_e in c_events:
        # original_e (date,event_type)
        e = Event(original_e[1], original_e[0])
        
        while len(Q) > 0 and cal_interval_in_days(Q[0].t, e.t) > window:
            ep = Q.pop(0)
            RW = filter(lambda r: r.i != ep.id, RW)
        if len(Q) > 0:
            xp = cal_interval_in_days(Q[-1].t, e.t)
            cp = 1 if xp > 0 else 0
        else:
            xp = cp = 0
        if e.id in [r_.id for r_ in Q]:
            while RW[0].i != e.id:
                r = RW[0]
                RWP.append(Record(r.i, e.id, r.x + xp, r.c + cp)) 
                RW.pop(0)
        for r in RW:
            rp = Record(r.i, e.id, r.x + xp, r.c + cp)
            RWP.append(rp)
            R.append(rp)
        
        RW = RWP
        RWP = []
        
        Q = filter(lambda ep: ep.id != e.id, Q)
        RW = filter(lambda r: r.i != e.id, RW)
        
        r = Record(e.id, e.id, 0, 0)
        RW.append(r)
        Q.append(e)
#         print '------------------------------------------------'
#         print 'current event ',e
#         print 'Q:',Q
#         print 'RW:',RW
#         print 'R:',R

    return R

def cal_interval_in_days(d1, d2):
    return (d2 - d1).total_seconds() / 3600.0 / 24.0
    # return d2 - d1
def cal_duij(R, i, j):
    Rp = filter(lambda r: r.i == i and r.j == j, R)
    if len(Rp) == 0:
        return None
    counts = np.array(map(lambda r:r.c, Rp))
    intervals = np.array(map(lambda r:r.x, Rp))
    return np.sum(intervals / np.log2(2 + counts)) / np.sum(1. / np.log2(2 + counts))
        
def generate_PIMF_data(data, base_file_path=op.join("..", "..", "output", "data2", "PIMF"), is_test=False):
    logger = get_logger(__name__)
    
    if not op.exists(base_file_path):
        os.makedirs(base_file_path)
        
    output_col_map = op.join(base_file_path, 'column_map.csv')
    delta_t = 20
    table = data['table']
    event_types = data['event_types']
    pickle.dump(table, open(op.join(base_file_path, 'table'), 'w'))
    logger.info('table dumped')
       
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
                
    # generate col(event) map
    event_types_map = dict()
    for i in xrange(len(event_types)):
        event_types_map[event_types[i]] = i
    pickle.dump(event_types_map, open(op.join(base_file_path, 'event_map'), 'w'))  
    
    # generate row map
    row_map = dict()
    row_idx = 0
    for cid in table.iterkeys():
        row_map[cid] = row_idx
        row_idx += 1
    pickle.dump(row_map, open(op.join(base_file_path, 'user_map'), 'w'))
    logger.info("user map dumped")
    
    # generate utility matrix
    utility = np.zeros((len(table), len(event_types)))
    for cid, c_data in table.iteritems():
        events = [e[1] for e in c_data['events']]
        count = Counter(events)
        for k in count:
            utility[row_map[cid], event_types_map[k]] = count[k]
    np.savetxt(op.join(base_file_path, 'utility'), utility)
    logger.info("utility matrix dumped")
    
    if is_test:
        return
    # generate d(u,i,j)
    # d = {(u,i,j):duij}
    d = dict()
    total = len(table)
    cur = 0
    for cid, c_data in table.iteritems():
        R = generate_R(c_data['events'], delta_t)
        events = set([e[1] for e in c_data['events']])
        d[cid] = dict()
        for e in events:
            for e2 in events:
                duij = cal_duij(R, e, e2)
                if duij:
                    d[(cid, event_types_map[e], event_types_map[e2])] = cal_duij(R, e, e2)
                    # d[cid][(event_types_map[e], event_types_map[e2])] = cal_duij(R, e, e2)
                    # d[cid][(e, e2)] = cal_duij(R, e, e2)
        logger.info('d complete {}%'.format(int(cur * 1. / total * 100)))
        cur += 1
    
    # generate dp(u,i,j)
    dp = dict()
    cur = 0
    users_count = len(table)
    for u in table.iterkeys():
        for i in xrange(len(event_types)):
            for j in xrange(len(event_types)):
                company = table[u]['company']
                uijs = filter(lambda x: x[1] == i and x[2] == j, d.keys())
                if len(uijs) != 0:
                    duijs = [d[key] for key in uijs]
                    sim = [2 if company == table[x[0]]['company'] else 1 for x in uijs]
                    same_company_count = len(filter(lambda x: table[x]['company'] == company, table.keys()))
                    denominator = users_count + same_company_count
                    numerator = np.dot(duijs, sim)
                    if np.isnan(denominator) or np.isnan(numerator) or np.abs(denominator) < 1e-10:
                        logger.error("msg u:{}\ni:{}\n\j:{}\nduijs:{}\n".format(u, i, j, str(duijs)))
                        raise Exception("find a nan")
                        
                    dp[(u, i, j)] = numerator / denominator
                        
        logger.info('d complete {}%'.format(int(cur * 1. / users_count * 100)))
        cur += 1
        
    del d            
    pickle.dump(dp, open(op.join(base_file_path, 'dp'), 'w'))
    return dp
        
def generate_PIMF_data2(data, base_file_path=op.join("..", "..", "output", "data2", "PIMF")):
    logger = get_logger(__name__)
    
    if not op.exists(base_file_path):
        os.makedirs(base_file_path)
        
    output_col_map = op.join(base_file_path, 'column_map.csv')
    delta_t = 20
    table = data['table']
    event_types = data['event_types']
    pickle.dump(table, open(op.join(base_file_path, 'table'), 'w'))
    logger.info('table dumped')
       
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
                
    # generate col(event) map
    event_types_map = dict()
    for i in xrange(len(event_types)):
        event_types_map[event_types[i]] = i
    pickle.dump(event_types_map, open(op.join(base_file_path, 'event_map'), 'w'))  
    events_count = len(event_types_map)
    
    # generate row map
    row_map = dict()
    row_idx = 0
    for cid in table.iterkeys():
        row_map[cid] = row_idx
        row_idx += 1
    pickle.dump(row_map, open(op.join(base_file_path, 'user_map'), 'w'))
    logger.info("user map dumped")
    users_count = len(row_map)
    
    # generate utility matrix
    utility = np.zeros((users_count, events_count))
    for cid, c_data in table.iteritems():
        events = [e[1] for e in c_data['events']]
        count = Counter(events)
        for k in count:
            utility[row_map[cid], event_types_map[k]] = count[k]
    np.savetxt(op.join(base_file_path, 'utility'), utility)
    logger.info("utility matrix dumped")
    
    # generate d(u,i,j)
    # d = {(u,i,j):duij}
    d = np.zeros((users_count, events_count, events_count))
    cur = 0
    for cid, c_data in table.iteritems():
        R = generate_R(c_data['events'], delta_t)
        events = set([e[1] for e in c_data['events']])
        for e in events:
            for e2 in events:
                duij = cal_duij(R, e, e2)
                if duij:
                    d[row_map[cid], event_types_map[e], event_types_map[e2]] = duij
                    # d[cid][(event_types_map[e], event_types_map[e2])] = cal_duij(R, e, e2)
                    # d[cid][(e, e2)] = cal_duij(R, e, e2)
        logger.info('d complete {}%'.format(int(cur * 1. / users_count * 100)))
        cur += 1
    
    C = np.zeros((len(row_map), len(row_map)))
    for u1 in table.iterkeys():
        for u2 in table.iterkeys():
            if table[u1]['company'] == table[u1]['company']:
                C[row_map[u1], row_map[u2]] = 2
            else:
                C[row_map[u1], row_map[u2]] = 1
    C = C / C.sum(1)[:, None]
    logger.info('C computed')
    # generate dp(u,i,j)    
    dp = np.tensordot(C, d, ([1], [0]))
    logger.info('dp computed')
    
    del d            
    pickle.dump(dp, open(op.join(base_file_path, 'dp'), 'w'))
    return dp
if __name__ == '__main__':
#     d = dict()
#     d['table'] = {'jkdajfakjsdklfj':{'events':[(1, 'c'), (8, 'd'), (13, 'b'), (16, 'e'), (18, 'b'), (22, 'a')]}}
    generate_PIMF_data(read_in(source_path=op.join("..", "..", "output", "data2", "original", 'simpleB.csv')),
                        base_file_path=op.join("..", "..", "output", "data2", "validate", "pimf"))
#     R = generate_R([(1, 'c'), (8, 'd'), (13, 'b'), (16, 'e'), (18, 'b'), (22, 'a')], 20)
#     for i in ['a','b','c','d','e']:
#         for j in ['a','b','c','d','e']:
#             print i,j,cal_duij(R, i, j)

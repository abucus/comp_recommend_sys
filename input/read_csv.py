import csv
import numpy as np
from datetime import datetime
with open("/home/tengmf/a.csv") as cf:
    reader = csv.reader(cf, delimiter=',')
    table = {}
    event_types = []
    date_format = '%Y-%m-%d %H:%M:%S.0%f'
    for row in reader:
        if(row[0] not in table):
            table[row[0]]=[]
        table[row[0]].append((datetime.strptime(row[3],date_format),row[7]))
        if(row[7] not in event_types):
            event_types.append(row[7])
    
    for v in table.itervalues():
        v.sort(key = lambda l:l[0])


# genearte the mapping between the column number and column name
event_types.sort()
col_names = []
col_num_name_map = {}
col_num = 1
for i in range(0,len(event_types)):
    for j in range(0, len(event_types)):
        col_names.append(event_types[i]+'@'+event_types[j])
        col_num_name_map[event_types[i]+'@'+event_types[j]] = col_num
        col_num += 1
        print i,j

# generate matrix and write to csv
m = np.full((len(table.keys()),1+len(event_types)**2),-1)
row_names = []
r = 0
for k,v in table.iteritems():
    m[r,0] = int(r)
    row_names.append(str(k))
    for i in range(0,len(v)):
        for j in range(i+1, len(v)):
            if(v[i] == v[j]):
                continue
            else:
                m[r,col_num_name_map[v[i][1]+'@'+v[j][1]]] = (v[j][0] - v[i][0]).total_seconds()
#                print "put ",(v[j][0] - v[i][0]).total_seconds()," in cell ",r," ",col_num_name_map[v[i][1]+'@'+v[j][1]]
    r += 1        
np.savetxt("./output.csv", m, delimiter=",")

"""
lk = ""
lv = ""
for k,v in table.iteritems():
    lk = k
    lv = v
"""
# generate well format csv
with open('./final.csv','w') as out_file:
    writer = csv.writer(out_file, delimiter=',')
    writer.writerow(['id']+col_names)
    with open('./output.csv', 'r') as in_file:
        reader = csv.reader(in_file, delimiter=',')
        row_num = 0
        for row in reader:
            row[0] = row_names[row_num] 
            writer.writerow(row)
            row_num += 1

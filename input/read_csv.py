import csv

with open("/home/tengmf/interaction.csv") as cf:
    reader = csv.reader(cf, delimiter=',')
    table = {}
    event_types = []
    for row in reader:
        if(row[0] not in table):
            table[row[0]]=[]
        table[row[0]].append((row[3],row[7]))
        if(row[7] not in event_types):
            event_types.append(row[7])
#    print event_types,'\n',table
    print event_types,'\n',table.keys()

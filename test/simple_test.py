'''
Created on Jan 30, 2015

@author: tengmf
'''
import csv
import os.path as op

matrix_path = op.join("c:","full_matrix")
output_path = op.join("c:","output")
with open(matrix_path, "r") as rf:
    with open(output_path) as wf:
        count_minus_1 = count_0 = count_non_0 = 0
        reader = csv.reader(rf, delimiter = ",")
        writer = csv.writer(wf, delimiter = ",")
        for row in reader:
            cId = row[0]
            row_sum = 0
            row_positive_count = 0
            for i in range(1, len(row)):
                value = float(i)
                if(i == -1.):
                    count_minus_1 += 1
                elif(i == 0.):
                    count_0 += 1
                else:
                    count_non_0 += 1
                    row_positive_count += 1
                    row_sum += i
            if(row_sum > 0):
                avg = row_sum/row_positive_count
            else:
                avg = 0
            writer.write([cId, avg])


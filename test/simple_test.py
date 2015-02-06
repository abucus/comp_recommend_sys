'''
Created on Jan 30, 2015

@author: tengmf
'''
import csv
import os.path as op
import matplotlib.pyplot as plt

matrix_path = op.join("e:\\","full_matrix.csv")
stat_output_path = op.join("e:\\","stat.txt")
with open(matrix_path, "r") as rf:
    with open(stat_output_path, "w") as wf:
        count_minus_1 = count_0 = count_non_0 = 0
        avg_list = []
        reader = csv.reader(rf, delimiter = ",")
        reader.next()
        writer = csv.writer(wf, delimiter = ",")
        for row in reader:
            cId = row[0]
            row_sum = 0
            row_positive_count = 0
            for i in range(1, len(row)):
                value = float(row[i])
                if(value < 0.):
                    count_minus_1 += 1
                elif(value < 1.):
                    count_0 += 1
                else:
                    count_non_0 += 1
                    row_positive_count += 1
                    row_sum += value/3600./24.
            if(row_sum > 0):
                avg = row_sum*1./row_positive_count
            else:
                avg = 0
            writer.writerow([cId, avg])
            avg_list.append(avg)
        print "-1:",count_minus_1
        print " 0:",count_0
        print " +:",count_non_0
        print " total_customer:",len(avg_list)
        plt.hist(avg_list)
        plt.show()

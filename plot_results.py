import matplotlib.pyplot as plt
import csv

generation = []
avg = []
max = []
min = []

with open('data.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        generation.append(row[0])
        avg.append(float(row[1]))
        max.append(float(row[2]))
        min.append(float(row[3]))

plt.plot(generation, max, color='b', linestyle='dashed',
         marker='o', label="Max")
plt.plot(generation, avg, color='g', linestyle='dashed',
         marker='o', label="Average")
plt.plot(generation, min, color='r', linestyle='dashed',
         marker='o', label="Min")

plt.xticks(rotation=25)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.grid()
plt.show()

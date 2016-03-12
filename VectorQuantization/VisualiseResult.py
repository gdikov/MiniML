

sample_points = []
with open("testInput_6clusters_unknown.txt", "r") as f:
	lines = [line[:-1] for line in f]
	# uncomment in the data contains a line with the number of the clusters
	# lines = lines[1:]
	lines = [x.split(',') for x in lines]
	for line in lines:
		sample_points.append((float(line[0]), float(line[1])))

xs1 = [p[0] for p in sample_points]
ys1 = [p[1] for p in sample_points]

xt1 = []
yt1 = []
with open("testOutput_6clusters_unknown.txt", "r") as f:
	lines = [line[:-1] for line in f]
	lines = [x.split(',') for x in lines]
	for line in lines:	
		xt1.append(float(line[0]))
		yt1.append(float(line[1]))

import matplotlib.pyplot as plt 

tp1, = plt.plot(xs1, ys1, linestyle='', marker='o', color='b', markersize=5)


trp1, = plt.plot(xt1, yt1, linestyle='', marker='*', color='r', markersize=12)

# plt.legend([tp1, trp1], ['Cities', 'Tour'])

plt.show()










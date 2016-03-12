

cities = []
with open("testInput_world.txt", "r") as f:
	lines = [line[:-1] for line in f]
	lines = [x.split(',') for x in lines]
	for line in lines:
		cities.append((int(line[1]), int(line[2])))

xs1 = [p[0] for p in cities]
ys1 = [p[1] for p in cities]

xt1 = []
yt1 = []
with open("testOutput_world.txt", "r") as f:
	lines = [line[:-1] for line in f]
	for line in lines:
		xt1.append(cities[int(line)-1][0])
		yt1.append(cities[int(line)-1][1])

import matplotlib.pyplot as plt 

tp1, = plt.plot(xs1, ys1, linestyle='', marker='o', color='b', markersize=10)


trp1, = plt.plot(xt1, yt1, linestyle='-', marker='o', color='r', markersize=5)

# plt.legend([tp1, trp1], ['Cities', 'Tour'])

plt.show()










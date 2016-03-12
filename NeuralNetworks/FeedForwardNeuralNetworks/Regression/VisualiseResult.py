

points_training = []
points_testing = []
testing = False
with open("testInput.txt", "r") as f:
	lines = [line[:-1] for line in f]
	lines = [x.split(',') for x in lines]
	for line in lines:
		if line == ['0', '0']:
			testing = True
			continue
		if not testing:	
			points_training.append((float(line[0]), float(line[1])))
		else:
			points_testing.append(float(line[0]))

xs1 = [p[0] for p in points_training]
ys1 = [p[1] for p in points_training]

xt1 = []
yt1 = []
with open("testOutput.txt", "r") as f:
	lines = [line[:-1] for line in f]
	for line, index in zip(lines, range(0, len(lines))):
		xt1.append(points_testing[index])
		yt1.append(float(line))

import matplotlib.pyplot as plt 

tp1, = plt.plot(xs1, ys1, linestyle='', marker='o', color='b', markersize=4)


trp1, = plt.plot(xt1, yt1, linestyle='', marker='*', color='r', markersize=15)

plt.legend([tp1, trp1], ['Training points', 'Testing points'])

plt.show()










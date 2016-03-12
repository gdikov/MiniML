

points_training = []
points_testing = []
testing = False
with open("testInput.txt", "r") as f:
	lines = [line[:-1] for line in f]
	lines = [x.split(',') for x in lines]
	for line in lines:
		if line == ['0', '0', '0']:
			testing = True
			continue
		if not testing:	
			points_training.append((float(line[0]), float(line[1]), int(line[2])))
		else:
			points_testing.append((float(line[0]), float(line[1])))

xs1 = [p[0] for p in points_training if p[2]==1]
xs2 = [p[0] for p in points_training if p[2]==-1]
ys1 = [p[1] for p in points_training if p[2]==1]
ys2 = [p[1] for p in points_training if p[2]==-1]

xt1 = []
xt2 = []
yt1 = []
yt2 = []
with open("testOutput.txt", "r") as f:
	lines = [line[:-1] for line in f]
	for line, index in zip(lines, range(0, len(lines))):
		if int(line) == 1:
			xt1.append(points_testing[index][0])
			yt1.append(points_testing[index][1])
		else:
			xt2.append(points_testing[index][0])	
			yt2.append(points_testing[index][1])

import matplotlib.pyplot as plt 

tp1, = plt.plot(xs1, ys1, linestyle='', marker='o', color='r', markersize=4)
tp2, = plt.plot(xs2, ys2, linestyle='', marker='o', color='b', markersize=4)


trp1, = plt.plot(xt1, yt1, linestyle='', marker='*', color='r', markersize=20)
trp2, = plt.plot(xt2, yt2, linestyle='', marker='*', color='b', markersize=20)

plt.legend([tp1, tp2, trp1, trp2], ['Training point +1', 'Training point -1', 'Testing point +1', 'Testing point -1'])

plt.show()










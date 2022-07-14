from statistics import mean, stdev
import os

means = []
nodes = {}
for f in os.listdir('../logs/'):
    if f.startswith('job_') and f.split('.')[1][0] == 'o':
        times = []
        print('Logfile: ' + f)
        with open('../logs/' + f) as infile:
            for line in infile:
                if "Modelling was successfull and took" in line:
                    times.append(float(line.split(' ')[5]))
        try:
            m = mean(times)
            print('Mean time: ' + str(m))
            means.append(m)
            try:
                nodes[f.split('.')[0].split('_')[1]].append(m)
            except:
                nodes[f.split('.')[0].split('_')[1]] = [m]
        except:
            print('NO TIMES')
        print('')

print('OVERALL MEAN: %i' %mean(means))
for node in nodes:
    print('MEAN FOR %s: %i' %(node, mean(nodes[node])))

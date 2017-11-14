import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle


restDir = ''
thres = 600
sim_data = pickle.load(open(restDir+'simres_'+str(thres)+'.p', "rb"))

lw_val = 3
ls_val = ['-', '--', ':', '-', '--', ':']

consumptionList = sim_data['consumptionList']
per5List = sim_data['per5List']
controlList = sim_data['controlList']
policies = sim_data['policies']
trafficPattern = sim_data['trafficPattern']
thres = sim_data['thres']
nHourPerDay = 24

xAxis = np.arange(0, nHourPerDay)

plt.figure(1)
for i in range(len(policies)):
    plt.plot(xAxis, consumptionList[i], ls=ls_val[i], lw=lw_val, label=policies[i])
# plt.axis([0, 23, 1455, 1500])
_, _, y1, y2 = plt.axis()
plt.axis((0, 23, y1, y2))
plt.ylabel('Consumption (W)')
plt.xlabel('Hour')
plt.grid(True)
plt.legend(loc="best")
plt.savefig(restDir+'consumption.eps', bbox_inches='tight')


plt.figure(2)
for i in range(len(policies)):
    plt.plot(xAxis, per5List[i]/1e3, ls=ls_val[i], lw=lw_val, label=policies[i])
plt.plot(xAxis, np.ones(nHourPerDay)*thres/1e3, ls=':', lw=lw_val, label='$Q_{min}$')
# plt.axis([0, 23, 1455, 1500])
_, _, y1, y2 = plt.axis()
plt.axis((0, 23, y1, y2))
plt.ylabel('Ratio of UEs satifying QoS')
plt.xlabel('Hour')
plt.grid(True)
plt.legend(loc="upper left")
plt.savefig(restDir+'QoS.eps', bbox_inches='tight')



controlVectorList = list()

for i in range(len(policies)):
    controlArray = np.zeros((3, nHourPerDay))
    c = controlList[i]
    for j in range(nHourPerDay):
        controlArray[0, j] = c[j][0]  # nPicos
        controlArray[1, j] = c[j][1]  # ABS
        controlArray[2, j] = c[j][2]  # CRE
    controlVectorList.append(controlArray)


fig = plt.figure(3)
sub1 = fig.add_subplot(311)
for i in range(len(policies)):
    sub1.plot(xAxis, controlVectorList[i][0, :], ls=ls_val[i], lw=lw_val, label=policies[i])
# sub1.legend(loc="upper left")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
_, _, y1, y2 = plt.axis()
plt.axis((0, 23, y1-1, y2+1))
plt.ylabel('$p\'$')
plt.xlabel('Hour')
plt.grid(True)

sub1 = fig.add_subplot(312)
for i in range(len(policies)):
    sub1.plot(xAxis, controlVectorList[i][1, :], ls=ls_val[i], lw=lw_val, label=policies[i])
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3, ncol=3, mode="expand", borderaxespad=0.)
_, _, y1, y2 = plt.axis()
plt.axis((0, 23, y1-.1, y2+.1))
plt.ylabel('ABS ratio ($\gamma$)')
plt.xlabel('Hour')
plt.grid(True)

sub1 = fig.add_subplot(313)
for i in range(len(policies)):
    sub1.plot(xAxis, controlVectorList[i][2, :], ls=ls_val[i], lw=lw_val, label=policies[i])
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3, ncol=3, mode="expand", borderaxespad=0.)
_, _, y1, y2 = plt.axis()
plt.axis((0, 23, y1-1, y2+1))
plt.ylabel('CRE bias ($\phi$)')
plt.xlabel('Hour')
plt.grid(True)
plt.savefig(restDir+'paramConf.eps', bbox_inches='tight')


plt.figure(4)
trafficPatternAxis = np.linspace(0, 24, 1440)
plt.plot(trafficPatternAxis, np.random.poisson(trafficPattern*10000-150), lw=1, label='$\lambda$')
plt.plot(trafficPatternAxis, trafficPattern*10000-150, lw=1, c='k', label=r"$\bar{\lambda}$")
plt.axis([0, 23, 0, 1500])
plt.ylabel('Traffic Intensity (UEs per sector)')
plt.xlabel('Hour')
# plt.legend(loc="best")
plt.grid(True)
plt.savefig(restDir+'trafficPattern.eps', bbox_inches='tight')

plt.show()

# for i in range(len(policies)):
#     print(policies[i])
#     print(np.sum(consumptionList[i] * 3600))
#     print(np.sum(per5List[i] > thres) / 24)
#
# print(policies[0])
# print(1-np.sum(consumptionList[0] * 3600) / np.sum(consumptionList[2] * 3600))
# print(policies[1])
# print(1-np.sum(consumptionList[1] * 3600) / np.sum(consumptionList[2] * 3600))


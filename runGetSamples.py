import getNetworkSamples as netSamp
import numpy as np
import multiprocessing
import os.path


def func(p):
    return s.getSamples(p)

nPicos = 6
pPico = 2/3

trafficVal = np.round(np.linspace(50, 1500, 40)).astype(int)

absVal = np.array([0, 1, 2, 3, 4, 5, 6, 7])
creVal = np.array([0, 3, 6, 9, 12, 18])
repetitions = 250

# mean is 3233989.10393
QoSthres = 100000

nActivePicosVal = np.arange(nPicos+1)

points = np.array(np.meshgrid(trafficVal, nActivePicosVal, absVal, creVal)).T.reshape(-1,4)
nPoints = len(points[:, 1])  # len(trafficVal) * len(nActivePicosVal) * len(absVal) * len(creVal)

s = netSamp.GetNetworksamples()
s.initialize(nPicos, pPico, repetitions, QoSthres)
pointList = []
for p in range(nPoints):
    pointList.append(points[p, :])

pool = multiprocessing.Pool(processes=40)
samples = pool.map(func, pointList)
pool.close()
pool.join()
print('done')


consumptionSamples = np.zeros((nPoints, repetitions))
per5thrSamples = np.zeros((nPoints, repetitions))
meanThrSamples = np.zeros((nPoints, repetitions))
maxCellUsageSamples = np.zeros((nPoints, repetitions))
percentQoSSamples = np.zeros((nPoints, repetitions))

for i in range(nPoints):
    consumptionSamples[i, :] = samples[i][0]
    per5thrSamples[i, :] = samples[i][1]
    meanThrSamples[i, :] = samples[i][2]
    maxCellUsageSamples[i, :] = samples[i][3]
    percentQoSSamples[i, :] = samples[i][4]

samplesDir = 'netsamples/'
if not os.path.exists(samplesDir):
    os.makedirs(samplesDir)
np.save(samplesDir+'points', points)
np.save(samplesDir+'consumption.npy', consumptionSamples)
np.save(samplesDir+'per5thr.npy', per5thrSamples)
np.save(samplesDir+'meanThr.npy', meanThrSamples)
np.save(samplesDir+'maxCellUsage.npy', maxCellUsageSamples)
np.save(samplesDir+'percentQoS.npy', percentQoSSamples)

import os.path
import network_sim as netsim
import numpy as np
import time
import pickle
from dynamic_programming import dynamic_programming_alg as dynprog

nPicos = 6
pPico = 2/3

nSimulatedDays = 1
nHourPerDay = 24

vConsumption = np.zeros((nSimulatedDays, nHourPerDay))
vPer5thr = np.zeros((nSimulatedDays, nHourPerDay))

thres = 680
model_path = 'classification/models/'
model_file = 'models_con_qos'+str(thres)
offline_file = 'offline_phase_data'
dp = dynprog.DynamicProgramming(model_path, model_file, offline_file)

policies = ['ADP ES IC', 'ADP ES', 'Default config.']
consumptionList = list()
per5List = list()
controlList = list()
initHour = 22
initialization_hours = np.arange(initHour, nHourPerDay)
initControl = np.array([nPicos, .125, 6])

QoSthres = 100000

hoursVector = np.append(initialization_hours, np.tile(np.arange(nHourPerDay), (nSimulatedDays, 1)))

for p in policies:

    print('Policy: '+p)
    ns = netsim.NetworkSim(nPicos, pPico, initHour, QoSthres)

    currentControl = initControl.copy()
    controls = list()
    d = -1

    for k in hoursVector:
        con, per, measuredTraffic, thr, cu, qos, rt = ns.simulateHour(currentControl[0], currentControl[1::])
        performanceMetric = qos * 1e3

        traffic = rt

        if p == policies[0]:  # Dynamic programming
            nextControl = dp.online_getNextControl(traffic, currentControl, np.mod(k+1, nHourPerDay))
        elif p == policies[1]:
            nextControl = dp.online_getNextControl_noeICIC(traffic, currentControl, np.mod(k+1, nHourPerDay))
        elif p == policies[2]:
            nextControl = np.array([nPicos, 0, 0])



        if d >= 0:
            print('Hour '+str(k))
            print('Next control: '+str(nextControl))
            controls.append(currentControl)
            vConsumption[d, k] = con
            vPer5thr[d, k] = performanceMetric

        if k == nHourPerDay-1:
            d += 1
        currentControl = nextControl

    consumptionList.append(np.mean(vConsumption, axis=0))
    per5List.append(np.mean(vPer5thr, axis=0))
    controlList.append(controls)

dp.closeAlgorithm()

trafficPattern = ns.getTrafficPattern()

sim_params = dict(nPicos=nPicos, pPico=pPico, nSimulatedDays=nSimulatedDays)
sim_res = dict(sim_params=sim_params, consumptionList=consumptionList, per5List=per5List, controlList=controlList, trafficPattern=trafficPattern, policies=policies, thres=thres)

restDir = 'results/'
# filename = time.strftime("%y%m%d%H%M")+'_percentQoS'+str(thres)+'_simres.p'
filename = 'simres_'+str(thres)+'.p'
if not os.path.exists(restDir):
    os.makedirs(restDir)
pickle.dump(sim_res, open(restDir+filename, "wb"))
print("Model saved in file: %s" % filename)





import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from dynamic_programming import neural_classif_usage as nn


class DynamicProgramming:
    def __init__(self, model_path, model_file, offline_file):

        self.model_path = model_path
        self.model_file = model_file
        self.offline_file = offline_file
        self.nControls = None
        self.activationCost = None
        self.nn = None
        self.nStages = 24
        self.regr = joblib.load(model_path+'regr_model.pkl')
        # if os.path.isfile(offline_file+'.p'):
        #     offline_phase = pickle.load(open(offline_file+'.p', "rb"))
        #     self.controlSpace = offline_phase['controlSpace']
        #     self.nControls = offline_phase['nControls']
        #     self.J = offline_phase['J']
        #     self.activationCost = offline_phase['activationCost']
        # else:
        #     self.offline_phase()
        self.offline_phase()

    def offline_phase(self):
        print('Executing offline phase.')
        UEgenerationFactor = 10000
        UEgenerationbias = 150
        averageTrafficPerHour = - UEgenerationbias + UEgenerationFactor * np.array([0.025, 0.07, 0.085, 0.0964, 0.1, 0.105, 0.11, 0.115, 0.12, 0.1225, 0.125, 0.13, 0.14, 0.16, 0.155, 0.15, 0.125, 0.1, 0.075, 0.05, 0.03, 0.024, 0.024, 0.027])
        nPicos = 6
        nActivePicosVal = np.arange(0, (nPicos+1))
        nABSvals = 8
        ABSval = np.linspace(0, 7/8, nABSvals)
        CREval = np.array([0, 6, 9, 12, 18])

        controlSpace = np.array(np.meshgrid(nActivePicosVal, ABSval, CREval)).T.reshape(-1,3)
        nControls = len(controlSpace[:, 0])

        ABS_CRE_bench = np.array([0, 0])
        nControls_bench = nPicos + 1
        controlSpace_bench = np.append(np.expand_dims(nActivePicosVal, axis=0).transpose(), np.tile(ABS_CRE_bench, (nControls_bench, 1)), axis=1)

        C = np.zeros((nControls, self.nStages))
        P = np.zeros((nControls, self.nStages))

        self.nn = nn.Classification(self.model_path, self.model_file)

        print('Creating matrices C and P...')
        for k in range(self.nStages):
            for s in range(nControls):
                state = np.concatenate([np.array([averageTrafficPerHour[k]]), controlSpace[s, :]])
                C[s, k] = self.nn.getConsumption(np.expand_dims(state, axis=0))[0]
                P[s, k] = self.nn.getQoS(np.expand_dims(state, axis=0))[0]


        J = np.zeros((nControls, self.nStages))

        p0Pico = 56
        beta = .1
        self.activationCost = beta * p0Pico

        print('Computing matrix J...')
        stageLoop = np.arange(0, self.nStages)[::-1]
        for k in stageLoop:

            if k == stageLoop[0]:  # if last stage
                costToGo = 0
                validActions = P[:, 0] > 0
                if sum(validActions) == 0:
                    validActions[np.argmax(P[:, 0])] = True
            else:
                costToGo = J[:, k+1]
                validActions = P[:, k+1] > 0
                if sum(validActions) == 0:
                    validActions[np.argmax(P[:, 0])] = True

            for s in range(nControls):

                currentActivePicos = controlSpace[s, 0]
                activationMatrix = np.maximum(controlSpace[:, 0] - currentActivePicos, 0)
                activationCostMatrix = activationMatrix * self.activationCost

                cost_in_k = C[:, k] + activationCostMatrix + costToGo
                J[s, k] = np.min(cost_in_k[validActions])

        self.J = J
        self.controlSpace = controlSpace
        self.nControls = nControls


        #offline phase for benchmark algorithm

        C = np.zeros((nControls_bench, self.nStages))
        P = np.zeros((nControls_bench, self.nStages))

        print('Creating matrices C and P for benchmark algorithm...')
        for k in range(self.nStages):
            for s in range(nControls_bench):
                state = np.concatenate([np.array([averageTrafficPerHour[k]]), controlSpace_bench[s, :]])
                C[s, k] = self.nn.getConsumption(np.expand_dims(state, axis=0))[0]
                P[s, k] = self.nn.getQoS(np.expand_dims(state, axis=0))[0]


        J_bench = np.zeros((nControls_bench, self.nStages))

        p0Pico = 56
        beta = .1
        self.activationCost = beta * p0Pico

        print('Computing matrix J for benchmark algorithm...')
        stageLoop = np.arange(0, self.nStages)[::-1]
        for k in stageLoop:

            if k == stageLoop[0]:  # if last stage
                costToGo = 0
                validActions = P[:, 0] > 0
                if sum(validActions) == 0:
                    validActions[np.argmax(P[:, 0])] = True
            else:
                costToGo = J_bench[:, k+1]
                validActions = P[:, k+1] > 0
                if sum(validActions) == 0:
                    validActions[np.argmax(P[:, 0])] = True

            for s in range(nControls_bench):

                currentActivePicos = controlSpace_bench[s, 0]
                activationMatrix = np.maximum(controlSpace_bench[:, 0] - currentActivePicos, 0)
                activationCostMatrix = activationMatrix * self.activationCost

                cost_in_k = C[:, k] + activationCostMatrix + costToGo
                J_bench[s, k] = np.min(cost_in_k[validActions])

        self.controlSpace_bench = controlSpace_bench
        self.nControls_bench = nControls_bench
        self.J_bench = J_bench
        # offline_phase = dict(controlSpace=controlSpace, nControls=nControls, J=J, activationCost=self.activationCost)
        # pickle.dump(offline_phase, open(self.offline_file+'.p', "wb"))

        print('Offline phase finished.')


    def online_getNextControl(self, traffic, currentControl, k): #regresion and NN

        if self.nn is None:
            self.nn = nn.Classification(self.model_path, self.model_file)

        states = np.append(np.tile(traffic, (self.nControls, 1)), self.controlSpace.copy(), axis=1)
        states[:, 2] = states[:, 2] * 8
        consumption_next_stage = self.regr.predict(states)
        QoS_class = self.nn.getQoS(states).transpose()[0]
        validActions = QoS_class > 0
        if not validActions.any():
            validActions[np.argmax(QoS_class)] = True
            print('There is no action satisfying the threshold!')

        currentActivePicos = currentControl[0]
        activationMatrix = np.maximum(self.controlSpace[:, 0] - currentActivePicos, 0)
        activationCostMatrix = activationMatrix * self.activationCost

        totalConsumption = consumption_next_stage + activationCostMatrix + self.J[:, np.mod(k+1, self.nStages)]
        totalConsumption[np.logical_not(validActions)] = np.inf

        control = self.controlSpace[np.argmin(totalConsumption), :]
        if control[0] == 0: #if 0 active picos, ABS and CRE deactivated
            control[2] = 0
        return control


    def online_getNextControl_noeICIC(self, traffic, currentControl, k): #no eICIC regr and NN

        if self.nn is None:
            self.nn = nn.Classification(self.model_path, self.model_file)

        states = np.append(np.tile(traffic, (self.nControls_bench, 1)), self.controlSpace_bench.copy(), axis=1)
        consumption_next_stage = self.regr.predict(states)
        QoS_class = self.nn.getQoS(states).transpose()[0]
        validActions = QoS_class > 0
        if not validActions.any():
            validActions[np.argmax(QoS_class)] = True
            print('There is no action satisfying the threshold!')

        currentActivePicos = currentControl[0]
        activationMatrix = np.maximum(self.controlSpace_bench[:, 0] - currentActivePicos, 0)
        activationCostMatrix = activationMatrix * self.activationCost

        totalConsumption = consumption_next_stage + activationCostMatrix + self.J_bench[:, np.mod(k+1, self.nStages)]
        totalConsumption[np.logical_not(validActions)] = np.inf

        control = self.controlSpace_bench[np.argmin(totalConsumption), :]
        return control


    def closeAlgorithm(self):
        self.nn.closeModel()

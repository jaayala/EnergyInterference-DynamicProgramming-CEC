import os.path
import numpy as np
import common as f


class NetworkSim:
    def __init__(self, nPicos, picoProbability, initHour, QoSthres, load=1):
        ISD = 500
        VISD = np.sqrt(3) * ISD
        self.apothem = np.sqrt(3)/6 * ISD
        self.maxUEperSector = 2000
        self.nInterferingMacros = 4
        self.macroPos = np.array([[0, VISD/2], [ISD/2, 0], [ISD/2, VISD], [ISD, VISD/2]])
        self.sectorCenter = self.macroPos[0] + np.array([ISD/3, 0])

        dataDir = 'netdata/'
        if load == 0 or (not os.path.isfile(dataDir+str(nPicos))):
            picoPos = f.picoCellGeneration(self.apothem, nPicos, self.sectorCenter, self.macroPos)
            self.picoPos = picoPos
            np.save(dataDir+str(nPicos), picoPos)
        else:
            self.picoPos = np.load(dataDir+str(nPicos))

        self.nPicos = nPicos
        self.probPico_0 = picoProbability
        self.fileLenghtBits = 8000

        # Pico Values TR 36.887
        self.macroPower = 10**((43 - 30)/10) # 43 dBm --> 13 dBW --> 20 W
        self.picoPower = 6.3 # 6.3 W

        self.nSubframes= 8
        self.crsProportion = .1
        self.subframeDuration = 1e-3
        self.nUsedSubframes = 0
        self.framesPerMin = 100
        self.UEgenerationFactor = 10000
        self.UEgenerationbias = 150
        self.minPerHour = 60
        self.hourPerDay = 24
        self.lastPicoControl = np.ones(self.nPicos)

        F = 10**0.5  # noise figure = 5 dB
        T_O = 290
        K_B = 1.3806504e-23
        BW = 200000
        N_O = F*T_O*K_B
        self.thermalNoise = N_O*BW  # thermal noise at the receiver
        self.W = 10e6  # Channel bandwidth (10 MHz)


        # Daily traffic generation
        # self.averageTrafficPerHour = np.array([0.04, 0.07, 0.085, 0.0964, 0.1, 0.105, 0.11, 0.115, 0.12, 0.1225, 0.125, 0.13, 0.14, 0.16, 0.155, 0.15, 0.125, 0.1, 0.075, 0.05, 0.03, 0.024, 0.024, 0.027])
        self.averageTrafficPerHour = np.array([0.025, 0.07, 0.085, 0.0964, 0.1, 0.105, 0.11, 0.115, 0.12, 0.1225, 0.125, 0.13, 0.14, 0.16, 0.155, 0.15, 0.125, 0.1, 0.075, 0.05, 0.03, 0.024, 0.024, 0.027])
        trafficPoly = np.polyfit(np.arange(self.hourPerDay), self.averageTrafficPerHour, 7)
        axis = np.linspace(0, self.hourPerDay-1, self.hourPerDay*self.minPerHour)
        self.averageTrafficPerMinute = np.polyval(trafficPoly, axis)

        self.currentHour = initHour
        self.currentMinute = initHour * self.minPerHour

        self.sortedActivationIndex = f.picoSelection(self)

        self.QoSthres = QoSthres


    def simulateHour(self, picoControl_in, eicicControl):

        if isinstance(picoControl_in, float) or isinstance(picoControl_in, np.int64):  # if only the number of active picos is indicated (len == 1), we use a heuristic to select which ones
            picoControl = np.zeros(self.nPicos)
            picoControl[self.sortedActivationIndex[:int(picoControl_in)]] = 1
        elif len(picoControl_in) == self.nPicos:
            picoControl = picoControl_in
        else:
            raise NameError('Invalid picoControl')

        self.nActivePicos = int(sum(picoControl))
        self.activePicosPos = self.picoPos[picoControl == 1, :]

        self.thrSamples = []
        self.cellUsage = [[] for _ in range(self.nActivePicos+1)]
        self.meanPicoUsage = np.zeros(self.nActivePicos)
        self.meanMacroUsage = 0

        self.meanConsumptionPerCell = np.zeros(self.nPicos+1)
        self.totalMeanConsumption = 0

        self.absRatio = eicicControl[0] * self.nSubframes # abs between 0 and 1 at input
        self.creBias = eicicControl[1] # in dBs
        if self.absRatio > 7 or self.absRatio < 0 or self.creBias > 18 or self.creBias < 0:
            raise NameError('Invalid eicicControl')

        if sum(picoControl) == 0:
            self.probPico = 0
        else:
            self.probPico = self.probPico_0

        measuredTraffic = np.zeros(self.minPerHour)
        nMinutesToAverage = 15

        for minute in range(self.minPerHour):
            self.UEpos = np.ones((self.maxUEperSector, 2)) * -1
            self.UEdata = np.ones(self.maxUEperSector)
            self.UEposPico = [np.ones((self.maxUEperSector, 2))*-1 for _ in range(self.nActivePicos)]
            self.UEdataPico = [np.ones(self.maxUEperSector) for _ in range(self.nActivePicos)]
            self.UEgainsPico = [np.zeros((self.maxUEperSector, self.nInterferingMacros+self.nActivePicos)) for _ in range(self.nActivePicos)]
            self.UEgains = np.zeros((self.maxUEperSector, self.nInterferingMacros+self.nActivePicos))

            currentMeanTraffic = self.averageTrafficPerMinute[self.currentMinute]
            # nNewUsers = np.random.poisson(currentMeanTraffic * self.maxUEperSector)
            nNewUsers = np.random.poisson(currentMeanTraffic * self.UEgenerationFactor - self.UEgenerationbias)
            measuredTraffic[minute] = nNewUsers

            f.generateUEs(nNewUsers, self)
            cellWithTraffic, _ = f.getCellWithTraffic(self)
            measureConsumption = True

            # print('Generated UEs: '+str(nNewUsers))

            self.nUsedSubframes = 1
            for subf in range(self.framesPerMin):  # framesPerMin frames are simulated every minute
                self.macroPowerVector = np.zeros(self.nSubframes)
                nABSsubframes = int(np.floor(self.absRatio) + (np.random.rand() < (self.absRatio - np.floor(self.absRatio))))
                self.macroPowerVector[:(self.nSubframes - nABSsubframes)] = 1
                f.execute_frame(self, np.logical_or(measureConsumption, subf < 5))

                cellWithTraffic_aux, nUsersVector = f.getCellWithTraffic(self)
                measureConsumption = np.logical_not(np.logical_xor(cellWithTraffic, cellWithTraffic_aux)).all() # Detects if a cell has been emptied

                # print('Subframe ' + str(subf) + ' UEs: ' + str(np.sum(nUsersVector)) + ' macroUEs: ' + str(nUsersVector[self.nActivePicos]))

            f.retrieveThroughputSamples(self)

            self.currentMinute = np.mod(self.currentMinute+1, self.minPerHour*self.hourPerDay)  # Circular increment

        f.computeConsumption(self, picoControl)
        self.lastPicoControl = picoControl
        self.currentHour = np.mod(self.currentHour+1, self.hourPerDay)

        per5thr = f.get5percentvalue(self)
        meanThr = f.getMeanThrValue(self)
        percentQoS = f.getQoSpercent(self)
        maxCellUsage = np.max(np.append(self.meanPicoUsage, self.meanMacroUsage))


        realTraffic = np.max(self.averageTrafficPerMinute[self.currentMinute:(self.currentMinute+self.minPerHour)] * self.UEgenerationFactor - self.UEgenerationbias)

        return self.totalMeanConsumption, per5thr, np.mean(measuredTraffic[(self.minPerHour-nMinutesToAverage)::]), meanThr, maxCellUsage, percentQoS, realTraffic

    def getTrafficPattern(self):
        return self.averageTrafficPerMinute

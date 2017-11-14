import numpy as np
import common as f
import os.path


class GetNetworksamples:
    def initialize(self, nPicos, picoProbability, repetitions, QoSthres, load=1):
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
            file = open(dataDir+str(nPicos), "wb")
            np.save(file, picoPos)
        else:
            self.picoPos = np.load(dataDir+str(nPicos))

        self.nPicos = nPicos
        self.probPico_0 = picoProbability
        self.fileLenghtBits = 8000

        #Pico Values TR 36.887
        self.macroPower = 10**((43 - 30)/10) # 43 dBm --> 13 dBW --> 20 W
        self.picoPower = 6.3 # 6.3 W

        self.nSubframes= 8
        self.crsProportion = .1
        self.subframeDuration = 1e-3
        self.nUsedSubframes = 0
        self.framesPerMin = 100
        self.minPerHour = 60
        self.hourPerDay = 24
        self.lastPicoControl = np.ones(self.nPicos)

        F = 10**0.5 # noise figure = 5 dB
        T_O = 290
        K_B = 1.3806504e-23
        BW = 200000
        N_O = F*T_O*K_B
        self.thermalNoise = N_O*BW # thermal noise at the receiver
        self.W = 10e6 # Channel bandwidth (10 MHz)
        self.repetitions = repetitions

        self.QoSthres = QoSthres

    def getSamples(self, point):


        consumptionSamples = np.zeros(self.repetitions)
        per5Samples = np.zeros(self.repetitions)
        meanThrSamples = np.zeros(self.repetitions)
        maxCellUsage = np.zeros(self.repetitions)
        percentQoS = np.zeros(self.repetitions)
        self.meanPicoUsage = np.zeros(point[1])
        self.meanMacroUsage = 0

        sortedActivationIndex = f.picoSelection(self)
        # print(point)
        print('o', end='', flush=True)

        for r in range(self.repetitions):

            traffic = point[0]
            nActivePicos = point[1]
            self.absRatio = point[2]
            self.creBias = point[3]

            self.nActivePicos = nActivePicos
            picoControl = np.zeros(self.nPicos)
            picoControl[sortedActivationIndex[:self.nActivePicos]] = 1
            self.activePicosPos = self.picoPos[picoControl == 1, :]

            if sum(picoControl) == 0:
                self.probPico = 0
            else:
                self.probPico = self.probPico_0

            self.thrSamples = []
            self.cellUsage = [[] for _ in range(self.nActivePicos+1)]
            self.meanConsumptionPerCell = np.zeros(self.nPicos+1)
            self.totalMeanConsumption = 0

            self.UEpos = np.ones((self.maxUEperSector, 2)) * -1
            self.UEdata = np.ones(self.maxUEperSector)
            self.UEposPico = [np.ones((self.maxUEperSector, 2))*-1 for _ in range(self.nActivePicos)]
            self.UEdataPico = [np.ones(self.maxUEperSector) for _ in range(self.nActivePicos)]
            self.UEgainsPico = [np.zeros((self.maxUEperSector, self.nInterferingMacros+self.nActivePicos)) for _ in range(self.nActivePicos)]
            self.UEgains = np.zeros((self.maxUEperSector, self.nInterferingMacros+self.nActivePicos))



            f.generateUEs(traffic, self)
            cellWithTraffic, _ = f.getCellWithTraffic(self)
            measureConsumption = True

            self.nUsedSubframes = 1
            for subf in range(self.framesPerMin):  # framesPerMin frames are simulated every minute
                self.macroPowerVector = np.zeros(self.nSubframes)
                nABSsubframes = int(np.floor(self.absRatio) + (np.random.rand() < (self.absRatio - np.floor(self.absRatio))))
                self.macroPowerVector[:(self.nSubframes - nABSsubframes)] = 1
                f.execute_frame(self, np.logical_or(measureConsumption, subf < 5))

                cellWithTraffic_aux, _ = f.getCellWithTraffic(self)
                measureConsumption = np.logical_not(np.logical_xor(cellWithTraffic, cellWithTraffic_aux)).all() # Detects if a cell has been emptied

            f.retrieveThroughputSamples(self)
            per5 = f.get5percentvalue(self)
            meanThr = f.getMeanThrValue(self)
            mcu = np.max(np.append(self.meanPicoUsage, self.meanMacroUsage))
            pQoS = f.getQoSpercent(self)

            f.computeConsumption(self, picoControl)

            consumptionSamples[r] = self.totalMeanConsumption
            per5Samples[r] = per5
            meanThrSamples[r] = meanThr
            maxCellUsage[r] = mcu
            percentQoS[r] = pQoS

        return consumptionSamples, per5Samples, meanThrSamples, maxCellUsage, percentQoS

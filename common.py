import numpy as np

def picoCellGeneration(apothem, nPicos, sectorCenter, macroPositions):

    picoPos = np.zeros((nPicos, 2))
    pico2picoMinDistance = 40
    pico2macroMinDistance = 75

    for i in range(nPicos):
        while 1:
            distance = np.random.rand() * apothem
            angle = np.random.rand() * 2 * np.pi
            picoPosAux = sectorCenter + np.array([distance*np.cos(angle), distance*np.sin(angle)])

            pico2picoDistance = np.sqrt(np.sum(np.square(np.tile(picoPosAux,(nPicos,1)) - picoPos), axis=1))
            pico2macroDistance = np.sqrt(np.sum(np.square(np.tile(picoPosAux,(len(macroPositions),1)) - macroPositions), axis=1))

            if min(pico2picoDistance) > pico2picoMinDistance and min(pico2macroDistance) > pico2macroMinDistance:
                break
        picoPos[i,:] = picoPosAux
    return picoPos

def generateUEs(nNewUsers, self):
    minDistanceUE2macro = 35
    minDistanceUE2pico = 10
    RadioPicoCoverage = 40

    antennaMacroGain = np.power(10, 14/10) # 14 dBi
    antennaPicoGain = np.power(10, 5/10) # 5 dBi

    shadowingVar = 10  # dB

    newUEPos = np.zeros((nNewUsers, 2))
    newUEgains = np.zeros((nNewUsers, self.nInterferingMacros + self.nActivePicos))

    for u in range(nNewUsers):
        if np.random.random() > self.probPico: # Macro UE generation
            while 1:
                distance = np.random.rand() * self.apothem
                angle = np.random.rand() * 2 * np.pi
                UEPosAux = self.sectorCenter + np.array([distance*np.cos(angle), distance*np.sin(angle)])

                UE2macroDistance = np.sqrt(np.sum(np.square(UEPosAux - self.macroPos[0])))
                UE2picoDistance = np.sqrt(np.sum(np.square(np.tile(UEPosAux,(self.nActivePicos,1)) - self.activePicosPos), axis=1))

                if (UE2macroDistance > minDistanceUE2macro and len(UE2picoDistance) == 0) or (UE2macroDistance > minDistanceUE2macro and min(UE2picoDistance) > minDistanceUE2pico):
                    break
            newUEPos[u,:] = UEPosAux

        else: # Pico UE generation
            selectedPico = np.random.randint(0,self.nActivePicos)
            while 1:
                distance = np.random.random() * (RadioPicoCoverage  - minDistanceUE2pico) + minDistanceUE2pico;
                angle = np.random.rand() * 2 * np.pi
                UEPosAux = self.activePicosPos[selectedPico] + np.array([distance*np.cos(angle), distance*np.sin(angle)])

                UE2macroDistance = np.sqrt(np.sum(np.square(UEPosAux - self.macroPos[0])))
                UE2picoDistance = np.sqrt(np.sum(np.square(np.tile(UEPosAux,(self.nActivePicos,1)) - self.activePicosPos), axis=1))

                if UE2macroDistance > minDistanceUE2macro and min(UE2picoDistance) > minDistanceUE2pico:
                    break
            newUEPos[u, :] = UEPosAux

        #Macro gains
        freeSpaceGain = channelModel(np.tile(UEPosAux, (self.nInterferingMacros, 1)), self.macroPos, 'm2ue')
        antennaGains = antennaGain(UEPosAux, self.macroPos)
        newUEgains[u, 0:self.nInterferingMacros] = np.multiply(freeSpaceGain, antennaGains[0]) * antennaMacroGain

        #Pico gains
        if self.nActivePicos > 0:
            newUEgains[u, self.nInterferingMacros:] = channelModel(np.tile(UEPosAux,(self.nActivePicos, 1)), self.activePicosPos, 'p2ue') * antennaPicoGain

        # Adding shadow fading to gain
        shadowingNoise = shadowingVar * np.random.randn(self.nInterferingMacros + self.nActivePicos)
        newUEgains[u,:] *= np.power(10, shadowingNoise/10)

        macroRSRP = newUEgains[u, 0] * self.macroPower

        if self.nActivePicos > 0:
            bestPicoIndex = np.argmax(newUEgains[u, self.nInterferingMacros:])
            picoRSRP = np.max(newUEgains[u, self.nInterferingMacros:]) * self.picoPower

            if macroRSRP > (picoRSRP * np.power(10, self.creBias/10)):
                associationTo = 0 # association to macro
            else:
                associationTo = 1 # association to pico
        else: # se asocian a la macro
            associationTo = 0 # association to macro

        if associationTo == 0:
            index = np.where(self.UEpos[:, 0]==-1)[0]
            if len(index) == 0:
                raise NameError('The struct is full of UEs. Increase the ''nMaxUE'' variable.')
            self.UEpos[index[0], :] = newUEPos[u, :]
            self.UEgains[index[0], :] = newUEgains[u, :]
            self.UEdata[index[0]] = self.fileLenghtBits
        else:
            index = np.where(self.UEposPico[bestPicoIndex][:, 0]==-1)[0]
            if len(index) == 0:
                raise NameError('The struct is full of UEs. Increase the ''nMaxUE'' variable.')
            self.UEposPico[bestPicoIndex][index[0], :] = newUEPos[u, :]
            self.UEgainsPico[bestPicoIndex][index[0], :] = newUEgains[u, :]
            self.UEdataPico[bestPicoIndex][index[0]] = self.fileLenghtBits


def antennaGain(UEpos, macroPos):
    nInterferingMacro = len(macroPos[:, 0])
    gain_dB = np.zeros((nInterferingMacro, 1))
    phi_3db = 70
    A_m = 25
    referenceVectors = np.array([[166.6667, 0], [-83.3333, 144.3376], [-83.3333, -144.3376]])  # Directions of the directional antennas.

    for i in range(nInterferingMacro):
        evaluationVector = UEpos - macroPos[i, :]

        angles = np.zeros(3)
        for j in range(3):
            # http://en.wikipedia.org/wiki/Cosine_similarity
            theta = np.arccos(np.dot(referenceVectors[j, :], evaluationVector) / (np.linalg.norm(evaluationVector) * np.linalg.norm(referenceVectors[j, :])))
            angles[j] = min(np.rad2deg(theta), 360 - np.rad2deg(theta))
        phi = min(angles)
        gain_dB[i] = - min(12 * np.square(phi/phi_3db), A_m)
    return np.power(10, gain_dB/10)


def channelModel(pos1, pos2, type):
    distance = np.sqrt(np.sum(np.square(pos1 - pos2), axis=1)) / 1000  #in Km

    if type == 'm2ue':
        L = 128.1+37.6 * np.log10(distance)
    elif type == 'p2ue':
        L = 140.71+36.7 * np.log10(distance)
    else:
        raise NameError('Invalid argument')
    gain = np.power(10, -L / 10)
    return gain



def execute_frame(self, measureConsumption):

    cellFrameUsage = np.zeros(self.nActivePicos+1)

    for subf in range(self.nSubframes):

        subFrameMacroPower = self.macroPowerVector[subf] * self.macroPower

        # check whether there is or not UEs in the eNBs
        cellsWithTraffic = np.ones(self.nActivePicos + 1)
        for cell in range(self.nActivePicos):
            if len(np.where(self.UEposPico[cell][:,0] > -1)[0]) == 0:  # there is no pico UEs
                cellsWithTraffic[cell] = 0
        if len(np.where(self.UEpos[:, 0] > -1)[0]) == 0:  # there is no macro UEs
            cellsWithTraffic[self.nActivePicos] = 0

        for p in range(self.nActivePicos + 1):

            if p < self.nActivePicos and cellsWithTraffic[p] == 1: # active pico with traffic case
                maxActiveUE = np.where(self.UEposPico[p][:,0] > -1)[0][-1] + 1
                auxGains = np.copy(self.UEgainsPico[p][:maxActiveUE,:])
                autoindex = self.nInterferingMacros + p
                cellTxPw = self.picoPower
                UEDataBuffer = self.UEdataPico[p][:maxActiveUE]

            elif p >= self.nActivePicos and cellsWithTraffic[p] == 1 and subFrameMacroPower > 0: # macro with traffic in non-ABS subframe
                maxActiveUE = np.where(self.UEpos[:,0] > -1)[0][-1] + 1
                auxGains = np.copy(self.UEgains[:maxActiveUE,:])
                autoindex = 0
                cellTxPw = subFrameMacroPower
                UEDataBuffer = self.UEdata[:maxActiveUE]
            else:
                continue

            # The interference of pico cells without traffic is removed
            for cell in range(self.nActivePicos):
                auxGains[:, self.nInterferingMacros + cell] *= cellsWithTraffic[cell]
            auxGains[:, 0] *= cellsWithTraffic[self.nActivePicos]


            signalGain = np.copy(auxGains[:, autoindex])
            auxGains[:, autoindex] = 0
            cellPowersWithABS = np.concatenate((np.tile(subFrameMacroPower, (maxActiveUE, self.nInterferingMacros)), np.tile(self.picoPower, (maxActiveUE, self.nActivePicos))), axis=1)
            noisePower_data = cellPowersWithABS * auxGains
            cellPowers = np.concatenate((np.tile(self.macroPower, (maxActiveUE, self.nInterferingMacros)), np.tile(self.picoPower, (maxActiveUE, self.nActivePicos))), axis=1)
            noisePower_CRS = cellPowers * auxGains
            noisePower = self.crsProportion * noisePower_CRS + (1-self.crsProportion) * noisePower_data

            sinr_vector = cellTxPw * signalGain / (self.thermalNoise + np.sum(noisePower, axis=1))
            capacity = self.W * np.log2(1 + sinr_vector)

            TransmitedData = (self.fileLenghtBits-UEDataBuffer+1) / self.nUsedSubframes

            PF_index = capacity / TransmitedData

            # Scheduling alternatives here.

            scheduleUEorder = np.argsort(PF_index)[::-1]

            subframeduration_aux = self.subframeDuration

            for ue_ind in range(len(scheduleUEorder)):
                ue = scheduleUEorder[ue_ind]

                if capacity[ue] > 0:
                    transmitedData = subframeduration_aux * capacity[ue]
                    if transmitedData <= UEDataBuffer[ue]:
                        UEDataBuffer[ue] -= transmitedData
                        cellFrameUsage[p] += subframeduration_aux / self.subframeDuration
                        break

                    currentSufUsed = UEDataBuffer[ue] / capacity[ue]
                    subframeduration_aux -= currentSufUsed
                    cellFrameUsage[p] += currentSufUsed / self.subframeDuration

                    self.thrSamples.append(self.fileLenghtBits / (self.nUsedSubframes * self.subframeDuration + currentSufUsed))

                    if p < self.nActivePicos:
                        self.UEposPico[p][ue, :] = -1
                        self.UEgainsPico[p][ue, :] = 0
                        self.UEdataPico[p][ue] = 1
                    else:
                        self.UEpos[ue,:] = -1
                        self.UEgains[ue,:] = 0
                        self.UEdata[ue] = 1

        self.nUsedSubframes += 1
    if measureConsumption:
        for p in range(self.nActivePicos + 1):
            self.cellUsage[p].append(cellFrameUsage[p] / self.nSubframes)


def retrieveThroughputSamples(self):
    for p in range(self.nActivePicos + 1):
        if p < self.nActivePicos:
            activeUEs = self.UEdataPico[p] > 1
            self.thrSamples.extend((self.fileLenghtBits - self.UEdataPico[p][activeUEs]) / (self.nUsedSubframes * self.subframeDuration))
        else:
            activeUEs = self.UEdata > 1
            self.thrSamples.extend((self.fileLenghtBits - self.UEdata[activeUEs]) / (self.nUsedSubframes * self.subframeDuration))


def get5percentvalue(self):
    sorted5persamples = np.sort(self.thrSamples)
    return sorted5persamples[round(len(sorted5persamples) * .05)]


def getMeanThrValue(self):
    return np.mean(self.thrSamples)


def computeConsumption(self, picoControl):
    p0Macro = 130
    pmaxMacro = 20
    p0Pico = 56
    pmaxPico = 6.3
    # psleepMacro = 75
    psleepPico = 39
    ntrxMacro = 6
    ntrxPico = 2
    beta = .5

    allPicosIndex = np.where(picoControl == 1)[0]
    for cell in range(self.nActivePicos):
        self.meanPicoUsage[cell] = np.mean(self.cellUsage[cell])
        self.meanConsumptionPerCell[allPicosIndex[cell]] = ntrxPico * (p0Pico + self.meanPicoUsage[cell] * pmaxPico)

    sleepingPicosIndex = np.where(picoControl == 0)[0]
    self.meanConsumptionPerCell[sleepingPicosIndex] = psleepPico

    self.meanMacroUsage = np.mean(self.cellUsage[self.nActivePicos])
    self.meanConsumptionPerCell[self.nPicos] = ntrxMacro * (p0Macro + self.meanMacroUsage * pmaxMacro)

    activationIndex = np.where((picoControl-self.lastPicoControl) == 1)
    self.meanConsumptionPerCell[activationIndex] += beta * p0Pico

    self.totalMeanConsumption = sum(self.meanConsumptionPerCell)



def picoSelection(self):  # Heuristic to switch off the pico cells

    macroWeight = .4
    picoWeight = 1 - macroWeight
    heurVal = np.zeros(self.nPicos)

    for p in range(self.nPicos):
        picoPosAux = np.tile(self.picoPos[p], (self.nPicos, 1))
        distance2picos = np.sqrt(np.sum(np.square(picoPosAux - self.picoPos), axis=1)) / 1000
        distance2macro = np.sqrt(np.sum(np.square(self.picoPos[p] - self.macroPos[0, :]), axis=0)) / 1000
        heurVal[p] = macroWeight * distance2macro + picoWeight * np.sum(distance2picos)

    return np.argsort(heurVal)


def getQoSpercent(self):
    return np.sum(np.asarray(self.thrSamples) >= self.QoSthres) / len(self.thrSamples)


def getCellWithTraffic(self):
    cellWithTraffic = np.zeros(self.nActivePicos + 1)
    for i in range(self.nActivePicos):
        cellWithTraffic[i] = np.sum(self.UEposPico[i][:, 0] > 0)
    cellWithTraffic[self.nActivePicos] = np.sum(self.UEpos[:, 0] > 0)

    return cellWithTraffic > 0, cellWithTraffic

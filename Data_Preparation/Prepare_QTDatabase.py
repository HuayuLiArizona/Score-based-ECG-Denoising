import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle


def prepare(QTpath='data/qt-database-1.0.0/'):
    # Desired sampling frecuency
    newFs = 360

    # Preprocessing signals
    namesPath = glob.glob(QTpath + "/*.dat")

    # final list that will contain all signals and beats processed
    QTDatabaseSignals = dict()

    register_name = None
    for i in namesPath:

        # reading signals
        aux = i.split('.dat')
        register_name = aux[0].split('/')[-1]
        signal, fields = wfdb.rdsamp(aux[0])
        qu = len(signal)

        # reading annotations
        ann = wfdb.rdann(aux[0], 'pu1')
        anntype = ann.symbol
        annSamples = ann.sample

        # Obtaining P wave start positions
        Anntype = np.array(anntype)
        idx = Anntype == 'p'
        Pidx = annSamples[idx]
        idxS = Anntype == '('
        Sidx = annSamples[idxS]
        idxR = Anntype == 'N'
        Ridx = annSamples[idxR]

        ind = np.zeros(len(Pidx))

        for j in range(len(Pidx)):
            arr = np.where(Pidx[j] > Sidx)
            arr = arr[0]
            ind[j] = arr[-1]

        ind = ind.astype(np.int64)
        Pstart = Sidx[ind]

        # Shift 40ms before P wave start
        Pstart = Pstart - int(0.04*fields['fs'])

        # Extract first channel
        auxSig = signal[0:qu, 0]
        
        beats = list()
        for k in range(len(Pstart)-1):
            remove = (Ridx > Pstart[k]) & (Ridx < Pstart[k+1])
            if np.sum(remove) < 2:
                beats.append(auxSig[Pstart[k]:Pstart[k+1]])

        # Creating the list that will contain each beat per signal
        beatsRe = list()

        # processing each beat
        for k in range(len(beats)):
            # Padding data to avoid edge effects caused by resample
            L = math.ceil(len(beats[k])*newFs/fields['fs'])
            normBeat = list(reversed(beats[k])) + list(beats[k]) + list(reversed(beats[k]))

            # resample beat by beat and saving it
            res = resample_poly(normBeat, newFs, fields['fs'])
            res = res[L-1:2*L-1]
            beatsRe.append(res)

        # storing all beats in each corresponding signal, list of list
        QTDatabaseSignals[register_name] = beatsRe

    # Save Data
    with open('data/QTDatabase.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(QTDatabaseSignals, output)
    print('=========================================================')
    print('MIT QT database saved as pickle file')
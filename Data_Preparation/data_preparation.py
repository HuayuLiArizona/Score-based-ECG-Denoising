import numpy as np
import _pickle as pickle
from Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB

def Data_Preparation(noise_version=1):

    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/NoiseBWL.pkl', 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals,_,_] = nstdb
    #[_, em_signals, _ ] = nstdb
    #[_, _, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    

    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]



    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        noise_test = bw_noise_channel2_b
        noise_train = bw_noise_channel1_a
    elif noise_version == 2:
        noise_test = bw_noise_channel1_b
        noise_train = bw_noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")

    #####################################
    # QTDatabase
    #####################################

    beats_train = []
    beats_test = []
    
    '''
    test_set = ['qt-database-1.0.0/sel123',  # Record from MIT-BIH Arrhythmia Database
                'qt-database-1.0.0/sel233',  # Record from MIT-BIH Arrhythmia Database

                'qt-database-1.0.0/sel302',  # Record from MIT-BIH ST Change Database
                'qt-database-1.0.0/sel307',  # Record from MIT-BIH ST Change Database

                'qt-database-1.0.0/sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'qt-database-1.0.0/sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database

                'qt-database-1.0.0/sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'qt-database-1.0.0/sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database

                'qt-database-1.0.0/sele0106',  # Record from European ST-T Database
                'qt-database-1.0.0/sele0121',  # Record from European ST-T Database

                'qt-database-1.0.0/sel32',  # Record from ``sudden death'' patients from BIH
                'qt-database-1.0.0/sel49',  # Record from ``sudden death'' patients from BIH

                'qt-database-1.0.0/sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'qt-database-1.0.0/sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]
    '''
    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database

                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database

                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database

                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database

                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database

                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH

                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]
    
    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)
    
    sn_train = []
    sn_test = []

    noise_index = 0
    
    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    #rnd_test = np.random.randint(low=150, high=200, size=len(beats_test)) / 100

    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save('rnd_test.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        
        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
            noise_index = 0


    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    
    X_test = np.array(sn_test)
    y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)


    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset
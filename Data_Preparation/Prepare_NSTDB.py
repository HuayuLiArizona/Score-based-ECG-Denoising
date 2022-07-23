import numpy as np
import wfdb
import _pickle as pickle

def prepare(NSTDBPath='data/mit-bih-noise-stress-test-database-1.0.0/'):
    bw_signals, bw_fields = wfdb.rdsamp(NSTDBPath + 'bw')
    em_signals, em_fields = wfdb.rdsamp(NSTDBPath + 'em')
    ma_signals, ma_fields = wfdb.rdsamp(NSTDBPath + 'ma')

    for key in bw_fields:
        print(key, bw_fields[key])

    for key in em_fields:
        print(key, em_fields[key])
        
    for key in ma_fields:
        print(key, ma_fields[key])    
    
    # Save Data
    with open('data/NoiseBWL.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump([bw_signals, em_signals, ma_signals], output)
    print('=========================================================')
    print('MIT BIH data noise stress test database (NSTDB) saved as pickle')
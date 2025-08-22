from coniferest.isoforest import IsolationForest
from coniferest.aadforest import AADForest
from coniferest.label import Label
from coniferest.session import Session
from coniferest.session.callback import (
    TerminateAfter, prompt_decision_callback,
)

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from progressbar import progressbar
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor


# read data
data = pd.read_parquet('/media/snad/data/gz2/astronomaly.parquet')

# remove columns related to ID and anomaly tag
data_use = data[list(data.keys())[:-2]]

# set up to use already known labels
def my_on_refit_callback(session):
    #print('Refitting model with known labels:')
    #print(session.known_labels)
    return None

def my_decision_callback(metadata, data, session):
    """Get data from the known table."""
    return Label.ANOMALY if metadata[1] else Label.REGULAR

def my_on_decision_callback(metadata, data, session):
    #print(f'Decision made for {metadata}: {session.last_decision}.')
    return None

class RecordCallback:
    def __init__(self):
        self.records = []

    def __call__(self, metadata, data, session):
        self.records.append(f'{metadata} -> {session.last_decision}')

    def print_report(self):
        #print('Records:')
        #print('\n'.join(self.records))
        return None

# do it 200 times
n_cores = 20

def aad_run(data, metadata, seed):
    """Train, evaluate and save results from the model."""
    
    model_temp_aad = AADForest(random_seed=seed)  
    record_callback_temp_aad = RecordCallback()

    session_temp_aad = Session(
        data=data.values,
        metadata=metadata,
        model=model_temp_aad,
        decision_callback=my_decision_callback,
        # We can give an only function/callable as a callback
        on_refit_callbacks=my_on_refit_callback,
        # Or a list of callables
        on_decision_callbacks=[
            my_on_decision_callback,
            record_callback_temp_aad,
            TerminateAfter(nloops),
        ],
    )
    session_temp_aad.run()
    labels_temp_aad = [list(session_temp_aad.known_labels.values())[i] == Label.ANOMALY 
                        for i in range(nloops)]
    y_temp_aad = [sum(labels_temp_aad(:i]) for i in range(len(labels_temp_aad))]

    return y_temp_aad

result =[]
seeds = [np.random.randint(10**9) for i in range(n)]

with ProcessPoolExecutor(max_workers=n_cores) as exe:
    exe.submit(aad_run, data_use, metadata)
        
    # Maps the realization with a iterable
    result = list(exe.map(aad_run, repeat(data_use), repeat(metadata), seeds))

y_aad_all = np.array(result)

# save to file
np.save("../data/aad_galaxyzoo_200runs.npy", y_aad_all)

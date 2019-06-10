# A code to test strategies for expiditing feature extraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, njit, vectorize
import glob
import timeit
from numba import types
import sys
#import cProfile
#import re


#define the event class

setup = '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, njit, vectorize
import glob

class Event:
    def __init__(self, lc_data_file):
        self.filename = lc_data_file
        self.phot_data = self.get_phot_data()
        self.snid = self.getSNID()
        self.ra, self.dec = self.getRADec()
        
    def get_phot_data(event):
        f = open(event.filename, 'r')
        file_info = f.readlines()
        f.close()
        columns = [x.split() for x in file_info if x[0:8] == 'VARLIST:'][0][1:]
        phot_data = [x.split()[1:] for x in file_info if x[0:4] == 'OBS:']
        df = pd.DataFrame(phot_data, columns=columns)
        #df = df[['MJD', 'FLT', 'PHOTFLAG', 'PHOTPROB', 'FLUXCAL', 'FLUXCALERR', 'PSF']]
        #for index, row in df.iterrows():
        #    row['MJD'] = round(float(row['MJD']), 1)
        #    row['PHOTFLAG'] = int(row['PHOTFLAG'])
            #row['PHOTPROB'] = float(row['PHOTPROB'])
        #    row['FLUXCAL'] = float(row['FLUXCAL'])
        #    row['FLUXCALERR'] = float(row['FLUXCALERR'])
        #    row['PSF'] = float(row['PSF'])
            
        #filter out double observations
        #df = df.drop_duplicates(subset=['MJD', 'FLT'], keep='first')
        return df.values

    def getSNID(event):
        f = open(event.filename, 'r')
        file_info = f.readlines()
        f.close()
        snid = [x.split() for x in file_info if x[0:5] == 'SNID:'][0][1]
        return snid

    def getRADec(event):
        f = open(event.filename, 'r')
        file_info = f.readlines()
        f.close()
        ra = float([x.split() for x in file_info if x[0:3] == 'RA:'][0][1])
        dec = float([x.split() for x in file_info if x[0:3] == 'DEC'][0][1])
        return ra, dec

# Collect test files
data_dir1 = 'real_test_iz4_1/data/*'
data_dir2 = 'real_test_iz4_2/data/*'
filenames = glob.glob(data_dir1) + glob.glob(data_dir2) #total length = 2164

# Split data files into lists of different lengths
list_sizes = [10, 50, 100, 500, 1000, 1500, 2000]
file_lists = [filenames[0:size] for size in list_sizes]


############# TESTS
#############

# Baseline test
def make_events(filelist):
    events = []
    for filename in filelist:
        events.append(Event(filename))
    return events

# replace for loops with list comprehension
def list_comp(filelist):
    return [Event(filename) for filename in filelist]

# vecotrize with numpy
def vectorized(filelist):
    vEvent = np.vectorize(Event)
    arr = np.empty(len(filelist), dtype=object)
    arr[:] = vEvent(np.array(filelist))
    return arr


# numba-ized for loop
#class EventType(types.Type):
#    def __init__(self):
#        super(EventType, self).__init__(name='Event')

#event_dtype = np.dtype({'names':['filename','phot_data','snid','ra','dec'], 
#                             'formats':[np.unicode, 
#                                        np.ndarray, 
#                                        np.unicode, 
#                                        np.double, 
#                                        np.double]})

#@jit
#def create_events_numba(filelist):
    
#    events = np.zeros(len(filelist), dtype=event_dtype)
#    #attribute access only in @jitted function
#    for ii in range(len(filelist)):
#        #open and read file
#        f = open(filelist[ii], 'r')
#        file_info = f.readlines()
#        f.close() 
#
#        events[ii].filename = filelist[ii]
#        events[ii].snid = [x.split() for x in file_info if x[0:5] == 'SNID:'][0][1]
#
#        columns = [x.split() for x in file_info if x[0:8] == 'VARLIST:'][0][1:]
#        phot_data = [x.split()[1:] for x in file_info if x[0:4] == 'OBS:']
#        events[ii].phot_data = pd.DataFrame(phot_data, columns=columns).values 
#        events[ii].ra = float([x.split() for x in file_info if x[0:3] == 'RA:'][0][1])
#        events[ii].dec = float([x.split() for x in file_info if x[0:3] == 'DEC'][0][1])
#    return events

#@njit
#def make_events_numba(filelist):
#    events = []
#    for filename in filelist:
#        events.append(Event(filename))
#    return events     

#@njit
def read_file(filename):
    #open and read file
    f = open(filename, 'r')                                                                               
    file_info = f.readlines()                                                                                 
    f.close()            

    snid = [x.split() for x in file_info if x[0:5] == 'SNID:'][0][1]
    ra = float([x.split() for x in file_info if x[0:3] == 'RA:'][0][1]) 
    dec = float([x.split() for x in file_info if x[0:3] == 'DEC'][0][1])

    columns = [x.split() for x in file_info if x[0:8] == 'VARLIST:'][0][1:]
    phot_data = [x.split()[1:] for x in file_info if x[0:4] == 'OBS:']
    arr = pd.DataFrame(data=phot_data, columns=columns).values
    return [filename, snid, arr, ra, dec]

#@njit
def create_events_numba(filelist):
    return [read_file(filename) for filename in filelist]
        

'''

number = 10
repeat = 5
list_sizes = [10, 50, 100, 500, 1000, 1500, 2000]
"""
print("Baseline")
#baseline
data = []
for i in range(len(list_sizes)):
    print(i)
    baseline = '''
file_list = file_lists[%i]
make_events(file_list)
'''%i

    times = np.array(timeit.repeat(setup=setup, stmt=baseline, repeat=repeat, number=number))/number
    avg, std = np.mean(times), np.std(times)
    data.append((list_sizes[i], avg, std))
baseline_df = pd.DataFrame(data=data, columns=['size', 'avg', 'std'])
baseline_df.to_csv('baseline.csv')

print("List Comp")
#list comp
data = []
for i in range(len(list_sizes)):
    print(i)
    listcomp = '''
file_list = file_lists[%i]
list_comp(file_list)
'''%i

    times = np.array(timeit.repeat(setup=setup, stmt=listcomp, repeat=repeat, number=number))/number
    avg, std = np.mean(times), np.std(times)
    data.append((list_sizes[i],avg, std))
list_comp_df = pd.DataFrame(data=data, columns=['size', 'avg', 'std'])
list_comp_df.to_csv('listcomp.csv')


print("Vectorize")
# numpy vectorization
data = []
for i in range(len(list_sizes)):
    print(i)
    vectorized = '''
file_list = file_lists[%i]
vectorized(file_list)
'''%i

    times = np.array(timeit.repeat(setup=setup, stmt=vectorized, repeat=10, number=number))/number 
    avg, std = np.mean(times), np.std(times)
    data.append((list_sizes[i],avg, std))
vectorized_df = pd.DataFrame(data=data, columns=['size', 'avg', 'std'])
vectorized_df.to_csv('vectorized.csv')

"""
print("Numba")
#numba-ized for loop
data = []
for i in range(len(list_sizes)):
    print(i)
    numbaized = '''
file_list = file_lists[%i]
create_events_numba(file_list)
'''%i

    times = np.array(timeit.repeat(setup=setup, stmt=numbaized, repeat=10, number=number))/number
    avg, std = np.mean(times), np.std(times)
    data.append((list_sizes[i],avg, std))
numbaized_df = pd.DataFrame(data=data, columns=['size', 'avg', 'std'])
numbaized_df.to_csv('numbaized.csv')


#sys.exit()
####
# plot

plt.figure()
#plt.errorbar(baseline_df['size'], baseline_df['avg'], baseline_df['std'], label='For Loop')
#plt.errorbar(list_comp_df['size'], list_comp_df['avg'], list_comp_df['std'], label='List Comprehension')
plt.errorbar(numbaized_df['size'], numbaized_df['avg'], numbaized_df['std'], label='Numba-ized For Loop')
#plt.errorbar(vectorized_df['size'], vectorized_df['avg'], vectorized_df['std'], label='Vectorized')
plt.legend()
plt.xlabel("Dataset Size (Number of Light Curve Files")
plt.ylabel("Wall-Clock Time (s)")
plt.show(block=True)

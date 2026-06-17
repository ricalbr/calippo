# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:45:45 2024

@author: ICELAB_VACUUM
"""

from upplab.uppinstruments import Qontrol
from upplab.uppinstruments import ThorlabsPM
from upplab.uppinstruments import LFiberSwitch
from upplab.uppinstruments import OPM150
from upplab.upp_tools import amplitude_fidelity
from upplab.upp_tools import sinkhorn

from uppscale_cnr.bellcal import BellCal

import numpy as np
import time
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt
import pickle
import datetime

def graph_compare_matrices(N, A, B, label_A = 'Target U', label_B = 'Measured U', calculate_fidelity = True):
    fig, axes = plt.subplots(1, 2)
    (ax1, ax2) = axes
    
    if calculate_fidelity:
        fid = amplitude_fidelity(A, B)
        fig.suptitle(r'$\mathcal{F}_{\text{ampl}}$ = ' + f'{fid*100:.3f}%', fontsize=16)
    
    axis_ticks = list(range(N))
    axis_ticks_labels = list(range(1,N+1))
    ax1.imshow(A, vmin = 0, vmax = 1)
    ax1.set_title(label_A)
    ax1.set_xticks(axis_ticks)
    ax1.set_xticklabels(axis_ticks_labels)
    ax1.set_yticks(axis_ticks)
    ax1.set_yticklabels(axis_ticks_labels)
    ax1.set_ylabel('Output')
    ax1.set_xlabel('Input')
    pu = ax2.imshow(B, vmin = 0, vmax = 1)
    ax2.set_title(label_B)
    ax2.set_xlabel('Input')
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.3,
                        hspace=0.3)
    plt.xticks(axis_ticks, labels=axis_ticks_labels)
    plt.yticks(axis_ticks, labels=axis_ticks_labels)
    
    fig.colorbar(pu, ax=axes.ravel().tolist(), fraction=0.021, pad=0.04)
    plt.show()

def read_row(wait_time = 0.1, compensate_dark = True):
    to_return = []
    for i in range(6):
        switch_in.set_port(i)
        time.sleep(wait_time)
        to_return.append(power_meter.read())
    
    if np.min(to_return) < 0:
        to_return -= np.min(to_return)
        
    if compensate_dark:
        switch_in.set_port(-1)
        time.sleep(wait_time)
        to_return -= power_meter.read()
    
    return np.array([to_return])

def plot_row(wait_time = 0.1, normalize = True, compensate_dark = True):
    to_return = np.array(read_row(wait_time), compensate_dark)
    if normalize:
        to_return = to_return/np.sum(to_return)
    plt.imshow(np.array(to_return).reshape((1, 6)))
    return to_return

switch_in = LFiberSwitch.LFiberSwitch(1, 16, resource='ASRL5::INSTR', verbose = True)
switch_in.channels = [1, 2, 3, 4, 5, 6, 10]
switch_in.transient = 0.1

# switch_out = LFiberSwitch.LFiberSwitch(1, 16, resource='ASRL6::INSTR', verbose = False)
# switch_out.channels = [1, 2, 3, 4, 5, 6]

imax = 4.7 # 10 in air, 4.7 in vacuum
qtrl = Qontrol.Qontrol('COM7', imax = imax, verbose = True)

#ch =   [26, 22, 25, 23, 24,
#         19, 28, 20, 27, 21,
#         31, 17, 30, 18, 29,
#         10,  6,  9,  7,  8,
#          3, 12,  4, 11,  5,
#         15,  1, 14,  2, 13] # NEWEST MAPPING

ch =    [17, 29, 18, 31, 16,
        27, 20, 28, 19, 30,
        23, 26, 21, 25, 22,
         1, 13,  2, 15,  0,
        11,  4, 12,  3, 14,
         7, 10,  5,  9,  6] # ANTONINO TEMPORAL MAPPING

qtrl.channels = ch

# power_meter = ThorlabsPM.ThorlabsPM('220711312', 1550)
# power_meter.average_count = 1
# power_meter.average_time = 0.1

# Optical Power Readout with Santec
wavelength = 1550
santec_opm = OPM150.OPM150()
for i in range(1, 25):
    santec_opm.active_channel = i
    santec_opm.wavelength = wavelength
santec_opm.active_channel = 1

# power_meter_norm = ThorlabsPM.ThorlabsPM('220711311', 1550)
# power_meter_norm.average_count = 1
# power_meter_norm.average_time = 0.1

#%% NORMALIZATION TESTING

switch_in.set_port(0)
# switch_out.set_port(1)

data = []
for i in tqdm(range(100)):
    power_measured = power_meter.read()
#    power_norm = power_meter_norm.read()
#    time.sleep(1)
#    data.append((power_measured, power_norm, power_measured/power_norm))

plt.plot([d[2] for d in data])
plt.title("Normalized Power")
plt.show()
plt.plot([d[1] for d in data])
plt.title("Power at Normalizing PM")
plt.show()
plt.plot([d[0] for d in data])
plt.title("Power at Output PM")
plt.show()

#%% WRAPPING

import bellone_wrapper

bwis = bellone_wrapper.input_switch(switch_in, transient=0.2)
bwcd = bellone_wrapper.current_driver(qtrl, transient=0.2)

# # If using power meter + no switch
# bwom = bellone_wrapper.output_measurement_no_switch(power_meter, None, None, transient=0.3)

# If using Santec OPM
bwom = bellone_wrapper.output_measurement_santec(santec_opm, 6, 0.5)
bwom.channels = list(range(18, 24))

N = 6

cal = BellCal(N, imax, bwis, bwcd, bwom)
# cal.load_params('calibration_int_only_202406271016_air.bin')
# cal.load_params('calibration_int_only_202406271948_vacuum.bin')


#%% MAPPING TESTS AND VERIFICATION sept 2025

switch_in.set_port(0)
for i in [0, 6, 12, 18, 24]:
    cal.calibrate_int_ps(i, 5, plot=True, n_samples=30, imax=12)


#%% MEASURE PASSIVE U

now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

data = []
for i in tqdm(range(10)):
    M_passive = cal.measure_U()
    M_passive_normalized = sinkhorn(M_passive, tolerance = 1e-5)[0]
    U_p = np.sqrt(M_passive_normalized)
    graph_compare_matrices(N, np.eye(N), U_p)
    data.append(U_p)
    
with open('passive_matrix_measurement_' + date_string + '_air.bin', 'wb') as f:
    pickle.dump(data, f)

#%% MEASURE PASSIVE U no sinkhorn

now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

data = []
for i in tqdm(range(10)):
    M_passive = cal.measure_U()
    #M_passive_normalized = sinkhorn(M_passive, tolerance = 1e-5)[0]
    U_p = np.sqrt(M_passive)
    graph_compare_matrices(N, np.eye(N), U_p)
    data.append(U_p)
    
with open('passive_matrix_measurement_' + date_string + '_air.bin', 'wb') as f:
    pickle.dump(data, f)


#%% AUTOMATED CALIBRATION

int_data = cal.calibrate_internal(overwrite=True, plot=True, n_samples=30, imax=imax)
xt_data = cal.measure_xt_all(max_distance=100, plot=True, n_samples=30, imax=imax)
xt_fit_data = cal.fit_xt_all(xt_data, xt_coeff_guess=0.2, overwrite=True, plot=True)
bwcd.set_all_zero()

now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

with open('calibration_data_int_only_' + date_string + '.bin', 'wb') as f:
    pickle.dump({'int_data':int_data,
    'xt_data':xt_data,
    'xt_fit_data':xt_fit_data}, f)

#%% IMPLEMENT SWITCHES - DATASET GENERATION

import random

def generate_random_permutations(matrix, num_permutations=10):
    permutations = []
    original_indices = list(range(matrix.shape[0]))
    while len(permutations) < num_permutations:
        random_indices = random.sample(original_indices, len(original_indices))
        permuted_matrix = matrix[random_indices, :]
        if not any(np.array_equal(permuted_matrix, perm) for perm in permutations):
            permutations.append(permuted_matrix)
    return permutations

sample_size = 8
U_t_set = [np.eye(N)] + [np.eye(N)[::-1]] + generate_random_permutations(np.eye(N), sample_size)

#%% IMPLEMENT SWITCHES - MEASURE MATRICES

data = []

for i, U_t in enumerate(U_t_set):
    switch_in.set_port(6)
    time.sleep(0.5)
    dark_power = 0.0
    time.sleep(0.1)
    switch_in.set_port(0)
    time.sleep(0.5)
    cal.implement_switching(U_t, imax)
    time.sleep(0.5)
    M_m = cal.measure_U() - dark_power
    if np.min(M_m) < 0:
        M_m -= np.min(M_m)
   # bwcd.set_all_zero()


for i in enumerate(ch):
    ic(qtrl.get_current(i))

    
    U_m = np.sqrt(sinkhorn(M_m, 1e-6)[0])
    fidelity = amplitude_fidelity(np.abs(U_t), U_m)
    graph_compare_matrices(N, U_t, U_m,
                           label_A=f'Target U #{i+1}',
                           label_B=f'Measured U #{i+1}')
    
    data.append({'U_t': U_t,
                 'U_m': U_m,
                 'M_m': M_m,
                 'fid': fidelity,
                 'drk': dark_power})

now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

with open(str(len(U_t_set)) + '_switches_' + date_string + '_vac.bin', 'wb') as f:
    pickle.dump(data, f)

fidelities = [d['fid'] for d in data]
plt.figure(0,(6, 4))
plt.title(f'Scatterplot of Fidelities for {len(U_t_set)} Switching matrices, Air')
plt.ylabel(r'$\mathcal{F}_{abs}$')
plt.xlabel('Matrix Index')
plt.ylim(0.85, 1)
plt.hlines(np.mean(fidelities), 0, len(U_t_set), linestyles='dashed')
plt.scatter(range(len(U_t_set)), fidelities)


#%% RANDOM POWERS 4 MACHINE LEARNING

redundant_idx = [1, 3, -1, -3, -5]

sample_size = 1000
currents = []
Us_raw = []
Us = []
for i in tqdm(range(sample_size)):
    # Draw a random square-uniform distribution of currents
    c = np.sqrt(np.random.random_sample(len(qtrl.channels)) * imax**2)
    
    # Set currents leaving out the redundant ones
    c[redundant_idx] = 0.0
    bwcd.currents = c
    time.sleep(3)
    
    # Measure resulting matrix
    switch_in.set_port(6)
    time.sleep(0.1)
    dark_power = power_meter.read()
    time.sleep(0.1)
    U_raw = cal.measure_U() - dark_power
    if np.min(U_raw) < 0:
        U_raw -= np.min(U_raw)
    
    # Sinkhorn
    U = np.sqrt(sinkhorn(U_raw, 1e-6)[0])
    
    # Store data
    currents.append(c)
    Us_raw.append(U_raw.reshape(-1))
    Us.append(U.reshape(-1))
    
    plt.title(f"Matrix #{i + 1}")
    plt.imshow(U)
    plt.show()

# Save data
bwcd.set_all_zero()
now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

with open(str(sample_size) + '_randpowers_' + date_string + '_ML.bin', 'wb') as f:
    pickle.dump({
        'currents':np.array(currents),
        'Us_raw': np.array(Us_raw),
        'Us': np.array(Us)
        }, f)


#%% LOAD ML CALIBRATION

# with open('model_vac_large_parameters_noBSerr_2024-07-12.pickle', 'rb') as f:
#     cal.params = pickle.load(f)

# with open('model_large_parameters_noBSerr_2024-07-10.pickle', 'rb') as f:
#     cal.params = pickle.load(f)

# with open('bellone_int_only_cal_air_20250917.pickle', 'rb') as f:
#     cal.params = pickle.load(f)

# with open('bellone_air_xl_cal_20250916.pickle', 'rb') as f:
#     cal.params = pickle.load(f)

with open('model_vac_large_parameters_noBSerr_2024-07-12.pickle', 'rb') as f:
     cal.params = pickle.load(f)
    
#with open('bellone_model_large_parameters_noBSerr_2025-09-23.pickle', 'rb') as f:
#    cal.params = pickle.load(f)

#%% SAME UNITARY MANY TIMES

from scipy.stats import unitary_group
haar = unitary_group.rvs
U_t = haar(N)
sample_size = 3
data = []
for i in tqdm(range(sample_size)):
    cal.set_U(U_t, imax = imax)
    time.sleep(3)
    
    # Measure resulting matrix
    switch_in.set_port(6)
    time.sleep(0.1)
    dark_power = power_meter.read()
    time.sleep(0.1)
    M_m = cal.measure_U() - dark_power
    if np.min(M_m) < 0:
        M_m -= np.min(M_m)
    
    bwcd.set_all_zero()
    
    # Sinkhorn
    U_m = np.sqrt(sinkhorn(M_m, 1e-6)[0])
    
    fidelity = amplitude_fidelity(np.abs(U_t), U_m)
    graph_compare_matrices(N, np.abs(U_t), U_m)
    
    data.append({'U_t': U_t,
                 'U_m': U_m,
                 'M_m': M_m,
                 'fid': fidelity,
                 'drk': dark_power})

# Save data
bwcd.set_all_zero()
now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

with open(str(sample_size) + '_same_haar_' + date_string + '_vac.bin', 'wb') as f:
    pickle.dump(data, f)



#%% IMPLEMENT RANDOM HAAR

from scipy.stats import unitary_group
# cal.xt_matrix = np.diag(np.diag(cal.xt_matrix))
sample_size = 1000
data = []
for i in tqdm(range(sample_size)):
    haar = unitary_group.rvs
    U_t = haar(N)
    cal.set_U(U_t, imax = imax)
    
    # Measure resulting matrix
    switch_in.set_port(6)
    time.sleep(3)
    dark_power = power_meter.read()
    time.sleep(0.1)
    switch_in.set_port(0)
    time.sleep(0.5)
    M_m = cal.measure_U() - dark_power
    if np.min(M_m) < 0:
        M_m -= np.min(M_m)
    
    # Sinkhorn
    U_m = np.sqrt(sinkhorn(M_m, 1e-6)[0])
    
    fidelity = amplitude_fidelity(np.abs(U_t), U_m)
    graph_compare_matrices(N, np.abs(U_t), U_m)
    
    data.append({'U_t': U_t,
                 'U_m': U_m,
                 'M_m': M_m,
                 'fid': fidelity,
                 'drk': dark_power})

# Save data
bwcd.set_all_zero()
now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

with open(str(sample_size) + '_haars_' + date_string + '_ML_recalibrated.bin', 'wb') as f:
    pickle.dump(data, f)

fidelities = [d['fid'] for d in data]
plt.figure(0,(6, 4))
plt.title(f'Scatterplot of Fidelities for {sample_size} Haar Random matrices, Air')
plt.ylabel(r'$\mathcal{F}_{abs}$')
plt.xlabel('Matrix Index')
plt.ylim(0.99, 1)
plt.hlines(np.mean(fidelities), 0, sample_size, linestyles='dashed')
plt.scatter(range(sample_size), fidelities)

#%% GENERATE RANDOM HAAR

# Create {samples} many random Haars

from scipy.stats import unitary_group
# cal.xt_matrix = np.diag(np.diag(cal.xt_matrix))
sample_size = 100
haar_data = [unitary_group.rvs(N) for i in range(sample_size)]

with open(r'C:\Users\ICE_VACUUM\Desktop\bellone\noxtalk_measurement_{sample_size}_haars.bin', 'wb') as f:
    pickle.dump(haar_data, f)

#%% IMPLEMENT RANDOM HAAR WITH NO XTALK AND NO OUTPUT FSWITCH

def read_row(wait_time = 0.1):
    to_return = []
    for i in range(6):
        switch_in.set_port(i)
        time.sleep(wait_time)
        to_return.append(power_meter.read())
    return np.array([to_return])

cal.load_params('cal_air_noxtalk_8949_fix.pickle')
# cal.load_params('cal_vac_noxtalk_9956_fix.pickle')
with open(r'C:\Users\ICE_VACUUM\Desktop\bellone\noxtalk_measurement_{sample_size}_haars.bin', 'rb') as f:
    haar_data = pickle.load(f)

data_dict = {}
for i in range(N):
    data_dict[f'{i}'] = []

for i in range(N):
    input(f"Please continue by switching to output {i}. Press ENTER to continue.")
    
    # Measuring dark
    switch_in.set_port(6)
    time.sleep(0.2)
    # dark_power = power_meter.read() / power_meter_norm.read()
    dark_power = power_meter.read()
    time.sleep(0.2)
    
    # Measuring i-th row
    switch_in.set_port(i)
    for U_t in tqdm(haar_data):
        cal.set_U(U_t, imax = imax)
        time.sleep(0.2)
        
        # Measure row at given input and store
        data_dict[f'{i}'].append(ic(read_row(0.1) - dark_power))

bwcd.set_all_zero()

with open(r'C:\Users\ICE_VACUUM\Desktop\bellone\crespi_matrices.bin', 'wb') as f:
    pickle.dump(data_dict, f)

#%% RAW DATA CONVERSION INTO REFINED DATA

with open(r'C:\Users\ICE_VACUUM\Desktop\bellone\crespi_matrices.bin', 'rb') as f:
    data_dict = pickle.load(f)

data = []
for i, U_t in tqdm(enumerate(haar_data)):
    
    M_m = np.vstack([data_dict[f'{j}'][i] for j in range(6)])
    ic(M_m)
    ic(np.min(M_m))
    if np.min(M_m) < 0:
        M_m = M_m - np.min(M_m)
    U_m = np.sqrt(sinkhorn(M_m, 1e-6)[0])
    graph_compare_matrices(N, np.abs(U_t), U_m)
    
    data.append({
        'U_t': U_t,
        'U_m': U_m,
        'M_m': M_m,
        'fid': amplitude_fidelity(np.abs(U_t), U_m),
        })

avg_fidelity = np.mean([d['fid'] for d in data])
ic(avg_fidelity)

with open(r'C:\Users\ICE_VACUUM\Desktop\bellone\crespi_matrices_refined.bin', 'wb') as f:
    pickle.dump(data, f)

#%% CRESPI EXPERIMENT 1

"""
Switches between a set of input ports, waiting some time between each switch.
"""

def measure_row_crespi(input_array, switch_time):
    for i in tqdm(input_array):
        switch_in.set_port(i)
        time.sleep(switch_time)

def measure_crespi_with_ref(input_array, switch_time, ref_input = 0, ref_switch_time = 0.1):
    for i in tqdm(input_array):
        switch_in.set_port(ref_input)
        time.sleep(ref_switch_time)
        switch_in.set_port(i)
        time.sleep(switch_time)

# input_array = [0, 6, 1, 6, 2, 6, 0]
# input_array = [0, 6] * 6
input_array = [1, 2, 6] * 10

time.sleep(5)
for U_t in U_t_set['U_t']:
    cal.set_U(U_t)
    time.sleep(1)
    measure_crespi_with_ref(input_array, 0.2, 0, 0.2)
bwcd.set_all_zero()

#%% CRESPI EXPERIMENT 2 DATA GENERATION

import scipy
from scipy.stats import unitary_group
haar = unitary_group.rvs

# Building U_1
U_1 = np.eye(6, dtype=np.complex128)
U_1[0:2, 0:2] = np.sqrt(1/2) * scipy.linalg.dft(2)
U_1[2:4, 2:4] = np.sqrt(1/2) * scipy.linalg.dft(2)
U_1[4:6, 4:6] = np.sqrt(1/2) * scipy.linalg.dft(2)
plt.imshow(np.abs(U_1))
plt.show()

# Building U_2
U_2 = np.eye(6, dtype=np.complex128)
U_2[0:3, 0:3] = np.sqrt(1/3) * scipy.linalg.dft(3)
U_2[3:6, 3:6] = np.sqrt(1/3) * scipy.linalg.dft(3)
plt.imshow(np.abs(U_2))
plt.show()

# Building U_3
U_3 = np.eye(6, dtype=np.complex128)
U_3[1:3, 1:3] = np.sqrt(1/2) * scipy.linalg.dft(2)
U_3[3:5, 3:5] = np.sqrt(1/2) * scipy.linalg.dft(2)
plt.imshow(np.abs(U_3))
plt.show()

# Building U_4
U_4 = np.eye(6, dtype=np.complex128)
U_4[0:2, 0:2] = np.sqrt(1/2) * scipy.linalg.dft(2)
U_4[4:6, 4:6] = np.sqrt(1/2) * scipy.linalg.dft(2)
plt.imshow(np.abs(U_4))
plt.show()

# Building U_5
U_5 = np.eye(6, dtype=np.complex128)
U_5[1:3, 1:3] = haar(2)
U_5[3:5, 3:5] = haar(2)
plt.imshow(np.abs(U_5))
plt.show()

# Building U_6
U_6 = np.eye(6, dtype=np.complex128)
U_6[0:3, 0:3] = haar(3)
U_6[3:6, 3:6] = haar(3)
plt.imshow(np.abs(U_6))
plt.show()

now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')
with open('crespi_matrices_' + date_string + '.bin', 'wb') as f:
    pickle.dump({'U_t': [U_1, U_2, U_3, U_4, U_5, U_6]}, f)

#%% CRESPI EXPERIMENT PART 2

# input_array = [0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 0]
# input_array = [0, 6] * 6
input_array = [6, 0, 1, 2, 3, 4, 5] * 10
switch_time = 0.15

# cal.set_U(U_3)
measure_row_crespi(input_array, switch_time)
# bwcd.set_all_zero()

#%% CRESPI EXPERIMENT PART 3 THE FINAL PART

qtrl.verbose = False
switch_in.verbose = False

# Load matrix set
with open(r'C:\Users\ICE_VACUUM\Desktop\bellone\crespi_final_experiment_100_haars.bin', 'rb') as f:
    haars = pickle.load(f)

# Alternate quickly between passive and active matrix
for U_t in tqdm(haars):
    bwcd.set_all_zero()
    time.sleep(1)
    
    for input_port in range(6):
        bwis.port = input_port
        time.sleep(0.2)
    bwis.port = 6
    
    cal.set_U(U_t)
    time.sleep(1)
    
    for input_port in range(6):
        bwis.port = input_port
        time.sleep(0.2)
    bwis.port = 6

bwcd.set_all_zero()
time.sleep(1)

for input_port in range(6):
    bwis.port = input_port
bwis.port = 6



#%% TEMPORAL DYNAMICS
#%% INITIALIZE THE SETUP

from Bell_Temporal_detection import AFG
import random
import time  
import pyvisa as visa 
import pylab as pl 
import numpy as np 

visa_address = 'USB::0x0699::0x03C7::C010302::INSTR'

afg = AFG('USB::0x0699::0x034F::C020348::INSTR')



rm = visa.ResourceManager()
scope = rm.open_resource(visa_address)
scope.timeout = 10000 # ms
scope.encoding = 'latin_1'
scope.read_termination = '\n'
scope.write_termination = None
scope.write('*cls') # clear ESR

print(scope.query('*idn?'))


   
    
channels = ["CH1", "CH2","CH3"]                                   #it can be generalized to N elements but i only need 2 for now 
scaled_wave = {}
scaled_time = {}
    


             
#%%generate and store a matrix batch

now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')

def generate_random_permutations(matrix, num_permutations=10):
    permutations = []
    original_indices = list(range(matrix.shape[0]))
    while len(permutations) < num_permutations:
        random_indices = random.sample(original_indices, len(original_indices))
        permuted_matrix = matrix[random_indices, :]
        if not any(np.array_equal(permuted_matrix, perm) for perm in permutations):
            permutations.append(permuted_matrix)
    return permutations


from scipy.stats import unitary_group

sample_size = 98
N = 6

data = []
U_t_set = [np.eye(N)] + [np.eye(N)[::-1]] + generate_random_permutations(np.eye(N), sample_size)
U_t_set = U_t_set + [unitary_group.rvs(N) for i in range(100)]
with open(str(len(U_t_set)) + '_setU_' + date_string + '_air.bin', 'wb') as f:
    pickle.dump(U_t_set, f)


#%%implement and measure a single row

data={}

switch_in.set_port(6)
time.sleep(0.2)
# data["darkpower"] = detector.measure_multiple_ch()

data["matrices"] = []
data["triggers"] = []
data["time"] = []
for i, U_t in enumerate(U_t_set):
    data["matrices"].append({"U_t" : U_t})
    data["triggers"].append({})

for j in range(3):
    for i, U_t in enumerate(U_t_set):
                                              
        cal.implement_switching(U_t, imax)      #set target matrix                           
        time.sleep(0.5)                         #trigger afg                     
        for k in range(6):
            bwis.port = k
            time.sleep(1)                           #tempo per far agganciare l'oscilloscopio        
            afg.set_on()                            #afg start by opening the circuit, no currents flowing
            time.sleep(3)
            
            for ch in channels:
                
                    # io config
                    scope.write('header 0')
                    scope.write('data:encdg RIBINARY')
                    scope.write(f'data:source {ch}') # channel
                    scope.write('data:start 2450000') # first sample
                    #record = int(scope.query('wfmpre:nr_pt?'))
                    #scope.write('data:stop {}'.format(record)) # last sample
                    scope.write('data:stop 3450000') # last sample
                    record = 100001
                    scope.write('wfmpre:byt_nr 1') # 1 byte per sample
                    
                    # acq config
                    scope.write('acquire:state 0') # stop
                    scope.write('acquire:stopafter SEQUENCE') # single
                    scope.write('acquire:state 1') # run
                    t5 = time.perf_counter()
                    r = scope.query('*opc?') # sync
                    t6 = time.perf_counter()
                    print('acquire time: {} s'.format(t6 - t5))
                    
                    # data query
                    t7 = time.perf_counter()
                    bin_wave = scope.query_binary_values('curve?', datatype='b', container=np.array)
                    t8 = time.perf_counter()
                    print('transfer time: {} s'.format(t8 - t7))
                    
                    # retrieve scaling factors
                    tscale = float(scope.query('wfmpre:xincr?'))
                    tstart = float(scope.query('wfmpre:xzero?'))
                    vscale = float(scope.query('wfmpre:ymult?')) # volts / level
                    voff = float(scope.query('wfmpre:yzero?')) # reference voltage
                    vpos = float(scope.query('wfmpre:yoff?')) # reference position (level)
                    
                    # error checking
                    r = int(scope.query('*esr?'))
                    print('event status register: 0b{:08b}'.format(r))
                    r = scope.query('allev?').strip()
                    print('all event messages: {}'.format(r))
                    # horizontal (time)
                    
                    
                    
                    total_time = tscale * record
                    tstop = tstart + total_time
                    scaled_time[ch] = np.linspace(tstart, tstop, num=record, endpoint=False)
                    # vertical (voltage)
                    unscaled_wave = np.array(bin_wave, dtype='double') # data type conversion
                    scaled_wave[ch] = (unscaled_wave - vpos) * vscale + voff
                    

            
            data["time"] = scaled_time ["CH1"] 
            data["matrices"][i][2*j, k] = scaled_wave["CH2"]
            data["matrices"][i][2*j + 1, k] = scaled_wave["CH3"]
            data["triggers"][i][j, k] = scaled_wave["CH1"]
            afg.set_off()                            
   
    '''
    
    '''
    
    bwcd.set_all_zero()
    print("Switch to outputs: %d , %d" % (j+2,j+3)) #mettere un try per andare avanti
    input("Press enter to go on")
    
now = datetime.datetime.now()
date_string = now.strftime('%Y%m%d%H%M')
with open(str(len(U_t_set)) + 'measureoutput' + date_string + '_vac.bin', 'wb') as f:
     pickle.dump(data, f)


def generate_matrix_timeline_from_data(index):
    axis = list(range(6))
    key_space = [(i, j) for i in axis for j in axis]
    matrix_timeline = []
    for t in range(len(data['matrices'][index][0, 0])):
        matrix_temporary = np.zeros((6, 6))
        for i, j in key_space:
            matrix_temporary[i, j] = data['matrices'][index][i, j][t]
        matrix_timeline.append(matrix_temporary.copy())
    return matrix_timeline



#%% remapping of the wrong matrices
import copy

data2 = copy.deepcopy(data)  #copia indipendente

with open('2measureoutput202510281713_air.bin', 'rb') as f:
    data = pickle.load(f)
    

for j in range(3):
    for i, U_t in enumerate(U_t_set):
        for k in range(6):
             
            data2["matrices"][i][2*j, k] =  data["matrices"][i][2*j + 1, k]
            data2["matrices"][i][2*j + 1, k] = data["triggers"][i][j , k] 
            data2["triggers"][i][j, k] = data["matrices"][i][2*j, k]

def generate_matrix_timeline_from_data2(index):
    axis = list(range(6))
    key_space = [(i, j) for i in axis for j in axis]
    matrix_timeline = []
    for t in range(len(data2['matrices'][index][0, 0])):
        matrix_temporary = np.zeros((6, 6))
        for i, j in key_space:
            matrix_temporary[i, j] = data2['matrices'][index][i, j][t]
        matrix_timeline.append(matrix_temporary.copy())
    return matrix_timeline
            
data2_normalized = sinkhorn(identitytime[0], tolerance = 1e-5)[0]
data2_renormalized = data2_normalized/ np.max(data2_normalized)
#%% CLOSE ALL


switch_in.close()
# switch_out.close()
qtrl.close()
# power_meter.close()
santec_opm.close()
# power_meter_norm.close()
# afg.close()
# detector.close()
#%%
with open('3_switches_202511051510_vac.bin', 'rb') as f:
    data3 = pickle.load(f)
data3[0]["U_m"]

identitytime = generate_matrix_timeline_from_data(0)
fidelity_evo = []

for t in range(0,500000,100):
    ic(t)
    data2_normalized = sinkhorn(identitytime[t])[0]
    data2_renormalized = data2_normalized / np.max(data2_normalized)
    fidelity_evo.append( amplitude_fidelity(U_t_set[0], data2_normalized))

plt.plot(range(0,500000,100), fidelity_evo)
with open( 'fideliti_evo' + date_string + '_air.bin', 'wb') as f:
     pickle.dump(fidelity_evo, f)

#%%


max_current = 4.7 # in mA
qtrl = Qontrol.Qontrol('COM5', imax = max_current, verbose = True)
qtrl.channels = list(range(32))
power_meter = ThorlabsPM.ThorlabsPM('220711311', 1550)
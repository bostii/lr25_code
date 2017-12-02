"""
bsub -n 2 -o ~/lr25_code/output/%J.log -q psanaq  mpirun python b_ana_intensity_scan.py
"""
from psana import *
import numpy as np
import os
from skbeam.core.accumulators.histogram import Histogram

run = 66
folder = "/reg/d/psdm/AMO/amolr2516/results/npzfiles/run00%d" %run
files = os.listdir(folder)

for nfi,fi in enumerate(files):
    f=np.load(os.path.join(folder,fi))
    if nfi==0:
    	print "start"
        gas_mean=np.nanmean(f['GasDetector'][:,0])
        gas_std=np.nanstd(f['GasDetector'][:,0])
        gas_max=np.nanmax(f['GasDetector'][:,0])
        His2d = Histogram((100/10,0,gas_max),(714/10,0,713))
        His2d_weights = Histogram((100/10,0,gas_max),(714/10,0,713))
        His2d_weights_not_norm = Histogram((100/10,0,gas_max),(714/10,0,713))
    for evt in range(len(f['GasDetector'][:,0])):
        His2d.fill(list(f['GasDetector'][evt,0]*np.ones(714)),range(714))
        values= list(np.nan_to_num(np.asarray(f['SHESwf'][evt,:])))
        His2d_weights.fill(list(f['GasDetector'][evt,0]*np.ones(714)),range(714),weights=values/f['GasDetector'][evt,0])
        His2d_weights_not_norm.fill(list(f['GasDetector'][evt,0]*np.ones(714)),range(714),weights=values)
    print "%.2f done" %(float(nfi)/len(files))

filename = "/reg/neh/home5/sjarosch/lr25_code/npz/ana_intensity_scan_run_%d" %run
np.savez(filename,His2d_centers=His2d.centers,His2d_val=His2d.values,His2d_weights_val=His2d_weights.values,His2d_weights_not_norm_vals=His2d_weights_not_norm.values)
"""
bsub -n 2 -o ~/lr25_code/output/%J.log -q psanaq  mpirun python ana_delay_scan.py
"""

from psana import *
import numpy as np
import os
from skbeam.core.accumulators.histogram import Histogram

run = 77
folder = "/reg/d/psdm/AMO/amolr2516/results/npzfiles/run00%d" %run
files = os.listdir(folder)

#Histogramm variables
GD_bins = 10
GD_min = 0.0
#GD_max = max value in first file

energy1_bins = 10
#energy1_min =
#energy1_max =

energy2_bins = 10
#energy2_min = 
#energy2_max =

SHESpix_bins = 714
SHES_pix_min = 0
SHES_pix_max = 714

for nfi,fi in enumerate(files):
    data=np.load(os.path.join(folder,fi))
    evtnr=len(data['GasDetector'][:,0])
    if nfi==0:
        GD_max=np.nanmax(data['GasDetector'][:,0])
        energies_1_pos = np.asarray(data['UXSpc'][:,0])
        energies_2_pos = np.asarray(data['UXSpc'][:,3])
        energies_pos = np.sort(np.vstack((energies_1_pos,energies_2_pos)),axis=0)
        #energy1_mean = np.nanmean(energies_pos[0])
        #energy2_mean = np.nanmean(energies_pos[1])
        energy1_mean = 550
        energy2_mean = 750
        energy1_min = energy1_mean-100
        energy1_max = energy1_mean+100
        energy2_min = energy2_mean-100
        energy2_max = energy2_mean+100
        energy_threshold = 0.1*np.nanmean(data['UXSpc'][:,2])
        
        #bin: (GD),(energy_1),(energy_2),(SHS_pix)
        His2d = Histogram((GD_bins,GD_min,GD_max),(energy1_bins,energy1_min,energy1_max), \
                          (energy2_bins,energy2_min,energy2_max),(SHESpix_bins,SHES_pix_min,SHES_pix_max))
        His2d_w = Histogram((GD_bins,GD_min,GD_max),(energy1_bins,energy1_min,energy1_max), \
                          (energy2_bins,energy2_min,energy2_max),(SHESpix_bins,SHES_pix_min,SHES_pix_max))
    
    for evt in range(evtnr):
        #check for threshold of intensity in both pulses
        UXSpc=np.asarray(data['UXSpc'][evt,:])
        if UXSpc[2]<energy_threshold or UXSpc[5]<energy_threshold or data['GasDetector'][evt,0]<GD_min:
            continue
        if UXSpc[0]>UXSpc[3]:
            UXSpc=np.roll(UXSpc,3)
        values= list(np.nan_to_num(np.asarray(data['SHESwf'][evt,:])))
        His2d.fill(list(data['GasDetector'][evt,0]*np.ones(714)),list(UXSpc[0]*np.ones(714)),\
                   list(UXSpc[3]*np.ones(714)),range(714))
        His2d_w.fill(list(data['GasDetector'][evt,0]*np.ones(714)),list(UXSpc[0]*np.ones(714)),\
                     list(UXSpc[3]*np.ones(714)),range(714),weights=values)
    print "%.2f done" %(float(nfi+1)/len(files))
print "done"


filename = "/reg/neh/home5/sjarosch/lr25_code/npz/ana_delay_scan_run_%d" %run
np.savez(filename,His2d_centers=His2d.centers,His2d_val=His2d.values,His2d_weights_val=His2d_w.values)
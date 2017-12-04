"""
bsub -n 64 -o ~/lr25_code/output/%J.log -q psanaq  mpirun python ana_delay2.py

bsub -n 64 -o ~/lr25_code/output/%J.log -q psnehhiprioq  mpirun python ana_delay2.py --run 77
"""

from psana import *
import numpy as np
import time
import os
from skbeam.core.accumulators.histogram import Histogram
from mpi4py import MPI

import argparse


run_default = 77  #if not specified by --run
parser = argparse.ArgumentParser(description='Specify the run')
parser.add_argument('--run', type=int, default= run_default,
                    help='put run number')

args =  parser.parse_args()
run = int(args.run)
#print testrun
#print type(testrun)

# For parallelisation
comm = MPI.COMM_WORLD # define parallelisation object
rank = comm.Get_rank() # which core is script being run on
size = comm.Get_size() # no. of CPUs being used

print "%d nodes working, rank: %d" %(size,rank)

folder = "/reg/d/psdm/AMO/amolr2516/results/npzfiles/run%04d" %run
files = os.listdir(folder)

#Histogramm variables
GD_bins = 5
GD_min = 0.4
GD_max = 1.1

energy1_bins = 10
#energy1_min =
#energy1_max =

energy2_bins = 10
#energy2_min = 
#energy2_max =

energy1_mean = 550
energy2_mean = 750
energy1_min = energy1_mean-100
energy1_max = energy1_mean+100
energy2_min = energy2_mean-100
energy2_max = energy2_mean+100

SHESpix_bins = 714
SHES_pix_min = 0
SHES_pix_max = 714

#data=np.load(os.path.join(folder,files[0]))
#GD_max=np.nanmax(data['GasDetector'][:,0])
#energies_1_pos = np.asarray(data['UXSpc'][:,0])
#energies_2_pos = np.asarray(data['UXSpc'][:,3])
#energies_pos = np.sort(np.vstack((energies_1_pos,energies_2_pos)),axis=0)
#energy1_mean = np.nanmean(energies_pos[0])
#energy2_mean = np.nanmean(energies_pos[1])

e1_int_bins=5
e1_int_min=20000
e1_int_max=90000

e2_int_bins=5
e2_int_min=500
e2_int_max=60000

#energy_threshold = 50000

#bin: (GD),(energy_1),(energy_2),(e1_int),(e2_int),(SHS_pix)
His2d = Histogram((GD_bins,GD_min,GD_max),(energy1_bins,energy1_min,energy1_max),\
	(energy2_bins,energy2_min,energy2_max),(e1_int_bins,e1_int_min,e1_int_max),\
	(e2_int_bins,e2_int_min,e2_int_max),(SHESpix_bins,SHES_pix_min,SHES_pix_max))
His2d_w = Histogram((GD_bins,GD_min,GD_max),(energy1_bins,energy1_min,energy1_max),\
	(energy2_bins,energy2_min,energy2_max),(e1_int_bins,e1_int_min,e1_int_max),\
	(e2_int_bins,e2_int_min,e2_int_max),(SHESpix_bins,SHES_pix_min,SHES_pix_max))

commhist=np.zeros((GD_bins,energy1_bins,energy2_bins,e1_int_bins,e2_int_bins,SHESpix_bins))
commhist_w=np.zeros((GD_bins,energy1_bins,energy2_bins,e1_int_bins,e2_int_bins,SHESpix_bins))
commtest=np.zeros(1)


for nfi,fi in enumerate(files):
	if nfi%size!=rank: continue
	data=np.load(os.path.join(folder,fi))
	evtnr=len(data['GasDetector'][:,0])

	for evt in range(evtnr):
		#check for threshold of intensity in both pulses
		UXSpc=np.asarray(data['UXSpc'][evt,:])
		Pressure = data['Pressure'][evt]*1e5
		#if UXSpc[2]<energy_threshold or UXSpc[5]<energy_threshold or data['GasDetector'][evt,0]<GD_min:
		if data['GasDetector'][evt,0:2].mean()<GD_min:
			continue
		if UXSpc[0]>UXSpc[3]:
			UXSpc = np.roll(UXSpc,3)
		values= list(np.nan_to_num(np.asarray(data['SHESwf'][evt,:]/Pressure)))
		His2d.fill(list(data['GasDetector'][evt,0:2].mean()*np.ones(714)),list(UXSpc[0]*np.ones(714)),\
			list(UXSpc[3]*np.ones(714)),list(UXSpc[2]*np.ones(714)),\
			list(UXSpc[5]*np.ones(714)),range(714))
		His2d_w.fill(list(data['GasDetector'][evt,0:2].mean()*np.ones(714)),list(UXSpc[0]*np.ones(714)),\
			list(UXSpc[3]*np.ones(714)),list(UXSpc[2]*np.ones(714)), \
			list(UXSpc[5]*np.ones(714)),range(714),weights=values)
		#print "%f, %f" %(UXSpc[2],UXSpc[5])
	print "rank %d done file %d" %(rank,nfi)
print "rank %d done"

if rank==0:
	time.sleep(3)

if 'His2d' in locals():
	comm.Reduce(His2d.values,commhist)
	comm.Reduce(His2d_w.values,commhist_w)
	comm.Reduce(np.ones(1),commtest)

if rank==0:
	filename = "/reg/neh/home5/sjarosch/lr25_code/npz/ana_delay2_run_%04d" %run
	np.savez(filename,His2d_centers=His2d.centers,His2d_val=commhist,His2d_weights_val=commhist_w)
	print "%d of %d have reported" %(commtest,size)
MPI.Finalize()
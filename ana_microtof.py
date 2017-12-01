"""
bsub -q psanaq -n 64 -o ~/output/%J.out python ana_microtof.py 

"""
from psana import *
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('/reg/neh/home/sjarosch/LR25_Analysis/PythonRoutines')
from skbeam.core.accumulators.histogram import Histogram
from FEEGasProcessing import FEEGasProcessor
from psmon.plots import XYPlot,Image,Hist,MultiPlot
from psmon import publish
#publish.init(post=12304)

run = 52
ds = DataSource("exp=AMO/amolr2516:run=%d:smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live" % run)
procFEE=FEEGasProcessor()
MINITOF_det = Detector('ACQ1')
EBEAM_det = Detector('EBeam')


vals = []
vals_l3 = []
ct = 0
for nevt, evt in enumerate(ds.events()):
    FEE_shot_energy = procFEE.ShotEnergy(evt)
    EBEAM = EBEAM_det.get(evt)
    if EBEAM is None:
        continue
    EBEAM_beam_energy = EBEAM.ebeamL3Energy()  # beam energy in MeV
    if FEE_shot_energy is None:
        continue
    if nevt%50==0:
        ct += 1
        vals.append(FEE_shot_energy)
        vals_l3.append(EBEAM_beam_energy)
    if ct > 50:
        print "done"
        break
vals_std = np.asarray(vals).std()
vals_mean = np.asarray(vals).mean()
threshold = vals_mean - 2*vals_std
l3_mean = int(np.asarray(vals_l3).mean())
print "threshold: %f, mean: %f, std: %f, l3: %d" % (threshold,vals_mean,vals_std,EBEAM_beam_energy)

numbins_L3Energy=100
minhistlim_L3Energy=l3_mean-50
maxhistlim_L3Energy=l3_mean+50

numbins_L3EnergyWeighted=100
minhistlim_L3EnergyWeighted=minhistlim_L3Energy
maxhistlim_L3EnergyWeighted=maxhistlim_L3Energy

hist_L3Energy=Histogram((numbins_L3Energy,minhistlim_L3Energy,maxhistlim_L3Energy))
hist_L3EnergyWeighted=Histogram((numbins_L3EnergyWeighted,minhistlim_L3EnergyWeighted,maxhistlim_L3EnergyWeighted))
hist_L3EnergyWeighted_unnorm=Histogram((numbins_L3EnergyWeighted,minhistlim_L3EnergyWeighted,maxhistlim_L3EnergyWeighted))


ds = DataSource("exp=AMO/amolr2516:run=%d:smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live" % run)
for nevt, evt in enumerate(ds.events()):

    wf= MINITOF_det.waveform(evt)
    #wf_time = MINITOF_det.wftime(evt)
    EBEAM = EBEAM_det.get(evt)
    if wf is None or EBEAM is None:
        continue
    EBEAM_beam_energy = EBEAM.ebeamL3Energy()  # beam energy in MeV
    FEE_shot_energy = procFEE.ShotEnergy(evt)
    #print FEE_shot_energy

    if FEE_shot_energy is None or FEE_shot_energy<threshold:
        continue
    hist_L3Energy.fill(EBEAM_beam_energy)
    hist_L3EnergyWeighted.fill([EBEAM_beam_energy], weights=[np.abs(wf[1][:10000].sum())/FEE_shot_energy])
    hist_L3EnergyWeighted_unnorm.fill([EBEAM_beam_energy], weights=[np.abs(wf[1][:10000].sum())])
    if nevt%500==0:
        print nevt
        multi = MultiPlot(nevt,'MULTI')
        vShotsPerL3Energy=np.nan_to_num(hist_L3EnergyWeighted.values/hist_L3Energy.values)
        L3_weighted_plot = XYPlot(nevt, 'L3', hist_L3EnergyWeighted.centers[0], vShotsPerL3Energy)
        #publish.send('L3', L3_weighted_plot)
        multi.add(L3_weighted_plot)
        vShotsPerL3Energy_un=np.nan_to_num(hist_L3EnergyWeighted_unnorm.values/hist_L3Energy.values)
        L3_weighted_plot_un = XYPlot(nevt, 'L3UN', hist_L3EnergyWeighted.centers[0], vShotsPerL3Energy_un)
        #publish.send('L3UN', L3_weighted_plot_un)
        multi.add(L3_weighted_plot_un)
        publish.send('MULTI',multi)
        filename = "../npz/ana_itof_run_%d" %run
        np.savez(filename,vShotsPerL3Energy=vShotsPerL3Energy,vShotsPerL3Energy_un=vShotsPerL3Energy_un,hist_L3Energy_centers=hist_L3EnergyWeighted.centers[0])

    #break
vShotsPerL3Energy=np.nan_to_num(hist_L3EnergyWeighted.values/hist_L3Energy.values)

print "tset"

L3_weighted_plot = XYPlot(nevt, 'L3', hist_L3EnergyWeighted.centers[0], vShotsPerL3Energy)
publish.send('L3', L3_weighted_plot)
time.sleep(0.5)
#while 1==1:
 #   x = 1
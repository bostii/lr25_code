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


ds = DataSource("exp=AMO/amolr2516:run=42:smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live")
procFEE=FEEGasProcessor()
MINITOF_det = Detector('ACQ1')
EBEAM_det = Detector('EBeam')

numbins_L3Energy=100
minhistlim_L3Energy=3300
maxhistlim_L3Energy=3400

numbins_L3EnergyWeighted=100
minhistlim_L3EnergyWeighted=3300
maxhistlim_L3EnergyWeighted=3400

hist_L3Energy=Histogram((numbins_L3Energy,minhistlim_L3Energy,maxhistlim_L3Energy))
hist_L3EnergyWeighted=Histogram((numbins_L3EnergyWeighted,minhistlim_L3EnergyWeighted,maxhistlim_L3EnergyWeighted))

for nevt, evt in enumerate(ds.events()):
    wf= MINITOF_det.waveform(evt)
    #wf_time = MINITOF_det.wftime(evt)
    EBEAM = EBEAM_det.get(evt)
    if wf is None or EBEAM is None:
        continue
    EBEAM_beam_energy = EBEAM.ebeamL3Energy()  # beam energy in MeV
    FEE_shot_energy = procFEE.ShotEnergy(evt)
    #print FEE_shot_energy
    if FEE_shot_energy is None:
        continue
        
    hist_L3Energy.fill(EBEAM_beam_energy)
    hist_L3EnergyWeighted.fill([EBEAM_beam_energy], weights=[np.abs(wf[1][:10000].sum())/FEE_shot_energy])
    
    if nevt > 1000:
        break
    #break
vShotsPerL3Energy=hist_L3EnergyWeighted/np.float(hist_L3Energy.values)

L3_weighted_plot = XYPlot(nevt, 'L3', hist_L3EnergyWeighted.centers[0], hist_L3EnergyWeighted.values)
publish.send('L3', L3_weighted_plot)
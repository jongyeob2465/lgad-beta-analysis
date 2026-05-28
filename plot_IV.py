import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import mplhep as hep

#define function for I-V curve plotting
def plot_iv_curve(voltage, current):
    plt.figure(figsize=(8, 6))
    hep.set_style(hep.style.CMS)
    plt.plot(voltage[2:], current[2:], marker='o', linestyle='-', color='b',label='LGAD pad current')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.yscale('log')
    plt.grid()

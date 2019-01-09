#r# This example shows a low-pass RC Filter.

####################################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

####################################################################################################

from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

####################################################################################################

#f# circuit_macros('low-pass-rc-filter.m4')

circuit = Circuit('Low-Pass RC Filter')

circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=1@u_V)

Rs0 = 5
R1 = 50
C1 = 0.5
Rt0 = 5

Rs = circuit.R('s', 'in', 1, Rs0@u_Ω)
R1 = circuit.R(1, 1, 2, R1@u_Ω)
C1 = circuit.C(1, 2, circuit.gnd, C1@u_pF)
Rt = circuit.R('t', 2, circuit.gnd, Rt0@u_Ω )

#r# The break frequency is given by :math:`f_c = \frac{1}{2 \pi R C}`

break_frequency = 1 / (2 * math.pi * float(R1.resistance * C1.capacitance))
print("Break frequency = {:.1f} Hz".format(break_frequency))
#o#

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.ac(start_frequency=1e8@u_Hz, stop_frequency=5e10@u_Hz, number_of_points=100,  variation='dec')
# print(analysis.out)

#r# We plot the Bode diagram.

# figure = plt.figure(1, (20, 10))
# figure = plt.figure(figsize=(6,10))
#
# plt.title("Bode Diagram of a Low-Pass RC Filter")
# axes = (plt.subplot(211), plt.subplot(212))
# bode_diagram(axes=axes,
#              frequency=analysis.frequency,
#
#              gain=20*np.log10(np.absolute(analysis.out)),
#              phase=np.angle(analysis.out, deg=False),
#              marker='.',
#              color='blue',
#              linestyle='-',
#          )
# for axe in axes:
#     axe.axvline(x=break_frequency, color='red')
#
# plt.tight_layout()
# plt.show()


#
# fig = plt.figure(figsize=(8,12))
# plt.subplot(411)
# plt.plot(analysis.frequency, 20*np.log10(np.absolute(analysis.out)), 'o-')
# plt.grid(True)
# plt.ylabel('Gain (out) [dB]')
# # plt.title(mycircuit.title + " - AC Simulation")
#
# plt.subplot(412)
# plt.plot(analysis.frequency, 20*np.log10(np.absolute(analysis['1'])), 'o-')
# plt.grid(True)
# plt.ylabel('Gain (out/in) [dB]')
#
# plt.subplot(413)
# plt.plot(analysis.frequency, 20*np.log10(np.absolute(analysis.out)/np.absolute(analysis['1'])), 'o-')
# plt.grid(True)
# plt.ylabel('Gain (out/in) [dB]')
#
# plt.xlabel('Frequency [Hz]')
# fig.savefig('ac_plot.png')
# plt.show()



#f# save_figure('figure', 'low-pass-rc-filter-bode-diagram.png')



V1 = np.array(analysis['1'])
I1 = np.array((analysis['in'] - analysis['1'])/Rs0)
V2 = np.array(analysis['2'])
I2 = np.array((analysis['2'])/Rt0)

#Z0 = np.array((V1-V2)/I1)
# Z0 = np.array((V1)/I1)
# Z0 = 50
Z01 = Rs0
Z02 = Rt0

#Ref: Dean A. Frickey, "Conversions between S, Z, Y, h, ABCD and T parameters which are valid for complex source and load impedances" ,IEEE Trans. Microwave thieory and techniques, vol42, No2, 1994
a1 = (V1 + I1*Z01)/(2*np.sqrt(Z01))
b1 = (V1 - I1*Z01)/(2*np.sqrt(Z01))
# a2 = (V2 - I2*Z02)/(2*np.sqrt(Z02))
b2 = (V2 + I2*Z02)/(2*np.sqrt(Z02))

S11 = b1/a1
S21 = b2/a1

fig = plt.figure(figsize=(6,12))
plt.subplot(211)
plt.plot(analysis.frequency, 20*np.log10(np.absolute(S21)), 'o-')
plt.grid(True)
plt.ylabel('S21 [dB]')
# plt.title(mycircuit.title + " - AC Simulation")

plt.plot(analysis.frequency, 20*np.log10(np.absolute(S11)), 'o-')
plt.grid(True)
plt.ylabel('S11 [dB]')

plt.subplot(212)
plt.plot(analysis.frequency, np.absolute((V1-V2)/I1), 'o-')
plt.grid(True)
plt.ylabel('Z0 [Ohm]')
plt.ylim(0,100)


plt.xlabel('Frequency [Hz]')
fig.savefig('ac_plot.png')
plt.show()
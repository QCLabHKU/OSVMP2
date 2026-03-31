import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
from scipy import signal
import sys

# Inputs
autocorrelation_option = 1 # 1 to calculate it, 2 to load a pre-calculated one
T = 150. # K
#T = 290. # K
time_step = 0.5 #fs
zoom_wavenum = 4500. # cm^-1
fraction_autocorrelation_function_to_fft = .12 # since you only want to fft the part that has meaningful statistics,
                                                # and at 1% of the trajectory all datapoints have at least 100 non-ish-correlated counts

try: 
    input_dipole_file = sys.argv[1]
    mol = 'eigen'
except IndexError:
    mol = 'eigen'
    #mol = 'zundel'
    basis = 'augccpvdz'#
    #basis = '631g'
    #mol = "%s_%s"%(mol, basis)
    input_dipole_file = '/Users/calvin/Dropbox/code/MPI/work/results/%s/%s_1e-2_0.2/dip_mom_%s_MP2.dat'%(basis, mol, mol)
    #input_dipole_file = '/Users/calvin/Dropbox/code/MPI/work/results/zundel_1e-2_0.2/dip_mom_zundel_MP2.dat'
output_autocorrelation_file = 'autocorr.txt'

# Constants
boltz = 1.38064852E-23 # m^2 kg s^-2 K^-1
lightspeed = 299792458. # m s^-1
reduced_planck = 1.05457180013E-34 # kg m^2 s^-1


###############################################################################
# Get autocorrelation function
###############################################################################
# Calculate autocorrelation function
if autocorrelation_option == 1:
    # Load data
    #time, dipole_x, dipole_y, dipole_z = np.loadtxt(input_dipole_file, skiprows=2, usecols=(1,2,3,4), unpack=True)
    dipole_x, dipole_y, dipole_z = np.loadtxt(input_dipole_file, usecols=(0,1,2), unpack=True)
    time = np.arange(len(dipole_x))*time_step
    print(dipole_x)
    # Do calculation
    # Note that this method of calculating an autocorrelation function is very fast, but it can be difficult to follow.
    # For readability, I've presented a more straightforward (but much, much slower) method in the commented block below.
    print("Calculating autocorrelation function.")
    # Shift the array
    if len(time) % 2 == 0:
        dipole_x_shifted = np.zeros(len(time)*2)
        dipole_y_shifted = np.zeros(len(time)*2)
        dipole_z_shifted = np.zeros(len(time)*2)
    else:
        dipole_x_shifted = np.zeros(len(time)*2-1)
        dipole_y_shifted = np.zeros(len(time)*2-1)
        dipole_z_shifted = np.zeros(len(time)*2-1)
    dipole_x_shifted[len(time)//2:len(time)//2+len(time)] = dipole_x
    dipole_y_shifted[len(time)//2:len(time)//2+len(time)] = dipole_y
    dipole_z_shifted[len(time)//2:len(time)//2+len(time)] = dipole_z
    # Convolute the shifted array with the flipped array, which is equivalent to performing a correlation
    autocorr_x_full = (signal.fftconvolve(dipole_x_shifted,dipole_x[::-1], mode='same')[(-len(time)):]
                       / np.arange(len(time), 0, -1))
    autocorr_y_full = (signal.fftconvolve(dipole_y_shifted,dipole_y[::-1], mode='same')[(-len(time)):]
                       / np.arange(len(time), 0, -1))
    autocorr_z_full = (signal.fftconvolve(dipole_z_shifted,dipole_z[::-1], mode='same')[(-len(time)):]
                       / np.arange(len(time), 0, -1))
    autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full
    # Truncate the autocorrelation array
    autocorr = autocorr_full[:int(len(time) * fraction_autocorrelation_function_to_fft)]
    print("Finished with autocorrelation function calculation.")

#    # More straightforward (but much, much slower) method of doing the calculation
#    autocorr=np.zeros(int(len(time) * fraction_autocorrelation_function_to_fft))
#    print("Calculating autocorrelation function. Will need to go up to: " + str(len(autocorr)))
#    for i in range(len(autocorr)):
#        total=0.
#        count=0.
#        for j in range(len(time)-i):
#            total += dipole_x[j]*dipole_x[j+i] + dipole_y[j]*dipole_y[j+i] + dipole_z[j]*dipole_z[j+i]
#            count += 1.
#        autocorr[i] = total/count
#        print("Finished with autocorrelation function calculation up to: " + str(i))
    
    # Save data
    np.savetxt(output_autocorrelation_file, np.column_stack((time[:len(autocorr)], autocorr)),
               header='Time(fs) Autocorrelation(e*Ang)')

# Load pre-calculated autocorrelation function
elif autocorrelation_option == 2:
    time, autocorr = np.loadtxt(output_autocorrelation_file, skiprows=1, unpack=True)

else:
    print("Not a valid option for 'autocorrelation_option'.")

timestep = (time[1]-time[0]) * 1.E-15 # converts time from femtoseconds to seconds


###############################################################################
# Calculate spectra
# Note that intensities are relative, and so can be multiplied by a constant to compare to experiment.
###############################################################################
# Calculate the FFTs of autocorrelation functions
lineshape = fftpack.dct(autocorr, type=1)[1:]
lineshape_frequencies = np.linspace(0, 0.5/timestep, len(autocorr))[1:]
lineshape_frequencies_wn = lineshape_frequencies / (100.*lightspeed) # converts to wavenumbers (cm^-1)

# Calculate spectra
field_description =  lineshape_frequencies * (1. - np.exp(-reduced_planck*lineshape_frequencies/(boltz*T)))
quantum_correction = lineshape_frequencies / (1. - np.exp(-reduced_planck*lineshape_frequencies/(boltz*T)))
                     # quantum correction per doi.org/10.1021/jp034788u. Other options are possible, see doi.org/10.1063/1.441739 and doi.org/10.1080/00268978500102801.
spectra = lineshape * field_description
spectra_qm = spectra * quantum_correction

# Save data
np.savetxt('IR-data.txt', np.column_stack((lineshape_frequencies_wn, lineshape, field_description, quantum_correction, spectra, spectra_qm)),
           header='Frequency(cm^-1), Lineshape, Field_description, Quantum_correction, Spectra, Spectra_qm')

###############################################################################
# Plots
###############################################################################
mask = (lineshape_frequencies_wn >= 0) & (lineshape_frequencies_wn <= zoom_wavenum)

'''# Plot autocorrelation function
plt.figure()
plt.plot(time[:len(autocorr)], autocorr)
plt.xlabel('Time (fs)')
plt.ylabel(r'Dipole Moment Autocorrelation Function (e$\AA$)')
plt.savefig('IR-autocorr.png')

# Plot lineshape
plt.figure()
plt.plot(lineshape_frequencies_wn, lineshape)
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-lineshape.png')
plt.figure()
plt.plot(lineshape_frequencies_wn[mask], lineshape[mask])
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-zoom-lineshape.png')

# Plot field description
plt.figure()
plt.plot(lineshape_frequencies_wn, field_description)
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-fielddescription.png')
plt.figure()
plt.plot(lineshape_frequencies_wn[mask], field_description[mask])
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-zoom-fielddescription.png')

# Plot quantum correction
plt.figure()
plt.plot(lineshape_frequencies_wn, quantum_correction)
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-quantumcorrection.png')
plt.figure()
plt.plot(lineshape_frequencies_wn[mask], quantum_correction[mask])
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-zoom-quantumcorrection.png')'''

# Plot spectra
plt.figure()
ax = plt.subplot(111)
plt.plot(lineshape_frequencies_wn, spectra)
plt.ylim(0, max(spectra))
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
plt.ylim(0, max(spectra))
plt.xlim(0,4500)
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
#plt.gca().invert_xaxis()
#plt.savefig('IR-spectra.png')
plt.show()

'''
plt.figure()
plt.plot(lineshape_frequencies_wn[mask], spectra[mask])
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
#plt.gca().invert_xaxis()
plt.savefig('IR-zoom-spectra.png')'''

'''# Plot spectra with quantum correction
plt.figure()
plt.plot(lineshape_frequencies_wn, spectra_qm)
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
plt.xlim(0,4500)
plt.ylim(0, max(spectra_qm))
#plt.gca().invert_xaxis()
plt.savefig('%s_IR-spectra-quantumcorrection.png'%mol)'''

plt.figure()
ax = plt.subplot(111)
plt.plot(lineshape_frequencies_wn[mask], spectra_qm[mask])
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
plt.ylim(0, max(spectra_qm[mask]))
plt.xlim(0,5000)
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
#plt.gca().invert_xaxis()
plt.show()
#plt.savefig('%s_IR-zoom-spectra-quantumcorrection.pdf'%mol)

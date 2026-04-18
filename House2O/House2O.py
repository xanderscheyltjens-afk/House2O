#Simulation code for the House2O project
#Authors: Mona Soors, Matti Cornille, Vincent Audenaert, Daan Grupping, Anis Dhewaju and Xander Scheyltjens
#Last updated: 18/04/2026
import numpy as np
import csv
import matplotlib.pyplot as plt

def absorption_coeffs(file):
    data = np.genfromtxt(file, skip_header=5)
    wavelen = data[:, 0] #in nanometer
    absorp_coeffs = data[:,1] #in cm^-1
    return wavelen, absorp_coeffs

def absorption(absorp_coeffs, d):
    absorbed_fraction = 1-np.exp(-absorp_coeffs*d)
    return absorbed_fraction


def solar_spectrum(file_name):
    with open(f"{file_name}", 'r') as f:
        file = csv.reader(f)
        header = next(file)  
        data = np.array(list(file), dtype='float64')
        wavelen = data[:, 0] #in nanometer
        irradiance = data[:,1] #in W/m^2/nm
    return wavelen, irradiance

def absorbed_power_spectrum(absorption_wavelen, absorp_coeffs, sun_wavelen, irradiance, d):
    common_wavelen = sun_wavelen
    alpha_interp = np.interp(sun_wavelen, absorption_wavelen, absorp_coeffs)
    absorbed_fraction = 1 - np.exp(-alpha_interp * d)
    absorbed_power_density = absorbed_fraction * irradiance
    absorbed_power_total = np.trapz(absorbed_power_density, common_wavelen)

    return common_wavelen, absorbed_power_density, absorbed_power_total

def plot(x_data, y_data, log_scale=False):
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)
    if log_scale==True:
        ax.loglog()
    ax.grid()


if __name__=="__main__":
    absorption_wavelen, absorp_coeffs = absorption_coeffs("Absorption_coefficients_water.txt")
    plot(absorption_wavelen, absorp_coeffs, True)
    plt.show()
    sun_wavelen, irradiance = solar_spectrum("AM1GH-standard.csv")
    plot(sun_wavelen, irradiance)
    plt.show()
    common_wavelen, absorbed_power_density, absorbed_power_total = absorbed_power_spectrum(absorption_wavelen, absorp_coeffs, sun_wavelen, irradiance, 1)
    plot(common_wavelen, absorbed_power_density)
    plt.show()
    print("The total absorbed power is: P_tot=", absorbed_power_total, "W/m^2")


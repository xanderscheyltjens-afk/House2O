#Simulation code for the House2O project
#Authors: Mona Soors, Matti Cornille, Vincent Audenaert, Daan Grupping, Anis Dhewaju and Xander Scheyltjens
#Last updated: 18/04/2026
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from numpy.core.umath import deg2rad
import pandas as pd
import matplotlib.ticker as ticker
import pvlib

#--------Reads external data------------------------------------------------------
def absorption_coeffs(file):
    #These first lines are if the code doesn't find the file even though it's in the same folder...
    #I'm just gonna use the full path. Change this if you try to run the code. Anywhare I use a path you need to give it the right one
    # here = os.path.dirname(os.path.abspath(__file__))
    # filepath = os.path.join(here, file)
    data = np.genfromtxt(f"{file}", skip_header=5)
    wavelen = data[:, 0] #in nanometer
    absorp_coeffs = data[:,1] #in cm^-1
    return wavelen, absorp_coeffs

def solar_spectrum(file_name):
    with open(f"{file_name}", 'r') as f:
        file = csv.reader(f)
        header = next(file)  
        data = np.array(list(file), dtype='float64')
        wavelen = data[:, 0] #in nanometer
        irradiance = data[:, 1] #in W/m^2/nm
    return wavelen, irradiance #No longer in use after switching to SMARTS

#-------Calculates reflection and absorption and stuff-----------------------------
def absorbed_power_spectrum(absorption_wavelen, absorp_coeffs, sun_wavelen, irradiance, d, aoi, glass=False):
    common_wavelen = sun_wavelen
    if glass:
        # Converts k to absorption coefficients, This is not the clearest code but I wanna reuse this function :)
        absorp_coeffs = 4*np.pi/(absorption_wavelen*10**(-7))*absorp_coeffs #We need to convert from nm to cm to match units

    alpha_interp = np.interp(sun_wavelen, absorption_wavelen, absorp_coeffs)
    d_eff = d/np.cos(aoi)
    absorbed_fraction = 1 - np.exp(-alpha_interp * d_eff)
    absorbed_power_density = absorbed_fraction * irradiance
    absorbed_power_total = np.trapz(absorbed_power_density, common_wavelen)

    return common_wavelen, absorbed_power_density, absorbed_power_total

def reflection_loss(index_1, index_2, aoi, irradiance):
    """Takes in the refractive indices of materials and angle of incidence and calculates the amount of light reflected using Snell's law and Fresnel equations"""
    # Convert degrees to radians
    aoi = deg2rad(aoi)
    # Calculates angle of transmission
    aot = np.arcsin(index_1*np.sin(aoi)/index_2)
    # Using Fresnel equations we determine reflectance for S and P polarisations
    #We assume the materials are non-magnetic as to substitute Z_i = Z_0/n_i where Z_0 is impedance of free space
    # Z_0 is then eliminated from the formulas so only refractive indices needed
    R_s = np.abs((index_1*np.cos(aoi)-index_2*np.cos(aot))/(index_1*np.cos(aoi)+index_2*np.cos(aot)))**2
    R_p = np.abs((index_1*np.cos(aot)-index_2*np.cos(aoi))/(index_1*np.cos(aot)+index_2*np.cos(aoi)))**2
    # Since sunlight is unpolarized we take the mean
    R_tot = (R_s+R_p)/2
    # Now transmission is easy
    T_tot = 1-R_tot
    # So the transmitted spectrum becomes
    new_irradiance = irradiance*T_tot
    return aot, new_irradiance

def air_glass_water(glass_filename, aoi_glass, irradiance): #Assumes a 5mm thick low iron glass pane 
    # Transition air to glass
    index_air = 1
    index_glass = 1.5168
    aoi_water, irradiance = reflection_loss(index_1= index_air, index_2= index_glass, aoi=aoi_glass, irradiance=irradiance )
    # Absorption glass
    with open(glass_filename, 'r') as f:
        file = csv.reader(f)
        header = next(file)  
        data = np.array(list(file), dtype='float64')
    glass_wavelen = data[:, 0] #in micrometer
    glass_wavelen *= 10**(3) #Set to nanometer
    extinction_coefficients_glass = data[:, 1] #in W/m^2/nm
    common_wavelen_glass, absorbed_power_density_glass, absorbed_power_total_glass = absorbed_power_spectrum(glass_wavelen, extinction_coefficients_glass, sun_wavelen, irradiance, d=0.5, aoi= aoi_water, glass=True)
    irradiance -= absorbed_power_density_glass
    # Transition glass to water
    index_water = 1.3325
    aot, irradiance = reflection_loss(index_1=index_glass, index_2=index_water, aoi=aoi_water, irradiance=irradiance)
    return irradiance, aot, absorbed_power_total_glass


#-------Visualisation--------------------------------------------------------------
def plot(x_data, y_data, log_scale=False):
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)
    if log_scale==True:
        ax.loglog()
    ax.grid()

def plot_spectrum(spectrum, solar, title=None):
    """
    Plot the computed spectral irradiance, broken down by component.
    Highlights the main atmospheric absorption bands.
    """
    wl = spectrum["wavelength_nm"]
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(wl, spectrum["poa_global"],      alpha=0.15, color="gold",   label="_nolegend_")
    ax.plot(wl, spectrum["poa_global"],      color="gold",   lw=2,   label="POA global (total)")
    ax.plot(wl, spectrum["poa_direct"],      color="orange", lw=1.5, label="POA direct")
    ax.plot(wl, spectrum["poa_sky_diffuse"], color="steelblue", lw=1.5, label="POA sky diffuse")
    ax.plot(wl, spectrum["poa_ground_diffuse"], color="sienna",    lw=1.2, label="POA ground diffuse", ls="--")
    ax.plot(wl, spectrum["dni_extra"],       color="gray",   lw=1,   label="Extraterrestrial", ls=":")

    # Mark major absorption bands
    absorption_bands = {
        "O₃ UV":   (280,  320, "violet"),
        "H₂O":     (930,  960, "cornflowerblue"),
        "H₂O":     (1100, 1165, "cornflowerblue"),
        "H₂O":     (1320, 1480, "cornflowerblue"),
        "CO₂+H₂O": (1760, 2000, "teal"),
        "H₂O":     (2500, 2900, "cornflowerblue"),
    }
    labelled = set()
    for label, (lo, hi, color) in absorption_bands.items():
        lbl = label if label not in labelled else "_nolegend_"
        ax.axvspan(lo, hi, alpha=0.12, color=color, label=lbl)
        labelled.add(label)

    # Visible light range indicator
    ax.axvspan(380, 700, alpha=0.06, color="lime", label="Visible (380–700 nm)")

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Spectral irradiance (W/m²/nm)", fontsize=12)
    ax.set_xlim(280, 2500)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    info = (
        f"Zenith: {solar['apparent_zenith']:.1f}°  |  "
        f"Azimuth: {solar['azimuth']:.1f}°  |  "
        f"AM: {solar['airmass']:.2f}  |  "
        f"AOI on surface: {solar['aoi']:.1f}°"
    )
    ax.set_title(
        title or f"Solar irradiance spectrum  —  {solar['datetime'].strftime('%Y-%m-%d %H:%M %Z')}",
        fontsize=13
    )
    ax.text(0.5, 0.97, info, ha="center", va="top",
            transform=ax.transAxes, fontsize=9, color="0.4")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig #Written by AI

#------SMARTS-implementation ------------------------------------------------------
# This is a python wrapper for SMARTS, taken from pySMARTS
def smartsAll(CMNT, ISPR, SPR, ALTIT, HEIGHT, LATIT, IATMOS, ATMOS, RH, TAIR, SEASON, TDAY, IH2O, W, IO3, IALT, AbO3, IGAS, ILOAD, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO,ApNO2, ApNO3, ApO3, ApSO2, qCO2, ISPCTR, AEROS, ALPHA1, ALPHA2, OMEGL, GG, ITURB, TAU5, BETA, BCHUEP, RANGE, VISI, TAU550, IALBDX, RHOX, ITILT, IALBDG,TILT, WAZIM,  RHOG, WLMN, WLMX, SUNCOR, SOLARC, IPRT, WPMN, WPMX, INTVL, IOUT, ICIRC, SLOPE, APERT, LIMIT, ISCAN, IFILT, WV1, WV2, STEP, FWHM, ILLUM,IUV, IMASS, ZENITH, AZIM, ELEV, AMASS, YEAR, MONTH, DAY, HOUR, LONGIT, ZONE, DSTEP, SMARTSPATH=None):
    r'''
    #data = smartsAll(CMNT, ISPR, SPR, ALTIT, HEIGHT, LATIT, IATMOS, ATMOS, RH, TAIR, SEASON, TDAY, IH2O, W, IO3, IALT, AbO3, IGAS, ILOAD, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO,ApNO2, ApNO3, ApO3, ApSO2, qCO2, ISPCTR, AEROS, ALPHA1, ALPHA2, OMEGL, GG, ITURB, TAU5, BETA, BCHUEP, RANGE, VISI, TAU550, IALBDX, RHOX, ITILT, IALBDG,TILT, WAZIM,  RHOG, WLMN, WLMX, SUNCOR, SOLARC, IPRT, WPMN, WPMX, INTVL, IOUT, ICIRC, SLOPE, APERT, LIMIT, ISCAN, IFILT, WV1, WV2, STEP, FWHM, ILLUM,IUV, IMASS, ZENITH, ELEV, AMASS, YEAR, MONTH, DAY, HOUR, LONGIT, ZONE, DSTEP)  
    # SMARTS Control Function
    # 
    #   Inputs:
    #       All variables are labeled according to the SMARTS 2.9.5 documentation.
    #       NOTICE THAT "IOTOT" is not an input variable of the function since is determined in the function 
    #       by sizing the IOUT variable.
    #   Outputs:
    #       data, is a matrix containing the outputs with as many rows as 
    #       wavelengths+1 (includes header) and as many columns as IOTOT+1 (column 1 is wavelengths)  
    #
    '''
    
    ## Init
    import os
    import pandas as pd
    import subprocess
    
    # Check if SMARTSPATH environment variable exists and change working
    # directory if it does.
    original_wd = None
    if 'SMARTSPATH' in os.environ:
        original_wd = os.getcwd()
        os.chdir(os.environ['SMARTSPATH'])
    else:
        if SMARTSPATH is not None:
            os.chdir(SMARTSPATH)
    
    try:
        os.remove('smarts295.inp.txt')
    except:
        pass
    try:
        os.remove('smarts295.out.txt')
    except:
        pass  
    try:       
        os.remove('smarts295.ext.txt')
    except:
        pass
    try:
        os.remove('smarts295.scn.txt')
    except:
        pass
        
    f = open('smarts295.inp.txt', 'w')
    
    IOTOT = len(IOUT.split())
    
    ## Card 1: Comment.
    if len(CMNT)>62:
        CMNT = CMNT[0:61] 

    CMNT = CMNT.replace(" ", "_")
    CMNT = "'"+CMNT+"'"
    print('{}' . format(CMNT), file=f)
    
    ## Card 2: Site Pressure
    print('{}'.format(ISPR), file=f)
    
    ##Card 2a:
    if ISPR=='0':
       # case '0' #Just input pressure.
        print('{}'.format(SPR), file=f)
    elif ISPR=='1':
        # case '1' #Input pressure, altitude and height.
        print('{} {} {}'.format(SPR, ALTIT, HEIGHT), file=f)
    elif ISPR=='2':
        #case '2' #Input lat, alt and height
        print('{} {} {}'.format(LATIT, ALTIT, HEIGHT), file=f)
    else:
        print("ISPR Error. ISPR should be 0, 1 or 2. Currently ISPR = ", ISPR)    
    
    ## Card 3: Atmosphere model
    print('{}'.format(IATMOS), file=f)
    
    ## Card 3a:
    if IATMOS=='0':
        #case '0' #Input TAIR, RH, SEASON, TDAY
        print('{} {} {} {}'.format(TAIR, RH, SEASON, TDAY), file=f)
    elif IATMOS=='1':        
        #case '1' #Input reference atmosphere
        ATMOS = "'"+ATMOS+"'"
        print('{}'.format(ATMOS), file=f)
    
    ## Card 4: Water vapor data
    print('{}'.format(IH2O), file=f)
    
    ## Card 4a
    if IH2O=='0':
        #case '0'
        print('{}'.format(W), file=f)
    elif IH2O=='1':
        #case '1'
        #The subcard 4a is skipped
        pass  #      print("")
    
    ## Card 5: Ozone abundance
    print('{}'.format(IO3), file=f)
    
    ## Card 5a
    if IO3=='0':
        #case '0'
        print('{} {}'.format(IALT, AbO3), file=f)
    elif IO3=='1':
        #case '1'
        #The subcard 5a is skipped and default values are used from selected 
        #reference atmosphere in Card 3. 
        pass #      print("")
    
    ## Card 6: Gaseous absorption and atmospheric pollution
    print('{}'.format(IGAS), file=f)
    
    ## Card 6a:  Option for tropospheric pollution
    if IGAS=='0':
        # case '0'
        print('{}'.format(ILOAD), file=f)

        ## Card 6b: Concentration of Pollutants        
        if ILOAD=='0':
            #case '0'
            print('{} {} {} {} {} {} {} {} {} {} '.format(ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, ApO3, ApSO2), file=f)
        elif ILOAD=='1':
            #case '1'
                #The subcard 6b is skipped and values of PRISTINE
                #ATMOSPHERIC conditions are assumed
            pass #     print("")
        elif ILOAD=='2' or ILOAD =='3' or ILOAD == '4':
            #case {'2', '3', '4'}
            #The subcard 6b is skipped and value of ILOAD will be used
            #as LIGHT POLLUTION (ILOAD = 2), MODERATE POLLUTION (ILOAD = 3), 
            #and SEVERE POLLUTION (ILOAD = 4).
            pass #     print("")
             
    elif IGAS=='1':
        #case '1'
        #The subcard 6a is skipped, and values are for default average
        #profiles.
        print("")
    
    ## Card 7:  CO2 columnar volumetric concentration (ppmv)
    print('{}'.format(qCO2), file=f)
    
    ## Card 7a: Option of proper extraterrestrial spectrum
    print('{}'.format(ISPCTR), file=f)
    
    ## Card 8: Aerosol model selection out of twelve
    AEROS = "'"+AEROS+"'"

    print('{}'.format(AEROS), file=f)
    
    ## Card 8a: If the aerosol model is 'USER' for user supplied information
    if AEROS=="'USER'":
        print('{} {} {} {}'.format(ALPHA1, ALPHA2, OMEGL, GG), file=f)
    else:
        #The subcard 8a is skipped
        pass #     print("")
    
    ## Card 9: Option to select turbidity model
    print('{}'.format(ITURB), file=f)
    
    ## Card 9a
    if ITURB=='0':
        #case '0'
        print('{}'.format(TAU5), file=f)
    elif ITURB=='1':
        #case '1'
        print('{}'.format(BETA), file=f)
    elif ITURB=='2':
        #case '2'
        print('{}'.format(BCHUEP), file=f)
    elif ITURB=='3':
        #case '3'
        print('{}'.format(RANGE), file=f)
    elif ITURB=='4':
        #case '4'
        print('{}'.format(VISI), file=f)
    elif ITURB=='5':
        #case '5'
        print('{}'.format(TAU550), file=f)
    else:
        print("Error: Card 9 needs to be input. Assign a valid value to ITURB = ", ITURB)
    
    ## Card 10:  Select zonal albedo
    print('{}'.format(IALBDX), file=f)
    
    ## Card 10a: Input fix broadband lambertial albedo RHOX
    if IALBDX == '-1':
        print('{}'.format(RHOX), file=f)
    else:
        pass #     print("")
        #The subcard 10a is skipped.
    
    ## Card 10b: Tilted surface calculation flag
    print('{}'.format(ITILT), file=f)
    
    ## Card 10c: Tilt surface calculation parameters
    if ITILT == '1':
        print('{} {} {}'.format(IALBDG, TILT, WAZIM), file=f)
        
        ##Card 10d: If tilt calculations are performed and zonal albedo of
        ##foreground.
        if IALBDG == '-1': 
            print('{}'.format(RHOG), file=f)
        else:
            pass #     print("")
            #The subcard is skipped 
    
    
    ## Card 11: Spectral ranges for calculations
    print('{} {} {} {}'.format(WLMN, WLMX, SUNCOR, SOLARC), file=f)
    
    ## Card 12: Output selection.
    print('{}'.format(IPRT), file=f)
    
    ## Card 12a: For spectral results (IPRT >= 1) 
    if float(IPRT) >= 1:
        print('{} {} {}'.format(WPMN, WPMX, INTVL), file=f)
        
        ## Card 12b & Card 12c: 
        if float(IPRT) == 2 or float(IPRT) == 3:
            print('{}'.format(IOTOT), file=f)
            print('{}'.format(IOUT), file=f)
        else:
            pass #     print("")
            #The subcards 12b and 12c are skipped.
    else:
        pass #     print("")
        #The subcard 12a is skipped
    
    ## Card 13: Circumsolar calculations
    print('{}'.format(ICIRC), file=f)
    
    ## Card 13a:  Simulated radiometer parameters
    if ICIRC == '1':
        print('{} {} {}'.format(SLOPE, APERT, LIMIT), file=f)
    else:
        pass #     print("")
        #The subcard 13a is skipped since no circumsolar calculations or
        #simulated radiometers have been requested.

    
    ## Card 14:  Scanning/Smoothing virtual filter postprocessor
    print('{}'.format(ISCAN), file=f)
    
    ## Card 14a:  Simulated radiometer parameters
    if ISCAN == '1': 
        print('{} {} {} {} {}'.format(IFILT, WV1, WV2, STEP, FWHM), file=f)
    else:
        pass #     print("")
        #The subcard 14a is skipped since no postprocessing is simulated.    
    
    ## Card 15: Illuminace, luminous efficacy and photosythetically active radiarion calculations
    print('{}'.format(ILLUM), file=f)
    
    ## Card 16: Special broadband UV calculations
    print('{}'.format(IUV), file=f)
    
    ## Card 17:  Option for solar position and air mass calculations
    print('{}'.format(IMASS), file=f)
    
    ## Card 17a: Solar position parameters:
    if IMASS=='0':
        #case '0' #Enter Zenith and Azimuth of the sun
        print('{} {}'.format(ZENITH, AZIM), file=f)
    elif IMASS=='1':
        #case '1' #Enter Elevation and Azimuth of the sun
        print('{} {}'.format(ELEV, AZIM), file=f)
    elif IMASS=='2':
        #case '2' #Enter air mass directly
        print('{}'.format(AMASS), file=f)
    elif IMASS=='3':
        #case '3' #Enter date, time and latitude
        print('{} {} {} {} {} {} {}'.format(YEAR, MONTH, DAY, HOUR, LATIT, LONGIT, ZONE), file=f)
    elif IMASS=='4':
        #case '4' #Enter date and time and step in min for a daily calculation.
        print('{}, {}, {}'.format(MONTH, LATIT, DSTEP), file=f)
    
    ## Input Finalization
    print('', file=f)
    f.close()
    
    ## Run SMARTS 2.9.5
    #dump = os.system('smarts295bat.exe')
    commands = ['smarts295bat', 'smarts295bat.exe']
    command = None
    for cmd in commands:
        if os.path.exists(cmd):
            command = cmd
            break

    if not command:
        print('Could not find SMARTS2 executable.')
        data = None
    else:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=open("output.txt", "w"), shell=True)
        p.wait()
        
        ## Read SMARTS 2.9.5 Output File
        try:
            data = pd.read_csv('smarts295.ext.txt', sep=r'\s+', ) #delim_whitespace=True
        except FileNotFoundError:
            data = pd.read_csv(os.path.join('OUTPUT','smarts295.ext.txt'), sep=r'\s+', ) 
    # try:
    #     os.remove('smarts295.inp.txt')
    # except:
    #     pass #     print("") 
    # try:
    #     os.remove('smarts295.out.txt')
    # except:
    #     pass #     print("")     
    # try:       
    #     os.remove('smarts295.ext.txt')
    # except:
    #     pass #     print("") 
    # try:
    #     os.remove('smarts295.scn.txt')
    # except:
    #     pass #     print("") 
    
    # Return to original working directory.    
    if original_wd:
        os.chdir(original_wd)

    return data

# ── Default atmospheric parameters ──────────────────────────────────────────── # This is also AI -> we will find our own values :)
# These are typical mid-latitude values. For a more accurate simulation of a
# specific location and season, use measured data from CAMS or similar sources.
DEFAULTS = {
    "precipitable_water":      1.42,   # cm    — atmospheric water vapour
    "ozone":                   0.344,  # atm-cm — total column ozone
    "aerosol_turbidity_500nm": 0.1,    # —      — aerosol optical depth at 500 nm
    "ground_albedo":           0.25,   # —      — ground reflectance (grass ~0.25)
    "surface_pressure":        101325, # Pa     — standard sea-level pressure
}

#Written by AI, uses smartsALL for spectrum computation
def compute_spectrum(
    latitude,
    longitude,
    datetime_input,
    tz="UTC",
    surface_tilt=90.0,
    surface_azimuth=180.0,
    precipitable_water=DEFAULTS["precipitable_water"],
    ozone=DEFAULTS["ozone"],
    aerosol_turbidity_500nm=DEFAULTS["aerosol_turbidity_500nm"],
    ground_albedo=DEFAULTS["ground_albedo"],
    surface_pressure=DEFAULTS["surface_pressure"],
    smarts_path=None,
):
    """
    Compute the solar irradiance spectrum at a given location and time
    using SMARTS 2.9.5 via pySMARTS.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees. North positive.
    longitude : float
        Longitude in decimal degrees. East positive.
        Note: SMARTS 2.9.5 uses East-positive convention, matching this input.
    datetime_input : str or datetime-like
        Date and time, e.g. '2024-06-21 12:00'. Interpreted in `tz`.
    tz : str
        Timezone string, e.g. 'Europe/Brussels', 'UTC'. Default 'UTC'.
    surface_tilt : float
        Tilt of the water surface from horizontal (degrees).
        0° = horizontal, 90° = vertical wall/window. Default 90°.
    surface_azimuth : float
        Compass direction the surface faces (degrees clockwise from North).
        180° = south-facing. Default 180°.
    precipitable_water : float
        Total column water vapour (cm). Strongly affects NIR. Default 1.42 cm.
    ozone : float
        Total column ozone (atm-cm). Default 0.344 atm-cm.
    aerosol_turbidity_500nm : float
        Aerosol optical depth at 500 nm (TAU5). Default 0.1.
    ground_albedo : float
        Broadband ground reflectance [0-1]. Default 0.25.
    surface_pressure : float
        Surface pressure (Pa). Default 101325 Pa (sea level).
    smarts_path : str or None
        Path to the SMARTS installation directory (containing smarts295bat or
        smarts295bat.exe). If None, the SMARTSPATH environment variable is used.

    Returns
    -------
    spectrum : pd.DataFrame
        DataFrame indexed by wavelength (nm) at 0.5 nm resolution with columns:
          wavelength_nm, dni_extra, dhi, dni,
          poa_direct, poa_sky_diffuse, poa_ground_diff, poa_global
        All irradiances in W/m²/nm.
    solar : dict
        Solar position and derived quantities:
          apparent_zenith, azimuth, airmass, aoi, sun_above_horizon, datetime
    """

    # ── Solar position (pvlib handles geometry; SMARTS handles radiative transfer) ──
    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz=tz)
    times = pd.DatetimeIndex([datetime_input], tz=tz)
    solpos = location.get_solarposition(times)

    apparent_zenith = float(solpos["apparent_zenith"].iloc[0])
    azimuth         = float(solpos["azimuth"].iloc[0])
    sun_above_horizon = apparent_zenith < 90.0

    aoi = float(pvlib.irradiance.aoi(surface_tilt, surface_azimuth, apparent_zenith, azimuth))

    if sun_above_horizon:
        relative_airmass = float(
            pvlib.atmosphere.get_relative_airmass(apparent_zenith, model="kastenyoung1989")
        )
    else:
        relative_airmass = np.nan

    solar = {
        "apparent_zenith":   apparent_zenith,
        "azimuth":           azimuth,
        "airmass":           relative_airmass,
        "aoi":               aoi,
        "sun_above_horizon": sun_above_horizon,
        "datetime":          times[0],
    }

    # ── Zero spectrum below the horizon ──────────────────────────────────────────
    if not sun_above_horizon:
        dummy_wl = np.arange(280.0, 4000.5, 0.5)
        zeros = np.zeros_like(dummy_wl)
        return pd.DataFrame({
            "wavelength_nm":   dummy_wl,
            "dni_extra":       zeros,
            "dhi":             zeros,
            "dni":             zeros,
            "poa_direct":      zeros,
            "poa_sky_diffuse": zeros,
            "poa_ground_diff": zeros,
            "poa_global":      zeros,
        }), solar

    # ── Prepare SMARTS inputs ─────────────────────────────────────────────────────
    t = times[0]

    # UTC offset as integer hours (what SMARTS calls ZONE)
    tz_offset = int(round(t.utcoffset().total_seconds() / 3600))

    # Decimal local time (SMARTS HOUR)
    hour_decimal = t.hour + t.minute / 60.0 + t.second / 3600.0

    # Season-appropriate reference atmosphere for temperature/pressure profiles.
    # We override water vapour and ozone ourselves (IH2O='0', IO3='0'), so this
    # only affects minor corrections like stratospheric temperature.
    atmos = 'MLS' if 4 <= t.month <= 9 else 'MLW'

    # Requested output columns (order determines iloc index in result):
    #   IOUT=1 → extraterrestrial   (col 1)
    #   IOUT=2 → DNI                (col 2)
    #   IOUT=3 → DHI                (col 3)
    #   IOUT=6 → direct tilted      (col 4)
    #   IOUT=7 → diffuse tilted     (col 5)
    #   IOUT=8 → global tilted      (col 6)
    IOUT_str = "1 2 3 6 7 8"

    # ── Call SMARTS ───────────────────────────────────────────────────────────────
    result = smartsAll(
        # Card 1: comment
        CMNT='House2O_spectrum',

        # Card 2/2a: pressure — ISPR=1 means we supply SPR (mbar), ALTIT (km), HEIGHT (km)
        ISPR='1',
        SPR=str(surface_pressure / 100.0),   # Pa → mbar
        ALTIT='0.0',                          # sea level; adjust for elevated sites (km)
        HEIGHT='0',
        LATIT=str(latitude),

        # Card 3/3a: reference atmosphere (provides T/P profiles; we override H2O and O3)
        IATMOS='1',
        ATMOS=atmos,
        RH='', TAIR='', SEASON='', TDAY='',  # unused when IATMOS='1'

        # Card 4/4a: precipitable water — IH2O=0 means we supply W directly
        IH2O='0',
        W=str(precipitable_water),

        # Card 5/5a: ozone — IO3=0 means we supply AbO3 directly
        IO3='0',
        IALT='0',                             # no altitude correction to ozone column
        AbO3=str(ozone),

        # Card 6/6a: gas absorption — IGAS=0, ILOAD=1 → pristine atmosphere defaults
        IGAS='0',
        ILOAD='1',
        ApCH2O='', ApCH4='', ApCO='', ApHNO2='', ApHNO3='',
        ApNO='', ApNO2='', ApNO3='', ApO3='', ApSO2='',

        # Card 7/7a: CO2 and extraterrestrial spectrum (0 = Gueymard 2004)
        qCO2='0.0',
        ISPCTR='0',

        # Card 8: aerosol model (tropospheric/rural, humidity dependent)
        AEROS='S&F_TROPO',
        ALPHA1='', ALPHA2='', OMEGL='', GG='',  # unused unless AEROS='USER'

        # Card 9/9a: turbidity — ITURB=0 means we supply TAU5 (AOD at 500 nm)
        ITURB='0',
        TAU5=str(aerosol_turbidity_500nm),
        BETA='', BCHUEP='', RANGE='', VISI='', TAU550='',

        # Card 10/10a: far-field (zonal) albedo — IALBDX='-1' lets us supply a float
        IALBDX='-1',
        RHOX=str(ground_albedo),

        # Card 10b/10c/10d: tilted surface — this is the key improvement over
        # SMARTSTimeLocation, which hardcodes TILT='0.0'
        ITILT='1',
        IALBDG='-1',                          # float foreground albedo
        TILT=str(surface_tilt),               # degrees from horizontal
        WAZIM=str(surface_azimuth),           # degrees clockwise from North
        RHOG=str(ground_albedo),

        # Card 11: spectral range and solar constant
        WLMN='280',
        WLMX='4000',
        SUNCOR='1.0',                         # overwritten by SMARTS when IMASS=3
        SOLARC='1367.0',

        # Card 12/12a/12b/12c: output format — IPRT=2 → spreadsheet-style ext.txt
        IPRT='2',
        WPMN='280',
        WPMX='4000',
        INTVL='.5',                           # 0.5 nm resolution output
        IOUT=IOUT_str,

        # Cards 13–16: circumsolar, scanning, illuminance, UV — all off
        ICIRC='0',
        SLOPE='', APERT='', LIMIT='',
        ISCAN='0',
        IFILT='', WV1='', WV2='', STEP='', FWHM='',
        ILLUM='0',
        IUV='0',

        # Card 17/17a: solar position from date+time+location (IMASS=3)
        IMASS='3',
        ZENITH='', AZIM='',
        ELEV='',
        AMASS='',
        YEAR=str(t.year),
        MONTH=str(t.month),
        DAY=str(t.day),
        HOUR=str(hour_decimal),
        LONGIT=str(longitude),                # East positive, same as our convention
        ZONE=str(tz_offset),
        DSTEP='',

        SMARTSPATH=smarts_path,
    )

    # ── Parse output ─────────────────────────────────────────────────────────────
    # Columns are positional — order matches IOUT_str exactly.
    # Col 0 is always Wvlgth (nm); subsequent cols follow IOUT order.
    wl        = result.iloc[:, 0].values  # wavelength (nm)
    dni_extra = result.iloc[:, 1].values  # IOUT=1: extraterrestrial
    dni       = result.iloc[:, 2].values  # IOUT=2: direct normal
    dhi       = result.iloc[:, 3].values  # IOUT=3: diffuse horizontal
    poa_dir   = result.iloc[:, 4].values  # IOUT=6: direct on tilted surface
    poa_dif   = result.iloc[:, 5].values  # IOUT=7: sky diffuse on tilted surface
    poa_glob  = result.iloc[:, 6].values  # IOUT=8: global on tilted surface

    # Ground-reflected component is what's left after direct + sky diffuse.
    # np.maximum guards against tiny floating-point negatives.
    poa_gnd = np.maximum(poa_glob - poa_dir - poa_dif, 0.0)

    spectrum = pd.DataFrame({
        "wavelength_nm":      wl,
        "dni_extra":          dni_extra,
        "dhi":                dhi,
        "dni":                dni,
        "poa_direct":         poa_dir,
        "poa_sky_diffuse":    poa_dif,
        "poa_ground_diffuse": poa_gnd,
        "poa_global":         poa_glob,
    })

    return spectrum, solar

#--------Test-environment----------------------------------------------------------------
if __name__=="__main__":
    # ----------Get solar spectrum (no longer used)-------------------------------------------------------------
    # sun_wavelen, irradiance = solar_spectrum("AM1GH-standard.csv")
    # plot(sun_wavelen, irradiance)
    # plt.show()

    #--------- Next section was AI showcase of its functions, with some personal adaptations---------------------
    # To be replaced with real data soon
    # ── Single-moment spectrum ─────────────────────────────────────────────────
    # Antwerp, Belgium — summer solstice, solar noon
    LAT, LON = 51.22, 4.40
    TZ = "Europe/Brussels"
    DATETIME = "2024-06-21 13:00"    # local time (CEST = UTC+2, solar noon ≈ 13:30)

    print(f"Computing spectrum for Antwerp, {DATETIME} CEST …")
    surface_tilt = 90
    spectrum, solar = compute_spectrum(
        latitude=LAT,
        longitude=LON,
        datetime_input=DATETIME,
        tz=TZ,
        surface_tilt=surface_tilt,       # vertical window
        surface_azimuth=180,   # south-facing
        # -- Atmospheric parameters (Belgian summer, typical) --
        precipitable_water=2.0,         # cm  — Belgium is fairly humid
        ozone=0.340,                    # atm-cm
        aerosol_turbidity_500nm=0.12,   # slightly urban
        ground_albedo=0.20,
        surface_pressure=101325,
        smarts_path=r"C:\Users\xande\OneDrive\Documents\GitHub\House2O\House2O\smarts-295-pc\SMARTS_295_PC"
    )

    if solar["sun_above_horizon"]:
        total_power = np.trapz(spectrum["poa_global"], spectrum["wavelength_nm"])
        print(f"  Solar zenith angle : {solar['apparent_zenith']:.1f}°")
        print(f"  Air mass           : {solar['airmass']:.3f}")
        print(f"  AOI on window      : {solar['aoi']:.1f}°")
        print(f"  Total POA power    : {total_power:.1f} W/m²")
    else:
        print("  Sun is below the horizon — zero irradiance.")

    fig1 = plot_spectrum(spectrum, solar)
    plt.show()
    sun_wavelen = spectrum["wavelength_nm"]
    #------ Import data for cloud modification factor (CMF) from PVGIS------------------------------
    #Generate the correct format for the date and time
    date, time = DATETIME.split(" ")
    year, month, day = date.split("-")
    hour, minute = time.split(":")

    PVGIS_DATETIME = month + day + ":" + hour + minute
    #Read file
    PVGIS_file = r"C:\Users\xande\OneDrive\Documents\GitHub\House2O\House2O\tmy_51.222_4.401_2005_2023.csv"
    with open(PVGIS_file, 'r') as f:
        file = csv.reader(f)
        PVGIS_data = np.array(list(file))
    #Find the right row where our date and time are correct, we ignore the year since it's irrelevant
    for idx, time in enumerate(PVGIS_data[:, 0]):
        time = str(time)
        if time[4:]==PVGIS_DATETIME:
            PVGIS_index=idx
    #Add a little check for if my numpy indexing was the other way around, if it's still here then I forgot to remove it
    if PVGIS_index==None:
        raise ValueError("Your indexing is wrong idiot")
    # Let's now adjust the direct and diffuse spectra from out clear sky model generated by SMARTS
    # We integrate the relvant spectra from SMARTS
    DNI = np.trapz(spectrum["dni"], spectrum["wavelength_nm"])
    DHI = np.trapz(spectrum["dhi"], spectrum["wavelength_nm"])
    # And then divide by data for cloudy days to get CMF
    direct_CMF = float(PVGIS_data[PVGIS_index, 4])/DNI
    diffuse_CMF = float(PVGIS_data[PVGIS_index, 5])/DHI
    # # Then we just adjust all spectra for our object
    # irradiance_direct = spectrum["poa_direct"]*direct_CMF
    # irradiance_sky_diffuse = spectrum["poa_sky_diffuse"]*diffuse_CMF
    # irradiance_ground_diffuse = spectrum["poa_ground_diffuse"]*diffuse_CMF
    # # And also sum them to get the global spectrum for cloudy days
    # irradiance_global = irradiance_direct+irradiance_ground_diffuse+irradiance_sky_diffuse

    # This was seemingly wrong but idk how. I'll keep using the old method for now
    global_CMF = float(PVGIS_data[PVGIS_index, 3])/(DNI*np.cos(np.radians(solar["apparent_zenith"]))+DHI) #We project the direct DNI on the horizontal plane to estimate the horizontal contribution
    # Then we just adjust all spectra for our object
    irradiance_direct = spectrum["poa_direct"]*direct_CMF
    irradiance_sky_diffuse = spectrum["poa_sky_diffuse"]*diffuse_CMF
    irradiance_global = spectrum["poa_global"]*global_CMF
    #From the difference of global-direct-sky diffuse irradiances we find the remaining ground diffuse irradiance on a cloudy day
    irradiance_ground_diffuse = irradiance_global-irradiance_direct-irradiance_sky_diffuse
    total_power_cloudy = np.trapz(irradiance_global, spectrum["wavelength_nm"])
    print(f"The total power with clouds is:{total_power_cloudy} W/m^2")

    #------ Get absorption coeffients for water ----------------------------------------------
    absorption_wavelen, absorp_coeffs = absorption_coeffs(r"C:\Users\xande\OneDrive\Documents\GitHub\House2O\House2O\Absorption_coefficients_water.txt")
    plot(absorption_wavelen, absorp_coeffs, True)
    plt.show()

    #------- Compute reflection losses and absorbed fraction by glass-------------------------------
    # We treat direct, sky and ground diffuse lighting seperately; starting with direct
    aoi_glass = solar["aoi"]
    glass_filename = r"C:\Users\xande\OneDrive\Documents\GitHub\House2O\House2O\Rubin-lowiron.csv"
    irradiance_direct, aot_direct, absorbed_power_total_glass_direct=air_glass_water(glass_filename=glass_filename, aoi_glass=aoi_glass, irradiance = irradiance_direct)
    # Now we compute sky diffuse
    aoi_sky_diffuse = 59.68-0.1388*surface_tilt+0.001497*surface_tilt**2 # This formula is from some paper I'm gonna add to the citations
    irradiance_sky_diffuse, aot_sky, absorbed_power_total_glass_sky=air_glass_water(glass_filename=glass_filename, aoi_glass=aoi_sky_diffuse, irradiance = irradiance_sky_diffuse)
    # And finally ground diffuse
    aoi_ground_diffuse = 90-0.5788*surface_tilt+0.002693*surface_tilt**2 # Same paper as above
    irradiance_ground_diffuse, aot_ground, absorbed_power_total_glass_ground=air_glass_water(glass_filename=glass_filename, aoi_glass=aoi_ground_diffuse, irradiance = irradiance_ground_diffuse)
    #We compute the losses from glass absorption for more insight
    absorbed_power_total_glass = absorbed_power_total_glass_direct+absorbed_power_total_glass_sky+absorbed_power_total_glass_ground

    #---------- Compute the absorption by water (what we want) ---------------------------------
    # I guess I'll keep the irradiance seperate for this too, although I'm not sure the angles are all correct and stuff
    # Those angles work for reflection, but idk if it's useable to compute path length throuhg water
    # Oh well, we did the same with the glass absportion, it's not like anyone has a better idea
    common_wavelen, absorbed_power_density_direct, absorbed_power_total_direct = absorbed_power_spectrum(absorption_wavelen, absorp_coeffs, sun_wavelen, irradiance_direct, d=15, aoi= aot_direct)
    _,absorbed_power_density_sky, absorbed_power_total_sky = absorbed_power_spectrum(absorption_wavelen, absorp_coeffs, sun_wavelen, irradiance_sky_diffuse, d=15, aoi= aot_sky)
    _,absorbed_power_density_ground, absorbed_power_total_ground = absorbed_power_spectrum(absorption_wavelen, absorp_coeffs, sun_wavelen, irradiance_ground_diffuse, d=15, aoi= aot_ground)
    # Now add all of them together to get the global absorption spectrum and total absorbed power
    absorbed_power_density = absorbed_power_density_direct+absorbed_power_density_sky+absorbed_power_density_ground
    absorbed_power_total = absorbed_power_total_direct+absorbed_power_total_sky+absorbed_power_total_ground
    # And then we see our total results!
    plot(common_wavelen, absorbed_power_density)
    plt.show()
    print("Glass stole about: ", absorbed_power_total_glass, "W/m^2")
    print("The total absorbed power is: P_tot=", absorbed_power_total, "W/m^2")
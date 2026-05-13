# Code House2O project: How much does the waterfall cool?
# Authors: Mona Soors, Matti Cornille, Vincent Audenaert, Daan Grupping, Anis Dhewaju and Xander Scheyltjens
# Last updated: 9/05/2026

import csv
import numpy as np
from House2O import BASE_DIR, general_use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----- Calculating the power needed for the waterfall for cooling ------------------------------------------------------
def h_weir(C_d=0.61, Re=400):

    """
    Compute the head over the weir for laminar flow.

    Parameters:
    C_d (float)             : Discharge coefficient (default: 0.61)
    Re (float)              : Reynolds number (default: 400 for laminar flow)

    Returns:
    float: Head over the weir in meters
    """
    mu = 0.001  # Dynamic viscosity of water in Pa.s
    rho = 997   # Density of water in kg/m^3
    g = 9.81   # Acceleration due to gravity in m/s²

    return ((3/2) * mu * Re / (C_d * rho * (2*g)**(1/2))) ** (2/3)

def flow_rate_waterfall(h_weir=None, C_d=0.61, b=2):
    """
    Compute the flow rate of the pump needed to have laminar flow in the waterfall.

    Parameters:
    h_weir (float)          : Head over the weir in meters (default: estimated for laminar flow)
    C_d (float)             : Discharge coefficient (default: 0.61)
    b (float)               : Width of the weir in meters (default: 2)
    g (float)               : Acceleration due to gravity in m/s² (default: 9.81)

    Returns:
    float: Flow rate in m^3/s
    """
    if h_weir is None:
        h_weir = h_weir()
    g=9.81
    return (2/3) * C_d * b * (2 * g * h_weir**3) ** 0.5

def pump_power_waterfall(flow_rate, eta=0.5, h=4.5):
    """
    Compute the power consumption of the pump for the waterfall.

    Parameters:
    flow_rate (float)       : Flow rate in m^3/s
    eta       (float)       : Efficiency of the pump (default: 0.5)
    h         (float)       : Height the water needs to be lifted in meters (default: 4.5, considering 2.44m as height of the house and the reservoir ~2m underground)
    
    Returns:
    float: Power consumption in Watts
    """
    rho = 997  # Density of water in kg/m^3
    g = 9.81   # Acceleration due to gravity in m/s²
    power = flow_rate * rho * g * h / eta
    return power

# ----- Calculating how much our waterfall cools ------------------------------------------------------
def waterfall_cooling_power(T_room,T_water, phi, h_c=0.5, A=2*2.44, L=2.45*10**6):
    """
    Compute the cooling effect of the waterfall

    Parameters:
    T_air (float)          : Air temperature in °C
    phi (float)            : Relative humidity of the air (0-1)
    h_c (float)            : Heat transfer coefficient (W/m^2K, default: 3 for free convection)
    A (float)              : Surface area (m^2)
    L (float)              : Latent heat of vaporization (J/kg)

    Returns:
    float: Cooling effect in Watts
    """
    rho_air=1.225; cp_air=1005
        
    # Calculate the saturation concentration of water in the air at the water temperature and at the air temperature
    a = 8.07131; b = 1730.63; c = 233.426
    p_sat_s = (10 ** (a - b / (T_water + c)))*133.322  # Saturation pressure in Pa
    c_s = p_sat_s*0.01801528 / (8.314 * (T_water + 273.15))  # Saturation concentration in kg/m^3
    p_sat_inf = (10 ** (a - b / (T_room + c)))*133.322  # Saturation pressure in Pa
    c_inf = phi*p_sat_inf*0.01801528 / (8.314 * (T_room + 273.15))  # Saturation concentration in kg/m^3

    Q_evap = h_c * A * L * (c_s - c_inf) / (rho_air * cp_air)  # Evaporative cooling power in Watts
    Q_conv = h_c * A * (T_room - T_water)  # Convective cooling power in Watts
    Q_tot = Q_evap + Q_conv  # Total cooling power in Watts

    return Q_tot, Q_evap, Q_conv

def interp(t_sec, data):
    """Linear interpolation of T_ref [°C] at time t [s]."""
    t_hr = t_sec / 3600.0
    for i in range(len(data) - 1):
        h0, T0 = data[i]
        h1, T1 = data[i + 1]
        if h0 <= t_hr <= h1:
            return T0 + (T1 - T0) * (t_hr - h0) / (h1 - h0)
    return data[-1][1]




def waterfall_room_simulation(DATETIME="2024-03-21 15:00", room_volume=4*4*2.44+2*1.56*4, room_surface=4*4+3*4*2.44+4*1.56+2.54*4,):
    """
    Euler integration of the room energy balance.

    ODE:  C_eff · dT/dt  =  C_eff · (dT_ref/dt)  −  Q_waterfall(T)

    Parameters:
    None

    Returns:
    None (but prints the cooling effect at each time step)
    """
    # ----- Temperatures should be read from script -----------------------
    T_water = 35 # Later nog aan te vullen met Daan zijn script.
    room_temperatures = [( 0, 22.0),( 1, 21.5),( 2, 21.0),( 3, 20.8),( 4, 20.5),( 5, 20.3),( 6, 20.5),( 7, 21.0),( 8, 22.0),( 9, 23.2),(10, 24.1),(11, 24.8),(12, 25.3),
    (13, 25.7),(14, 26.0),(15, 25.8),(16, 25.4),(17, 24.9),(18, 24.3),(19, 23.7),(20, 23.2),(21, 22.8),(22, 22.4),(23, 22.1),(24, 22.0)]


    #------ Import data from PVGIS------------------------------
    PVGIS_results, absorbed_power_density, absorbed_power_total = general_use()
    # Generate the correct format for the date and time
    date, time = DATETIME.split(" ")
    _, month, day = date.split("-")
    hour, _ = time.split(":")
    PVGIS_DATETIME = month + day + ":" + hour + "00"

    # Read file for temperatur and humidity
    PVGIS_file =  BASE_DIR + r"\tmy_51.222_4.401_2005_2023.csv"
    with open(PVGIS_file, 'r') as f:
        file = csv.reader(f)
        PVGIS_data = np.array(list(file))

    # Find the right row where our date and time are correct, we ignore the year
    for idx, time in enumerate(PVGIS_data[:, 0]):
        time = str(time)
        if time[4:]==PVGIS_DATETIME:
            PVGIS_index=idx

    # We read all of the info for our chosen date and time into their own variables
    PVGIS_results = {
    "temperature":      float(PVGIS_data[PVGIS_index, 1]), #in Celsius
    "relative_humidity":float(PVGIS_data[PVGIS_index, 2]),#in percent
    "GHI":              float(PVGIS_data[PVGIS_index, 3]), #in W/m^2
    "DNI":              float(PVGIS_data[PVGIS_index, 4]), #in W/m^2
    "DHI":              float(PVGIS_data[PVGIS_index, 5]), #in W/m^2
    "IR_radiation":     float(PVGIS_data[PVGIS_index, 6]), #in W/m^2
    "windspeed10m":     float(PVGIS_data[PVGIS_index, 7]), #in meters/second
    "wind_direction10m":float(PVGIS_data[PVGIS_index, 8]), #in degrees
    "surface_pressure": float(PVGIS_data[PVGIS_index, 9]), #in Pascal
    }
    # T_air = PVGIS_results["temperature"]
    phi = PVGIS_results["relative_humidity"]/100
    T_water = 35 # Later nog aan te vullen met Daan zijn script.



    # ------ Fluid properties (water at ~20 °C) ---------------------------------------
    rho_w = 998.0            # kg/m³
    mu_w = 1.002e-3          # Pa·s
    l_vap = 2.45e6           # J/kg   — latent heat of vaporisation
    # Air properties (~20 °C) 
    rho_air = 1.204          # kg/m³
    c_air = 1005.0         # J/(kg·K)

    # ── Wall / structural thermal mass ─────────────────────────────────────────
    wall_density   = 2000.0  # kg/m³  — concrete / brick
    wall_cp        = 880.0   # J/(kg·K)
    wall_depth     = 0.02    # m      — effective thermal penetration depth
                            #          (~2 cm for diurnal cycle, Fourier estimate)

    # ------ Effective thermal capacitance of the room (air + walls) ---------------------------------------
    C_air   = rho_air * room_volume * c_air
    m_wall  = wall_density * wall_depth * room_surface
    C_walls = m_wall * wall_cp
    C_eff   = C_air + C_walls


    # ------ MAIN SIMULATION LOOP ------------------------------------------------------
    t_end = room_temperatures[-1][0] * 3600.0
    dt_sim     = 60.0        # [s] Euler time-step (1 min)
    steps = int(t_end / dt_sim)

    T = interp(0, room_temperatures)       # start at the reference temperature
    results = []

    for step in range(steps):
        t = step * dt_sim

        T_ref_now  = interp(t,          room_temperatures)
        T_ref_next = interp(t + dt_sim, room_temperatures)
        dT_ref_dt  = (T_ref_next - T_ref_now) / dt_sim    # K/s

        Q_tot, Q_evap, Q_conv  = waterfall_cooling_power(T, T_water, phi)

        dT_dt = dT_ref_dt - Q_tot / C_eff

        results.append({
            "time_hr"       : round(t / 3600.0, 4),
            "T_simulated_C" : round(T,           3),
            "T_reference_C" : round(T_ref_now,   3),
            "dT_cooling_K"  : round(T_ref_now - T, 3),
            "Q_total_W"     : round(Q_tot,       2),
            "Q_evap_W"      : round(Q_evap,        2),
            "Q_conv_W"      : round(Q_conv,        2),
        })

        T += dT_dt * dt_sim
        # Safety clamp (should not be needed with reasonable inputs)
        # T = max(min(T, 60.0), -5.0)


    # ----- Print results in a readable format ------------------------------------------------------
    print(f"\n  {'Hour':>4}  {'T_sim °C':>9}  {'T_ref °C':>9}  "
          f"{'ΔT K':>7}  {'Q_tot W':>9}  {'Q_evap W':>9}")
    print("  " + "─" * 56)
    seen = set()
    for r in results:
        hr = int(r["time_hr"])
        if hr not in seen and (r["time_hr"] - hr) < (dt_sim / 3600):
            seen.add(hr)
            print(f"  {hr:>4}  {r['T_simulated_C']:>9.2f}  "
                  f"{r['T_reference_C']:>9.2f}  "
                  f"{r['dT_cooling_K']:>7.2f}  "
                  f"{r['Q_total_W']:>9.1f}  "
                  f"{r['Q_evap_W']:>9.1f}")
    print("  " + "─" * 56)
    vals = [r["dT_cooling_K"] for r in results]
    Qs   = [r["Q_total_W"]    for r in results]
    print(f"  Peak cooling below reference : {max(vals):.2f} K")
    print(f"  Mean cooling effect          : {sum(vals)/len(vals):.2f} K")
    print(f"  Mean cooling power           : {sum(Qs)/len(Qs):.1f} W")

    return results


#--------Test-environment----------------------------------------------------------------
if __name__=="__main__":
    h_weir_value = h_weir()
    print("Head over the weir with Re=400:", h_weir_value, "m")
    print("Flow rate waterfall =", flow_rate_waterfall(h_weir=h_weir_value), "m^3/s")
    print("Pump power waterfall =", pump_power_waterfall(flow_rate_waterfall(h_weir=h_weir_value)), "W")
    cooling_effect = waterfall_cooling_power(T_room=25,T_water=35,phi=0.5)
    print("The cooling effect of the waterfall is: ", cooling_effect, "W")
    waterfall_room_simulation()
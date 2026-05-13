# Code House2O project: Simulating the thermal performance of a waterfall cooling system with Daan's thermal simulation.
# Authors: Mona Soors, Matti Cornille, Vincent Audenaert, Daan Grupping, Anis Dhewaju and Xander Scheyltjens
# Last updated: 10/05/2026


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
def waterfall_cooling_power(T_room, T_water, phi, h_c=3, A=2*2.44, L=2.45*10**6):
    """
    Compute the cooling effect of the waterfall

    Parameters:
    T_room (float)         : Room temperature in °C
    T_water (float)        : Water temperature in °C
    phi (float)            : Relative humidity of the air (0-1)
    h_c (float)            : Heat transfer coefficient (W/m^2K, default: 3  for free convection)
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


# ----- Combining thermal and cooling simulation function ------------------------------------------------------
def waterfall_thermal_simulation(DATETIME="2024-03-21 15:00", room_volume=4*4*2.44+2*1.56*4, room_surface=4*4+3*4*2.44+4*1.56+2.54*4,):
    """
    Euler integration of the room energy balance.

    ODE:  C_eff · dT/dt  =  C_eff · (dT_ref/dt)  −  Q_waterfall(T)

    Parameters:
    None

    Returns:
    None (but prints the cooling effect at each time step)
    """
    # ----- Thermal simulation -----------------------
    # 1. Load the TMY data
    df_tmy = pd.read_csv('House2O/tmy_51.222_4.401_2005_2023.csv') 
    
    # Parse TMY time strings (Format: YYYYMMDD:HHMM)
    df_tmy['month'] = df_tmy['time(UTC)'].str[4:6].astype(int)
    df_tmy['day'] = df_tmy['time(UTC)'].str[6:8].astype(int)
    df_tmy['hour'] = df_tmy['time(UTC)'].str[9:11].astype(int)

    # 2. Load the Spring Power Cache data
    df_power = pd.read_csv('output_plots/OptimalAngle/Spring_fine_cache.csv')
    
    # Filter for a specific angle (e.g., 38 degrees) since the cache has multiple
    chosen_angle = 38
    df_power = df_power[df_power['angle'] == chosen_angle].copy()
    
    # Ensure date formats align for merging
    df_power['date_str'] = df_power['date'].astype(str)
    df_power['hour'] = df_power['hour'].astype(int)

    # 3. Construct a continuous timeline
    # We must construct a timeline including nights so dt remains exactly 3600 seconds.
    min_date = pd.to_datetime(df_power['date'].min())
    max_date = pd.to_datetime(df_power['date'].max())
    
    # Create an hourly range from the first day 00:00 to the last day 23:00
    date_range = pd.date_range(start=min_date, end=max_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
    
    df_sim = pd.DataFrame({'datetime': date_range})
    df_sim['month'] = df_sim['datetime'].dt.month
    df_sim['day'] = df_sim['datetime'].dt.day
    df_sim['hour'] = df_sim['datetime'].dt.hour
    df_sim['date_str'] = df_sim['datetime'].dt.strftime('%Y-%m-%d')

    # 4. Merge Data onto Timeline
    # Merge outside temperatures (ignoring the TMY year, joining on month/day/hour)
    df_sim = pd.merge(df_sim, df_tmy[['month', 'day', 'hour', 'T2m']], on=['month', 'day', 'hour'], how='left')
    
    # Merge the incoming solar power data (joining on exact date and hour)
    df_sim = pd.merge(df_sim, df_power[['date_str', 'hour', 'power']], on=['date_str', 'hour'], how='left')
    
    # Fill missing temperatures (just in case) and replace missing/nighttime power with 0 W/m2
    df_sim['T2m'] = df_sim['T2m'].ffill()
    df_sim['power'] = df_sim['power'].fillna(0)

    # Extract clean arrays for the simulation loop
    t_out_series = df_sim['T2m'].values
    power_per_m2_series = df_sim['power'].values
    hours = len(df_sim)


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



    # ------ Fluid properties (water at ~20 °C) ---------------------------------------
    rho_w = 998.0            # kg/m³
    mu_w = 1.002e-3          # Pa·s
    l_vap = 2.45e6  
    c_p_w = 4186.0          # J/(kg·K) — specific heat capacity of water
    # Air properties (~20 °C) 
    rho_air = 1.204          # kg/m³
    c_p_air = 1005.0         # J/(kg·K)

    # ── Wall / structural thermal mass ─────────────────────────────────────────
    wall_density   = 2000.0  # kg/m³  — concrete / brick
    wall_cp        = 880.0   # J/(kg·K)
    wall_depth     = 0.02    # m      — effective thermal penetration depth
                            #          (~2 cm for diurnal cycle, Fourier estimate)

    # ------ Effective thermal capacitance of the room (air + walls) ---------------------------------------
    C_air   = rho_air * room_volume * c_p_air
    m_wall  = wall_density * wall_depth * room_surface
    C_walls = m_wall * wall_cp
    C_eff   = C_air + C_walls

    # Volumes (m3) and Areas
    V_in = 51.52        # Volume of the house interior
    V_wat = 10           # Volume of the water compartment
    A_collector = 8.38   # Surface area of the absorber/panel (m2) <--- ADAPTS CSV DATA TO TOTAL VOLUME

    # Thermal Masses (Joules / Kelvin)
    C_in = V_in * rho_air * c_p_air 
    C_wat = V_wat * rho_w * c_p_w

    # Surface Areas (m2)
    A_in_out = 53.66      
    A_in_wat = 8.38       
    A_wat_out = 8.38       

    # U-values (W/m2.K)
    U_in_out = 0.15      
    U_in_wat = 1.1      
    U_wat_out = 1.1     


    # ------ MAIN SIMULATION LOOP ------------------------------------------------------
    # 6. Simulation Setup
    dt = 3600  # Time step = 1 hour
    T_in = np.zeros(hours)
    T_wat = np.zeros(hours)

    # Initial conditions (°C)
    T_in[0] = 20.0
    T_wat[0] = 30.0

    # 7. Simulation Loop (Euler Integration)
    for i in range(hours - 1):
        T_out = t_out_series[i]

        if T_in[i] > 30:
            P_wat_in = 0
        elif T_in[i] > 20:
            P_wat_in = power_per_m2_series[i] * A_collector * (30 - T_in[i])/10
        else:
            # Calculate total incoming power (W) based on current hour's W/m2 and collector area
            P_wat_in = power_per_m2_series[i] * A_collector
        
        # Heat flows between compartments (Watts)
        Q_in_out = U_in_out * A_in_out * (T_out - T_in[i])
        Q_wat_in = U_in_wat * A_in_wat * (T_wat[i] - T_in[i])
        Q_wat_out = U_wat_out * A_wat_out * (T_out - T_wat[i])
        Q_tot, Q_evap, Q_conv  = waterfall_cooling_power(T_in[i],T_wat[i],phi)

        # # Only apply waterfall cooling if it actually makes sense
        # if T_wat[i] < T_in[i]:  # water is cooler than room 
        #     cooling = Q_tot
        # else:
        #     cooling = 0.0
        
        # Temperature changes
        dT_in = (Q_in_out + Q_wat_in) / C_eff * dt
        dT_wat = (P_wat_in - Q_wat_in + Q_wat_out - Q_tot) / C_wat * dt
        
        T_in[i+1] = T_in[i] + dT_in
        T_wat[i+1] = T_wat[i] + dT_wat


    # ---------------------------------------------------------
    # 8. CALCULATE STATISTICS
    # ---------------------------------------------------------
    # Debgugging: Check that the time step is small enough for numerical stability (explicit Euler can be unstable if time step is too large)
    # Should all be << 1
    print(U_in_out * A_in_out / C_eff * dt)  
    print(U_in_wat * A_in_wat / C_eff * dt)

    avg_in = np.mean(T_in)
    min_in = np.min(T_in)
    max_in = np.max(T_in)
    total_swing = max_in - min_in
    std_dev = np.std(T_in)
    
    avg_out = np.mean(t_out_series)
    avg_diff = np.mean(T_in - t_out_series)
    
    # Calculate Average Daily Swing
    daily_swings = []
    for day_start in range(0, hours, 24):
        day_end = min(day_start + 24, hours)
        daily_segment = T_in[day_start:day_end]
        if len(daily_segment) > 0:
            daily_swings.append(np.max(daily_segment) - np.min(daily_segment))
    avg_daily_swing = np.mean(daily_swings)

    # Print to console
    print("\n--- SPRING SIMULATION STATISTICS ---")
    print(f"Average Inside Temp:      {avg_in:.2f} °C")
    print(f"Inside Temp Range:        {min_in:.2f} °C to {max_in:.2f} °C")
    print(f"Total Inside Swing:       {total_swing:.2f} °C")
    print(f"Average Daily Swing:      {avg_daily_swing:.2f} °C")
    print(f"Temperature Stability (σ):{std_dev:.2f} °C")
    print(f"Average Outside Temp:     {avg_out:.2f} °C")
    print(f"Avg Diff (Inside vs Out): +{avg_diff:.2f} °C")
    print("------------------------------------\n")

    # 9. Plot the results 
    plt.figure(figsize=(14, 7))
    plt.plot(df_sim['datetime'], t_out_series, label='Outside Temp', color='blue', alpha=0.5)
    plt.plot(df_sim['datetime'], T_in, label='Inside Temp', color='green')
    plt.plot(df_sim['datetime'], T_wat, label='Water Temp', color='red')
    
    # Create a string for the text box on the plot
    stats_text = (
        f"Avg Inside: {avg_in:.1f}°C\n"
        f"Daily Swing: {avg_daily_swing:.1f}°C\n"
        f"Avg Out: {avg_out:.1f}°C\n"
        f"Avg Diff: +{avg_diff:.1f}°C"
    )
    
    # Add text box to the top left of the plot
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('3-Compartment Thermal Simulation (Spring Period)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('thermal_simulation_spring.png')
    plt.show()

if __name__ == "__main__":
    waterfall_thermal_simulation()

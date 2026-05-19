import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def run_thermal_simulation():
    # 1. Load the TMY data
    df_tmy = pd.read_csv(r'C:\Users\Matti\Documents\GitHub\House2O\House2O\tmy_51.222_4.401_2005_2023.csv') 
    
    # Parse TMY time strings (Format: YYYYMMDD:HHMM)
    df_tmy['month'] = df_tmy['time(UTC)'].str[4:6].astype(int)
    df_tmy['day'] = df_tmy['time(UTC)'].str[6:8].astype(int)
    df_tmy['hour'] = df_tmy['time(UTC)'].str[9:11].astype(int)

    # 2. Load the Spring Power Cache data
    df_power = pd.read_csv(r'C:\Users\Matti\Documents\GitHub\House2O\Thermal_sim\AttemptingSomething\SpringDataOptimized_cache.csv') 
        
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
    df_sim = pd.merge(df_sim, df_tmy[['month', 'day', 'hour', 'T2m', 'WS10m']], on=['month', 'day', 'hour'], how='left')
    
    # Merge the incoming solar power data (joining on exact date and hour)
    # add the solar power per m2 absorbed by the glass itself
    df_sim = pd.merge(df_sim, df_power[['date_str', 'hour', 'power', 'power_glass']], on=['date_str', 'hour'], how='left') # also merge the power absorbed by the glass itself
    
    # Fill missing temperatures (just in case) and replace missing/nighttime power with 0 W/m2
    df_sim['T2m'] = df_sim['T2m'].ffill()
    df_sim['WS10m'] = df_sim['WS10m'].ffill()
    df_sim['power'] = df_sim['power'].fillna(0)
    df_sim['power_glass'] = df_sim['power_glass'].fillna(0)

    # Extract clean arrays for the simulation loop
    t_out_series = df_sim['T2m'].values
    wind_speed_series = df_sim['WS10m'].values
    power_per_m2_series = df_sim['power'].values
    power_glass_series = df_sim['power_glass'].values
    hours = len(df_sim)

    # Constants
    c_p_air = 1005      # Specific heat of air (J/kg.K)     # Literature
    rho_air = 1.225       # Density of air (kg/m3)          # Literature
    c_p_wat = 4184      # Specific heat of water (J/kg.K)   # Literature
    rho_wat = 1000      # Density of water (kg/m3)          # Literature
    c_p_plaster = 1090  # Needs checking (AI)
    rho_plaster = 1200  # Needs checking (AI)
    c_p_brick = 821     # Needs checking (AI)
    rho_brick = 1600    # Needs checking (AI)
    sigma = 5.67e-8     # Stefan-Boltzmann (W/m2.K4)        # Literature
    rho_glass = 2500    # Needs checking (AI)
    c_p_glass = 840    # Needs checking (AI)

    # Dimensions.
    V_in = 51.52        # Volume of the house interior     
    V_wat = 1.015       # Volume of the water compartment   
    V_basin = 4.19      # Volume of the cold water basin    
    V_fa = 0.0005       # Volume of waterfall

    A_collector = 10.15  # Surface area of the absorber
    A_walls1 = 51.52     # Surface area of the inside walls         # Calculated (still needs an extra check)
    A_walls2 = 51.52     # Surface area of the outside walls         # Calculated (still needs an extra check)
    A_wiso = 51.52     # Surface area of the outside walls         # Calculated (still needs an extra check)
    A_in_wat = 10.15     # only steep window
    A_ground = 16       # Surface area of the ground
    A_floor = 16        # Surface area of the floor
    A_basin = 12.57    # Surface area of the cold basin    # Assumption
    A_fa = 4.88         # surface area of the waterfall, m², Found in 6.2
    A_iso = 16

    d_iso = 0.15
    d_walls1 = 0.015    # Needs checking (AI)
    d_walls2 = 0.09     # Needs checking (AI)
    d_wiso = 0.12       # Needs checking (AI)
    d_ground = 0.3 
    d_basin = 1 
    # added to accomodate for windows: 
    d_glass = 0.05      # needs checking, pure assumption
    k_glass = 1.0       # needs checking (AI)

    # Assuming pumps keep water mass constant per compartment we can calculate thermal masses as if they were static volumes of water.
    C_in = V_in * rho_air * c_p_air                 # Thermal mass of the house interior
    C_wat = V_wat * rho_wat * c_p_wat               # Thermal mass of the water compartment (window)
    C_basin = V_basin * rho_wat * c_p_wat           # Thermal mass of the hot water basin
    C_floor = 3*10**6                               # Thermal mass of concrete floor
    C_ground = 84*10**6
    C_iso = 117600
    C_fa = V_fa*rho_wat*c_p_wat
    C_walls1 = A_walls1*d_walls1*rho_plaster*c_p_plaster
    C_walls2 = A_walls2*d_walls2*rho_brick*c_p_brick
    C_window1 = A_in_wat*d_glass*rho_glass*c_p_glass
    C_window2 = A_collector*d_glass*rho_glass*c_p_glass

    # Convection Coefficients (W/m2.K)
    # h_out is according to Gemini: 10 + 4v , thus calculating per step in loop
    h_in = 8
    h_out = 23
    h_walls1 = h_in
    h_walls2 = h_out
    h_fa = 3 # section 6.2
    h_floor = 10

    # Conduction Coefficients (W/m2.K)
    k_iso = 0.002 #polyurethane
    k_ground = 1.2  #Xander: Mona is echt GOATed, ik vond geen bronnen (maar ik had het ook snel opgegeven :))
    k_walls1 = 38 #Xander: gonna add full calculation in main soon (double check tho)
    k_walls2 = 5.666 #Xander: gonna add full calculation in main soon (double check tho)
    k_wiso = 0.208
    # Radiation Coefficients (W/m2.K)
    epsilon_w_in = 0.95       # Emissivity (this is water, right?)
    epsilon_w_out = 0.95      # Emissivity (this is water, right?)
    epsilon_fa = 0.95         # Emissivity
    epsilon_walls1 = 0.91      # Emissivity
    epsilon_walls2 = 0.93      # Emissivity
    epsilon_floor = 0.97      # Emissivity

    # advection coefficients (W/m2.K)
    
    # Pump capacities (This will need to go to a dynamic system.)
    m_dot_pump = 33.33/3600  # Mass flow rate of cold water (kg/s) -> chose slowest setting from Anis' document on google drive
    m_dot_floor = 33.33/3600

    # Constants for the waterfall
    L_v = 2.45e+6 # Latent heat, J/kg, found in 6.2
    M_water = 0.018 # Molar mass water, kg/mol, found in 6.2
    R = 8.314 # Gas constant, J/mol*K, found in 6.2
    A_ant = 8.07131 # Constant A in Antoine equation, found in 6.2
    B_ant = 1730.63 # Constant B in Antoine equation, found in 6.2
    C_ant = 233.426 - 273.15 # Constant C in Antoine equation, found in 6.2, convert for use with Kelvin

    # 6. Simulation Setup
    dt = 3600  # Time step = 1 hour
    T_in = np.zeros(hours)
    T_wat = np.zeros(hours)
    T_basin = np.zeros(hours)
    T_ground = np.zeros(hours)
    T_floor = np.zeros(hours)
    T_iso = np.zeros(hours)
    T_fa = np.zeros(hours)
    T_walls1 = np.zeros(hours)
    T_walls2 = np.zeros(hours)
    # added inside and outside windows
    T_window1 = np.zeros(hours)
    T_window2 = np.zeros(hours)
    

    # Initial conditions (K),  Needs update.
    T_in[0] = 273.15 + 20
    T_wat[0] = 273.15 + 20
    T_basin[0] = 273.15 + 20
    T_ground[0] = 273.15 + 20
    T_floor[0] = 273.15 + 20
    T_iso[0] = 273.15 + 20
    T_fa[0] = 273.15 + 20
    T_walls1[0] = 273.15 + 20
    T_walls2[0] = 273.15 + 20
    # add inside and outside windows
    T_window1[0] = 273.15 + 20
    T_window2[0] = 273.15 + 20

    current_T_in = T_in[0]
    current_T_wat = T_wat[0]
    current_T_basin = T_basin[0]
    current_T_ground = T_ground[0]
    current_T_floor = T_floor[0]
    current_T_iso = T_iso[0]
    current_T_fa = T_fa[0]
    current_T_walls1 = T_walls1[0]
    current_T_walls2 = T_walls2[0]
    # same here
    current_T_window1 = T_window1[0]
    current_T_window2 = T_window2[0]

    prev_waterfall_on = False # initialization of the waterfall control logic
    waterfall_active = np.zeros(hours, dtype= bool) 

    # 7. Simulation Loop (Euler Integration)
    for i in tqdm(range(hours - 1)):
        current_T_out = t_out_series[i] + 273.15 # Need kelvin for (T^4-T^4)
        # adjusting temperature? 30-> 25
        if T_in[i] > 273.15 + 30: # If temperature rises, a screen will be placed over the window to reduce incoming solar radiation.
            P_sun = 0
        elif T_in[i] > 273.15 + 20:
            P_sun = power_per_m2_series[i] * (273.15 + 30 - T_in[i])/10
        else:
            # Calculate incoming power (W/m²) based on current hour
            P_sun = power_per_m2_series[i]
        
        P_sun_glass = power_glass_series[i] # Calculate power absorbed by the glass itself, which reduces the power reaching the interior

        # (7b). Control logic (evaluated once per hour)
        T_setpoint_low = 273.15 + 23
        T_setpoint_high = 273.15 + 23

        too_cold = T_in[i] < T_setpoint_low
        too_hot = T_in[i] > T_setpoint_high
        
        comfortable = not too_cold and not too_hot

        floor_heating_on = too_cold
        waterfall_on = too_hot

        m_dot_floor_active = m_dot_floor if floor_heating_on else 0.0
        m_dot_fa_active = m_dot_pump if waterfall_on else 0.0       

        if waterfall_on and not prev_waterfall_on:
                    current_T_fa = current_T_basin
        prev_waterfall_on = waterfall_on
        waterfall_active[i] = waterfall_on


        # Heat flows between compartments (Watts)
        for _ in range(3600): # Sub-iterations for better stability
            # Calculations for the waterfall:
            # And gate evaporation
            if current_T_in < 293.15 or not waterfall_on:
                m_dot_evap = 0
            else:
                p_sat_fa = (10**(A_ant - (B_ant / (C_ant+current_T_fa)))) * 133.322  #Antoines formula, 133.322 converts mmhg -> Pa
                p_sat_in = (10**(A_ant - (B_ant / (C_ant+current_T_in)))) * 133.322  #Antoines formula, 133.322 converts mmhg -> Pa
                c_fa = p_sat_fa * M_water / (R * current_T_fa) # Ideal gas law waterfall
                c_in = p_sat_in * M_water / (R * current_T_in) # Ideal gas law room
                m_dot_evap = h_fa * A_fa * (c_fa - c_in)/ (rho_air * c_p_air)
                
            # Temperature changes
            # new: change in temperature for the outside window
            dT_window2 = A_collector * (
                P_sun_glass +
                h_out * (current_T_out - current_T_window2) +
                sigma * epsilon_w_out * (current_T_out**4 - current_T_window2**4) +
                k_glass / d_glass * (current_T_wat - current_T_window2)
                )



            dT_wat = A_collector * (
                P_sun + 
                k_glass / d_glass * (current_T_window2 - current_T_wat) + # from outer glass
                k_glass / d_glass * (current_T_window1 - current_T_wat) # from inner glass
                )+ (
                m_dot_pump * c_p_wat * (current_T_basin - current_T_wat) )  # evap removed as it should only be in the waterfall equation

            dT_window1 = A_collector * (
                k_glass / d_glass * (current_T_wat - current_T_window1) + # from water
                h_in * (current_T_in - current_T_window1) + # from inside air
                sigma * epsilon_w_in * (current_T_in**4 - current_T_window1**4) # radiation from inside air
            )

            dT_in = A_in_wat * (
                h_in * (current_T_window1 - current_T_in) +
                sigma * epsilon_w_in * (current_T_window1**4 - current_T_in**4)) \
                + A_fa * (
                h_fa * (current_T_fa - current_T_in) +
                sigma * epsilon_fa * (current_T_fa**4 - current_T_in**4))\
                + A_walls1 * (
                h_walls1 * (current_T_walls1 - current_T_in) +
                sigma * epsilon_walls1 * (current_T_walls1**4 - current_T_in**4))\
                + A_floor * (
                h_floor * (current_T_floor - current_T_in) +
                sigma * epsilon_floor * (current_T_floor**4 - current_T_in**4)    
                )
            
            dT_walls1 = A_walls1 * (
                h_walls1 * (current_T_in - current_T_walls1) +
                sigma * epsilon_walls1 * (current_T_in**4 - current_T_walls1**4) +
                k_walls1 / d_walls1 * (current_T_in - current_T_walls1) +
                k_wiso / d_wiso * (current_T_walls2 - current_T_walls1)
            )

            dT_walls2 = A_walls2 * (
                h_walls2 * (current_T_out - current_T_walls2) +
                sigma * epsilon_walls2 * (current_T_out**4 - current_T_walls2**4) +
                k_walls2 / d_walls2 * (current_T_out - current_T_walls2) +
                k_wiso / d_wiso * (current_T_walls1 - current_T_walls2)
            )

            # Adjusted to only use when it's too hot
            dT_fa = m_dot_fa_active * c_p_wat * (current_T_basin - current_T_fa) +\
                waterfall_on * (h_fa * A_fa * (current_T_in - current_T_fa) +\
                sigma * epsilon_fa * (current_T_in**4 - current_T_fa**4)) +\
                - m_dot_evap * L_v

            dT_floor = m_dot_floor_active * c_p_wat * (current_T_basin - current_T_floor) +\
                A_floor * (
                h_floor * (current_T_in - current_T_floor) +
                sigma * epsilon_floor * (current_T_in**4 - current_T_floor**4) +
                k_iso * (current_T_iso - current_T_floor)/d_iso)
            
            dT_iso = k_iso * A_iso / d_iso * (current_T_floor + current_T_ground - 2 * current_T_iso)
            
            dT_ground = k_ground * A_ground / d_ground * (current_T_iso + current_T_basin - 2 * current_T_ground)

            dT_basin = m_dot_pump * c_p_wat * (current_T_wat - current_T_basin) +\
                m_dot_floor_active * c_p_wat * (current_T_floor - current_T_basin) +\
                m_dot_fa_active * c_p_wat * (current_T_fa - current_T_basin) +\
                k_ground * A_basin / d_basin * (current_T_ground - current_T_basin) 

            current_T_wat += dT_wat * dt / (C_wat * 3600)
            current_T_in += dT_in * dt / (C_in * 3600)
            current_T_floor += dT_floor * dt / (C_floor * 3600)
            current_T_basin += dT_basin * dt / (C_basin * 3600)
            current_T_ground += dT_ground * dt / (C_ground * 3600)
            current_T_iso += dT_iso * dt / (C_iso * 3600)
            current_T_fa += dT_fa * dt / (C_fa * 3600)
            current_T_walls1 += dT_walls1 * dt / (C_walls1 * 3600)
            current_T_walls2 += dT_walls2 * dt / (C_walls2 * 3600)

            current_T_window1 += dT_window1 * dt / (C_window1 * 3600) # update C
            current_T_window2 += dT_window2 * dt / (C_window2 * 3600) # 

        
        T_in[i+1] = current_T_in
        T_wat[i+1] = current_T_wat
        T_floor[i+1] = current_T_floor
        T_basin[i+1] = current_T_basin
        T_ground[i+1] = current_T_ground
        T_iso[i+1] = current_T_iso
        T_fa[i+1] = current_T_fa
        T_walls1[i+1] = current_T_walls1
        T_walls2[i+1] = current_T_walls2

        T_window1[i+1] = current_T_window1
        T_window2[i+1] = current_T_window2


    #
    # Restore temperatures to Celsius 
    #
    T_in -= 273.15
    T_wat -= 273.15
    T_floor -= 273.15
    T_basin -= 273.15
    T_ground -= 273.15
    T_iso -= 273.15
    T_fa -= 273.15
    T_walls1 -= 273.15
    T_walls2 -= 273.15

    T_window1 -= 273.15
    T_window2 -= 273.15

    # ---------------------------------------------------------
    # 8. CALCULATE STATISTICS
    # ---------------------------------------------------------
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

    # 9. Plot the results with interactive checkboxes
    T_fa_plot = T_fa.copy()
    T_fa_plot[~waterfall_active] = np.nan # keep track of the times where the waterfall is active to plot correctly

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.25) # Make room on the left for checkboxes

    # Store lines in a dictionary to access them in the toggle function
    lines = {
        'Outside Temp': ax.plot(df_sim['datetime'], t_out_series, label='Outside Temp', color='blue', alpha=0.4)[0],
        'Inside Temp':  ax.plot(df_sim['datetime'], T_in, label='Inside Temp', color='green', linewidth=2)[0],
        'Water Temp':   ax.plot(df_sim['datetime'], T_wat, label='Water Temp', color='red')[0],
        'Floor Temp':   ax.plot(df_sim['datetime'], T_floor, label='Floor Temp', color='orange')[0],
        'Basin Temp':   ax.plot(df_sim['datetime'], T_basin, label='Basin Temp', color='purple')[0],
        'Ground Temp':  ax.plot(df_sim['datetime'], T_ground, label='Ground Temp', color='gray')[0],
        'Waterfall Temp': ax.plot(df_sim['datetime'], T_fa_plot, label='Waterfall Temp', color='cyan', alpha=0.7)[0],
        'Walls1 Temp':    ax.plot(df_sim['datetime'], T_walls1, label='Walls1 Temp', color='brown', alpha=0.7)[0],
        'Walls2 Temp':    ax.plot(df_sim['datetime'], T_walls2, label='Walls2 Temp', color='brown', alpha=0.7)[0],
        'Window1 Temp':   ax.plot(df_sim['datetime'], T_window1, label='Window1 Temp', color='lightblue', alpha=0.7)[0], # added 
        'Window2 Temp':   ax.plot(df_sim['datetime'], T_window2, label='Window2 Temp', color='lightblue', alpha=0.7)[0] # added
    }

    # Text box for stats
    stats_text = (
        f"Avg Inside: {avg_in:.1f}°C\n"
        f"Daily Swing: {avg_daily_swing:.1f}°C\n"
        f"Avg Out: {avg_out:.1f}°C\n"
        f"Avg Diff: +{avg_diff:.1f}°C"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Thermal Simulation Components (Toggle visibility on the left)')
    ax.grid(True)

    # Define checkbox positions [left, bottom, width, height]
    from matplotlib.widgets import CheckButtons
    rax = plt.axes([0.02, 0.4, 0.15, 0.35])
    labels = list(lines.keys())
    # Set default visibility (True for all)
    visibility = [True] * len(labels)
    check = CheckButtons(rax, labels, visibility)

    # Callback function to toggle lines
    def toggle_visibility(label):
        lines[label].set_visible(not lines[label].get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle_visibility)

    plt.savefig(r'C:\Users\Matti\Documents\GitHub\House2O\Thermal_sim\AttemptingSomething\test_thermal_simulation_added_windows_different_panelsettings.png')
    plt.show()

if __name__ == "__main__":
    run_thermal_simulation()
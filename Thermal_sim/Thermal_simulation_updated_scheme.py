import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_thermal_simulation():
    # 1. Load the TMY data
    df_tmy = pd.read_csv('House2O/tmy_51.222_4.401_2005_2023.csv') 
    
    # Parse TMY time strings (Format: YYYYMMDD:HHMM)
    df_tmy['month'] = df_tmy['time(UTC)'].str[4:6].astype(int)
    df_tmy['day'] = df_tmy['time(UTC)'].str[6:8].astype(int)
    df_tmy['hour'] = df_tmy['time(UTC)'].str[9:11].astype(int)

    # 2. Load the Spring Power Cache data
    df_power = pd.read_csv('output_plots/OptimalAngle/Spring_fine_cache.csv') # Needs update
    
    # Filter for a specific angle (e.g., 38 degrees) since the cache has multiple
    chosen_angle = 38  # tilt angle (Later add compas angle)
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
    df_sim = pd.merge(df_sim, df_tmy[['month', 'day', 'hour', 'T2m', 'WS10m']], on=['month', 'day', 'hour'], how='left')
    
    # Merge the incoming solar power data (joining on exact date and hour)
    df_sim = pd.merge(df_sim, df_power[['date_str', 'hour', 'power']], on=['date_str', 'hour'], how='left')
    
    # Fill missing temperatures (just in case) and replace missing/nighttime power with 0 W/m2
    df_sim['T2m'] = df_sim['T2m'].ffill()
    df_sim['WS10m'] = df_sim['WS10m'].ffill()
    df_sim['power'] = df_sim['power'].fillna(0)

    # Extract clean arrays for the simulation loop
    t_out_series = df_sim['T2m'].values
    wind_speed_series = df_sim['WS10m'].values
    power_per_m2_series = df_sim['power'].values
    hours = len(df_sim)

    # Constants
    c_p_air = 1005      # Specific heat of air (J/kg.K)     # Literature
    rho_air = 1.225       # Density of air (kg/m3)          # Literature
    c_p_wat = 4184      # Specific heat of water (J/kg.K)   # Literature
    rho_wat = 1000      # Density of water (kg/m3)          # Literature
    sigma = 5.67e-8     # Stefan-Boltzmann (W/m2.K4)        # Literature


    # Dimensions.
    V_in = 51.52        # Volume of the house interior      # Calculated (still needs an extra check)
    V_wat = 1           # Volume of the water compartment   # Assumption, (need to get data still)
    V_basin_cold = 1    # Volume of the cold water basin    # Assumption, (need to get data still)
    V_basin_hot = 1     # Volume of the hot water basin     # Assumption, (need to get data still)

    A_collector = 8.38  # Surface area of the absorber      # Calculated (still needs an extra check)
    A_walls = 51.52     # Surface area of the walls         # Calculated (still needs an extra check)
    A_in_wat = 8.38     # only steep window
    A_ground = 16       # Surface area of the ground        # Assumption
    A_floor = 16        # Surface area of the floor         # Assumption
    A_basin_cold = 1    # Surface area of the cold basin    # Assumption
    A_basin_hot = 1     # Surface area of the hot basin     # Assumption

    d_floor = 0.2       # Thickness of the floor            # Assumption
    d_basin_cold = 0.05 # Thickness of the cold basin       # Assumption
    d_basin_hot = 0.05  # Thickness of the hot basin        # Assumption
    d_walls = 0.2
    d_ground = 1


    # Assuming pumps keep water mass constant per compartment we can calculate thermal masses as if they were static volumes of water.
    C_in = V_in * rho_air * c_p_air                 # Thermal mass of the house interior
    C_wat = V_wat * rho_wat * c_p_wat               # Thermal mass of the water compartment (window)
    C_basin_cold = V_basin_cold * rho_wat * c_p_wat # Thermal mass of the cold water basin
    C_basin_hot = V_basin_hot * rho_wat * c_p_wat   # Thermal mass of the hot water basin
    C_floor = 10**6 #Temporary
    C_ground = 10**6 #Temporary

    # Convection Coefficients (W/m2.K)
    # h_out is according to Gemini: 10 + 4v , thus calculating per step in loop
    h_in = 10
    h_floor = 10
    h_basin_cold = 10

    # Conduction Coefficients (W/m2.K)
    k_walls = 0.03
    k_ground = 1
    # Radiation Coefficients (W/m2.K)
    epsilon = 1       # Emissivity
    # advection coefficients (W/m2.K)

    # 6. Simulation Setup
    dt = 3600  # Time step = 1 hour
    T_in = np.zeros(hours)
    T_wat = np.zeros(hours)
    T_basin_cold = np.zeros(hours)
    T_basin_hot = np.zeros(hours)
    T_ground = np.zeros(hours)
    T_floor = np.zeros(hours)
    
    # Pump capacities (This will need to go to a dynamic system.)
    m_dot_pump_cold = 0.001  # Mass flow rate of cold water (kg/s)
    m_dot_pump_hot = 0.001   # Mass flow rate of hot water (kg/s)
    m_dot_floor = 0.001      # Mass flow rate through the floor (kg/s)

    # Initial conditions (K),  Needs update.
    T_in[0] = 273.15 +20.0
    T_wat[0] = 273.15 +30.0
    T_basin_cold[0] = 273.15 + 15.0
    T_basin_hot[0] = 273.15 + 40.0
    T_ground[0] = 273.15 + 10.0
    T_floor[0] = 273.15 + 25.0

    current_T_in = T_in[0]
    current_T_wat = T_wat[0]
    current_T_basin_cold = T_basin_cold[0]
    current_T_basin_hot = T_basin_hot[0]
    current_T_ground = T_ground[0]
    current_T_floor = T_floor[0]


    # 7. Simulation Loop (Euler Integration)
    for i in range(hours - 1):
        current_T_out = t_out_series[i] + 273.15 # Need kelvin for (T^4-T^4)
        h_out = 1#0# + 4 * wind_speed_series[i] # Update h_out based on current wind speed
        if T_in[i] > 273.15 + 30:
            P_wat_in = 0
        elif T_in[i] > 273.15 + 20:
            P_wat_in = power_per_m2_series[i] * (273.15 + 30 - T_in[i])/10
        else:
            # Calculate incoming power (W/m²) based on current hour
            P_wat_in = power_per_m2_series[i]
        
        # Heat flows between compartments (Watts)
        for _ in range(60): # Sub-iterations for better stability
            # Temperature changes
            dT_wat = A_collector * (
                P_wat_in + 
                h_in * (current_T_in-current_T_wat) +
                h_out * (current_T_out - current_T_wat) + 
                sigma * epsilon * (current_T_in**4 - current_T_wat**4) +
                sigma * epsilon * (current_T_out**4 - current_T_wat**4)) + (
                m_dot_pump_cold * c_p_wat * (current_T_basin_cold - current_T_wat) +
                m_dot_pump_hot * c_p_wat * (current_T_basin_hot - current_T_wat) )
            
            dT_in = A_in_wat * (
                h_in * (current_T_wat - current_T_in) +
                sigma * epsilon * (current_T_wat**4 - current_T_in**4)) \
                + (
                k_walls * A_walls * (current_T_out - current_T_in)/d_walls +
                k_ground * A_ground * (current_T_ground - current_T_in)/d_ground) \
                + A_floor * (
                h_floor * (current_T_floor - current_T_in) +
                sigma * epsilon * (current_T_floor**4 - current_T_in**4)) \
                + h_basin_cold * A_basin_cold * (current_T_basin_cold - current_T_in) + \
                sigma * epsilon * A_basin_cold * (current_T_basin_cold**4 - current_T_in**4)

            dT_floor = m_dot_floor * c_p_wat * (current_T_basin_hot - current_T_floor) +\
                A_floor * (
                h_floor * (current_T_in - current_T_floor) +
                sigma * epsilon * (current_T_in**4 - current_T_floor**4) +
                k_ground * (current_T_ground - current_T_floor)/d_floor)

            dT_basin_hot = m_dot_pump_hot * c_p_wat * (current_T_wat - current_T_basin_hot) +\
                m_dot_floor * c_p_wat * (current_T_floor - current_T_basin_hot) +\
                k_ground * A_basin_hot * (current_T_ground - current_T_basin_hot)/d_basin_hot
            
            dT_basin_cold =  m_dot_pump_cold * c_p_wat * (current_T_wat - current_T_basin_cold) +\
                h_basin_cold * A_basin_cold * (current_T_in - current_T_basin_cold) +\
                sigma * epsilon * A_basin_cold * (current_T_in**4 - current_T_basin_cold**4) +\
                k_ground * A_basin_cold * (current_T_ground - current_T_basin_cold)/d_basin_cold
            
            dT_ground = k_ground * (
                A_ground * (current_T_in - current_T_ground)/d_ground +
                A_floor * (current_T_floor - current_T_ground)/d_floor +
                A_basin_cold * (current_T_basin_cold - current_T_ground)/d_basin_cold +
                A_basin_hot * (current_T_basin_hot - current_T_ground)/d_basin_hot
            )
            current_T_wat += dT_wat * dt / (C_wat * 60)
            current_T_in += dT_in * dt / (C_in * 60)
            current_T_floor += dT_floor * dt / (C_floor * 60)
            current_T_basin_hot += dT_basin_hot * dt / (C_basin_hot * 60)
            current_T_basin_cold += dT_basin_cold * dt / (C_basin_cold * 60)
            current_T_ground += dT_ground * dt / (C_ground * 60)

        T_in[i+1] = current_T_in
        T_wat[i+1] = current_T_wat
        T_floor[i+1] = current_T_floor
        T_basin_hot[i+1] = current_T_basin_hot
        T_basin_cold[i+1] = current_T_basin_cold
        T_ground[i+1] = current_T_ground
    #
    # Restore temperatures to Celsius 
    #
    T_in -= 273.15
    T_wat -= 273.15
    T_floor -= 273.15
    T_basin_hot -= 273.15
    T_basin_cold -= 273.15
    T_ground -= 273.15


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

    # 9. Plot the results 
    plt.figure(figsize=(14, 7))
    plt.plot(df_sim['datetime'], t_out_series, label='Outside Temp', color='blue', alpha=0.5)
    plt.plot(df_sim['datetime'], T_in, label='Inside Temp', color='green')
    plt.plot(df_sim['datetime'], T_wat, label='Water Temp', color='red')
    plt.plot(df_sim['datetime'], T_floor, label='Floor Temp', color='orange')
    plt.plot(df_sim['datetime'], T_basin_hot, label='Basin Hot Temp', color='purple')
    plt.plot(df_sim['datetime'], T_basin_cold, label='Basin Cold Temp', color='brown')
    plt.plot(df_sim['datetime'], T_ground, label='Ground Temp', color='gray')

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
    run_thermal_simulation()
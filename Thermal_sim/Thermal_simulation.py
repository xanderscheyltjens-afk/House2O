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

    # 5. Define Parameters
    c_p_air = 1005      # Specific heat of air (J/kg.K)
    rho_air = 1.225       # Density of air (kg/m3)
    c_p_wat = 4184      # Specific heat of water (J/kg.K)
    rho_wat = 1000      # Density of water (kg/m3)

    # Volumes (m3) and Areas
    V_in = 51.52        # Volume of the house interior
    V_wat = 10           # Volume of the water compartment
    A_collector = 8.38   # Surface area of the absorber/panel (m2) <--- ADAPTS CSV DATA TO TOTAL VOLUME

    # Thermal Masses (Joules / Kelvin)
    C_in = V_in * rho_air * c_p_air * 5 
    C_wat = V_wat * rho_wat * c_p_wat

    # Surface Areas (m2)
    A_in_out = 53.66      
    A_in_wat = 8.38       
    A_wat_out = 8.38       

    # U-values (W/m2.K)
    U_in_out = 0.15      
    U_in_wat = 1.1      
    U_wat_out = 1.1     

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
        
        # Temperature changes
        dT_in = (Q_in_out + Q_wat_in) / C_in * dt
        dT_wat = (P_wat_in - Q_wat_in + Q_wat_out) / C_wat * dt
        
        T_in[i+1] = T_in[i] + dT_in
        T_wat[i+1] = T_wat[i] + dT_wat

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
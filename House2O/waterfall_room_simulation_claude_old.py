"""
Room Temperature Evolution Simulation — Laminar Waterfall Cooling
=================================================================
Waterfall parameters (measured / given):
  - Width    : 2.0 m
  - Head (h) : 3.7 mm
  - Re       : ~400  (laminar, confirmed from Q)
  - Flow rate: 0.8 l/s

Physics model
─────────────
The room has an effective thermal capacitance C [J/K] that includes
both the air mass and the thermal mass of walls, floor, and ceiling.

Without the waterfall, the room follows the measured hourly temperature
profile T_ref(t), which implicitly encodes all other gains/losses
(sun, people, HVAC, ventilation, ...).

The waterfall adds an EXTRA cooling term:

    C · dT/dt  =  C · (dT_ref/dt)  −  Q_waterfall(T)

Starting from T(0) = T_ref(0), this integrates forward in time.
The result is the actual room temperature with the waterfall running.

Two heat-transfer mechanisms are modelled:
  1. Evaporative (latent)  — dominant; driven by vapour pressure difference
  2. Convective (sensible) — room air → water film

How to use
──────────
1. Set ROOM_TEMP_DATA to your group's measured hourly values.
2. Adjust FALL_HEIGHT, ROOM_VOLUME, and WALL_PARAMS to match your space.
3. Run:  python3 waterfall_room_simulation.py
4. Results are printed and saved to room_simulation_results.csv
"""

import math
import csv
import os

# ═══════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  — edit these to match your setup
# ═══════════════════════════════════════════════════════════════

# ── Waterfall geometry & flow ───────────────────────────────────────────────
WIDTH       = 2.0        # m      — curtain width
HEAD        = 3.7e-3     # m      — water head at the weir (3.7 mm)
Q_FLOW      = 0.8e-3     # m³/s   — volumetric flow rate  (0.8 l/s)
FALL_HEIGHT = 2.44        # m      — vertical drop of the water curtain
                         #          adjust to your actual installation

# ── Water temperature ───────────────────────────────────────────────────────
T_WATER = 18.0           # °C     — supply water temperature
                         #          (tap water; measure if possible)

# ── Ambient air conditions ──────────────────────────────────────────────────
RH_ROOM = 0.50           # —      — room relative humidity  (0 – 1)
                         #          update if your group has measured this

# ── Room geometry ───────────────────────────────────────────────────────────
ROOM_VOLUME  = 50.0      # m³     — room volume  (e.g. 5 m × 5 m × 2 m)
ROOM_SURFACE = 90.0      # m²     — total surface area of walls/floor/ceiling
                         #          for a 5×5×2 m box: 2*(25+10+10) = 90 m²

# ── Wall / structural thermal mass ─────────────────────────────────────────
WALL_DENSITY   = 2000.0  # kg/m³  — concrete / brick
WALL_CP        = 880.0   # J/(kg·K)
WALL_DEPTH     = 0.02    # m      — effective thermal penetration depth
                         #          (~2 cm for diurnal cycle, Fourier estimate)

# ── Fluid properties (water at ~20 °C) ─────────────────────────────────────
RHO_W = 998.0            # kg/m³
MU_W  = 1.002e-3         # Pa·s
L_VAP = 2.45e6           # J/kg   — latent heat of vaporisation

# ── Air properties (~20 °C) ────────────────────────────────────────────────
RHO_AIR = 1.204          # kg/m³
CP_AIR  = 1005.0         # J/(kg·K)

# ── Heat / mass transfer coefficients ──────────────────────────────────────
H_CONV = 5.0             # W/(m²·K)  — convective film coefficient
                         #             typical for natural convection over water
H_MASS = 0.010           # m/s       — mass-transfer coefficient (evaporation)
                         #             derived from Lewis relation

# ── Simulation ──────────────────────────────────────────────────────────────
DT_SIM     = 60.0        # s      — Euler time-step (1 min)
OUTPUT_CSV = "room_simulation_results.csv"

# ═══════════════════════════════════════════════════════════════
# 1.  HOURLY ROOM TEMPERATURE DATA  — replace with your measurements
#     Format: list of (hour, T_room [°C])
#     Must span at least 0 … N hours, monotonically increasing.
# ═══════════════════════════════════════════════════════════════
ROOM_TEMP_DATA = [
    ( 0, 22.0),
    ( 1, 21.5),
    ( 2, 21.0),
    ( 3, 20.8),
    ( 4, 20.5),
    ( 5, 20.3),
    ( 6, 20.5),
    ( 7, 21.0),
    ( 8, 22.0),
    ( 9, 23.2),
    (10, 24.1),
    (11, 24.8),
    (12, 25.3),
    (13, 25.7),
    (14, 26.0),
    (15, 25.8),
    (16, 25.4),
    (17, 24.9),
    (18, 24.3),
    (19, 23.7),
    (20, 23.2),
    (21, 22.8),
    (22, 22.4),
    (23, 22.1),
    (24, 22.0),
]

# ═══════════════════════════════════════════════════════════════
# 2.  WATERFALL PHYSICS
# ═══════════════════════════════════════════════════════════════

def sat_pressure(T_C):
    """Antoine equation → vapour saturation pressure [Pa] at T [°C]."""
    return 611.2 * math.exp(17.67 * T_C / (T_C + 243.5))


def waterfall_cooling_power(T_room_C):
    """
    Cooling power [W] extracted from the room by the waterfall.

    Evaporative term  — dominant; vapour pressure gradient drives evaporation.
    Convective term   — sensible heat from warm room air to cold water film.

    Returns (Q_total, Q_evap, Q_conv) in Watts.
    """
    A = WIDTH * FALL_HEIGHT          # m²  wetted curtain area

    # ── Evaporative (latent) ────────────────────────────────────────────────
    P_sat_surf = sat_pressure(T_WATER)
    P_sat_air  = sat_pressure(T_room_C)
    P_vap_air  = RH_ROOM * P_sat_air

    M_w = 0.018015                   # kg/mol  (water)
    R   = 8.314                      # J/(mol·K)
    rho_v_surf = P_sat_surf * M_w / (R * (T_WATER   + 273.15))
    rho_v_air  = P_vap_air  * M_w / (R * (T_room_C  + 273.15))

    delta_rho_v = max(rho_v_surf - rho_v_air, 0.0)   # evaporation only
    m_evap  = H_MASS * A * delta_rho_v                # kg/s
    Q_evap  = m_evap * L_VAP                          # W

    # ── Convective (sensible) ───────────────────────────────────────────────
    dT     = T_room_C - T_WATER
    Q_conv = H_CONV * A * max(dT, 0.0)                # W

    return Q_evap + Q_conv, Q_evap, Q_conv


def film_properties():
    """Derived laminar-film characteristics for the report header."""
    nu  = MU_W / RHO_W
    q   = Q_FLOW / WIDTH             # m²/s  unit-width flow rate
    Re  = q / nu                     # = Q / (W · ν)
    g   = 9.81
    h_f = (3 * nu * q / g) ** (1/3)  # Nusselt falling-film thickness [m]
    u   = q / h_f                    # mean velocity [m/s]
    return Re, h_f, u


# ═══════════════════════════════════════════════════════════════
# 3.  HELPERS
# ═══════════════════════════════════════════════════════════════

def interp(t_sec, data):
    """Linear interpolation of T_ref [°C] at time t [s]."""
    t_hr = t_sec / 3600.0
    for i in range(len(data) - 1):
        h0, T0 = data[i]
        h1, T1 = data[i + 1]
        if h0 <= t_hr <= h1:
            return T0 + (T1 - T0) * (t_hr - h0) / (h1 - h0)
    return data[-1][1]


def effective_thermal_capacitance():
    """
    C_eff = C_air + C_walls  [J/K]

    The wall contribution uses the diurnal thermal penetration depth
    (typically ~1–5 cm for concrete/brick).  Only the surface layer
    participates in the ~24-hour temperature swing.
    """
    C_air   = RHO_AIR * ROOM_VOLUME * CP_AIR
    m_wall  = WALL_DENSITY * WALL_DEPTH * ROOM_SURFACE
    C_walls = m_wall * WALL_CP
    return C_air + C_walls, C_air, C_walls


# ═══════════════════════════════════════════════════════════════
# 4.  SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════

def simulate():
    """
    Euler integration of the room energy balance.

    ODE:  C_eff · dT/dt  =  C_eff · (dT_ref/dt)  −  Q_waterfall(T)

    Interpretation
    ──────────────
    • If Q_waterfall = 0, T tracks T_ref exactly (no waterfall effect).
    • Q_waterfall > 0 pulls T below T_ref.
    • The gap  T_ref − T  is the cooling benefit of the waterfall.
    """
    C_eff, C_air, C_walls = effective_thermal_capacitance()
    t_end = ROOM_TEMP_DATA[-1][0] * 3600.0
    steps = int(t_end / DT_SIM)

    T = interp(0, ROOM_TEMP_DATA)       # start at the reference temperature
    results = []

    for step in range(steps):
        t = step * DT_SIM

        T_ref_now  = interp(t,          ROOM_TEMP_DATA)
        T_ref_next = interp(t + DT_SIM, ROOM_TEMP_DATA)
        dT_ref_dt  = (T_ref_next - T_ref_now) / DT_SIM    # K/s

        Q_tot, Q_ev, Q_co = waterfall_cooling_power(T)

        dT_dt = dT_ref_dt - Q_tot / C_eff

        results.append({
            "time_hr"       : round(t / 3600.0, 4),
            "T_simulated_C" : round(T,           3),
            "T_reference_C" : round(T_ref_now,   3),
            "dT_cooling_K"  : round(T_ref_now - T, 3),
            "Q_total_W"     : round(Q_tot,       2),
            "Q_evap_W"      : round(Q_ev,        2),
            "Q_conv_W"      : round(Q_co,        2),
        })

        T += dT_dt * DT_SIM
        # Safety clamp (should not be needed with reasonable inputs)
        T = max(min(T, 60.0), -5.0)

    return results, C_eff, C_air, C_walls


# ═══════════════════════════════════════════════════════════════
# 5.  OUTPUT
# ═══════════════════════════════════════════════════════════════

def print_header(C_eff, C_air, C_walls):
    Re, h_f, u = film_properties()
    lam = "✓  laminar" if Re < 500 else "✗  NOT laminar"
    print("=" * 63)
    print("    LAMINAR WATERFALL — ROOM TEMPERATURE SIMULATION")
    print("=" * 63)
    print(f"  Waterfall")
    print(f"    Width            : {WIDTH*100:.0f} cm")
    print(f"    Head at weir     : {HEAD*1000:.1f} mm")
    print(f"    Flow rate        : {Q_FLOW*1000:.1f} l/s")
    print(f"    Film Re (q/ν)    : {Re:.0f}  —  {lam}")
    print(f"    Film thickness   : {h_f*1000:.3f} mm  (Nusselt theory)")
    print(f"    Mean velocity    : {u:.3f} m/s")
    print(f"    Fall height      : {FALL_HEIGHT:.1f} m  →  wetted area {WIDTH*FALL_HEIGHT:.1f} m²")
    print(f"    Water temp       : {T_WATER:.1f} °C")
    print(f"  Room")
    print(f"    Volume           : {ROOM_VOLUME:.0f} m³")
    print(f"    Surface          : {ROOM_SURFACE:.0f} m²")
    print(f"    Humidity         : {RH_ROOM*100:.0f} %")
    print(f"    C_air            : {C_air/1e6:.3f} MJ/K")
    print(f"    C_walls          : {C_walls/1e6:.3f} MJ/K  (depth = {WALL_DEPTH*100:.0f} mm)")
    print(f"    C_effective      : {C_eff/1e6:.3f} MJ/K")
    print("-" * 63)
    for T_ex in [20.0, 23.0, 26.0]:
        Q_tot, Q_ev, Q_co = waterfall_cooling_power(T_ex)
        print(f"  Cooling @ {T_ex:.0f} °C : evap {Q_ev:.0f} W  +  conv {Q_co:.0f} W  =  {Q_tot:.0f} W total")
    print("=" * 63)


def print_table(results):
    print(f"\n  {'Hour':>4}  {'T_sim °C':>9}  {'T_ref °C':>9}  "
          f"{'ΔT K':>7}  {'Q_tot W':>9}  {'Q_evap W':>9}")
    print("  " + "─" * 56)
    seen = set()
    for r in results:
        hr = int(r["time_hr"])
        if hr not in seen and (r["time_hr"] - hr) < (DT_SIM / 3600):
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


def save_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved → {path}")


# ═══════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results, C_eff, C_air, C_walls = simulate()
    print_header(C_eff, C_air, C_walls)
    print("\n  Running simulation …")
    print_table(results)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_CSV)
    save_csv(results, out)
    print("\n  Done.")

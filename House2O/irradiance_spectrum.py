"""
House2O — Solar Irradiance Spectrum Pipeline
=============================================
Computes the solar spectral irradiance (W/m²/nm) at a given location and time
using the SPECTRL2 model (Bird & Riordan, 1984) via pvlib.

Outputs the spectrum broken down into:
  - Direct normal irradiance (DNI)
  - Diffuse horizontal irradiance (DHI)
  - Plane-of-array components (direct, sky diffuse, ground diffuse, global)

Note: SPECTRL2 is a clear-sky model. Cloud cover is not accounted for.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pvlib
import pySMARTS


# ── Default atmospheric parameters ────────────────────────────────────────────
# These are typical mid-latitude values. For a more accurate simulation of a
# specific location and season, use measured data from CAMS or similar sources.

DEFAULTS = {
    "precipitable_water":      1.42,   # cm    — atmospheric water vapour
    "ozone":                   0.344,  # atm-cm — total column ozone
    "aerosol_turbidity_500nm": 0.1,    # —      — aerosol optical depth at 500 nm
    "ground_albedo":           0.25,   # —      — ground reflectance (grass ~0.25)
    "surface_pressure":        101325, # Pa     — standard sea-level pressure
}


# ── Core spectrum calculation ──────────────────────────────────────────────────

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
):
    """
    Compute the solar irradiance spectrum at a given location and time.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees. North positive.
    longitude : float
        Longitude in decimal degrees. East positive.
    datetime_input : str or datetime-like
        Date and time, e.g. '2024-06-21 12:00'. Interpreted in `tz`.
    tz : str
        Timezone string, e.g. 'Europe/Brussels', 'UTC'. Default 'UTC'.
    surface_tilt : float
        Tilt of the water surface from horizontal, in degrees.
        0° = horizontal, 90° = vertical (wall/window). Default 90°.
    surface_azimuth : float
        Compass direction the surface faces, in degrees from north.
        180° = south-facing (optimal in northern hemisphere). Default 180°.
    precipitable_water : float
        Total column water vapour in cm. Strongly affects NIR absorption,
        which is especially relevant for water windows. Default 1.42 cm.
    ozone : float
        Total column ozone in atm-cm. Affects UV. Default 0.344 atm-cm.
    aerosol_turbidity_500nm : float
        Aerosol optical depth at 500 nm. Higher = more scattering/absorption
        (e.g. urban/dusty: ~0.2–0.4, clean rural: ~0.05–0.1). Default 0.1.
    ground_albedo : float
        Ground reflectance [0–1]. Affects diffuse ground component.
        Grass ~0.25, snow ~0.8, concrete ~0.3. Default 0.25.
    surface_pressure : float
        Surface air pressure in Pa. Decrease for high-altitude sites.
        Default 101325 Pa (sea level).

    Returns
    -------
    spectrum : pd.DataFrame
        122-row DataFrame indexed by wavelength (nm) with columns:
          wavelength_nm   — nm
          dni_extra       — W/m²/nm  extraterrestrial (top-of-atmosphere) DNI
          dhi             — W/m²/nm  diffuse horizontal irradiance
          dni             — W/m²/nm  direct normal irradiance
          poa_direct      — W/m²/nm  direct component on your surface
          poa_sky_diffuse — W/m²/nm  sky diffuse on your surface
          poa_ground_diff — W/m²/nm  ground-reflected diffuse on your surface
          poa_global      — W/m²/nm  total irradiance on your surface
    solar : dict
        Solar position and derived quantities:
          apparent_zenith — degrees
          azimuth         — degrees (from north)
          airmass         — relative air mass (Kasten-Young)
          aoi             — angle of incidence on the surface (degrees)
          sun_above_horizon — bool
          datetime        — the input time (tz-aware)
    """

    # --- Solar position ---
    location = pvlib.location.Location(
        latitude=latitude, longitude=longitude, tz=tz
    )
    times = pd.DatetimeIndex([datetime_input], tz=tz)
    solpos = location.get_solarposition(times)

    apparent_zenith = float(solpos["apparent_zenith"].iloc[0])
    azimuth = float(solpos["azimuth"].iloc[0])
    sun_above_horizon = apparent_zenith < 90.0

    # --- Angle of incidence on the defined surface ---
    aoi = float(
        pvlib.irradiance.aoi(surface_tilt, surface_azimuth, apparent_zenith, azimuth)
    )

    # --- Air mass ---
    if sun_above_horizon:
        relative_airmass = float(
            pvlib.atmosphere.get_relative_airmass(
                apparent_zenith, model="kastenyoung1989"
            )
        )
    else:
        relative_airmass = np.nan

    solar = {
        "apparent_zenith": apparent_zenith,
        "azimuth": azimuth,
        "airmass": relative_airmass,
        "aoi": aoi,
        "sun_above_horizon": sun_above_horizon,
        "datetime": times[0],
    }

    # --- Return zero spectrum if sun is below the horizon ---
    if not sun_above_horizon:
        wavelengths = pvlib.spectrum.spectrl2(
            apparent_zenith=0,       # dummy — just to get wavelength axis
            aoi=0, surface_tilt=0, ground_albedo=ground_albedo,
            surface_pressure=surface_pressure, relative_airmass=1,
            precipitable_water=precipitable_water, ozone=ozone,
            aerosol_turbidity_500nm=aerosol_turbidity_500nm, dayofyear=1,
        )["wavelength"]
        zeros = np.zeros_like(wavelengths)
        return pd.DataFrame({
            "wavelength_nm":   wavelengths,
            "dni_extra":       zeros,
            "dhi":             zeros,
            "dni":             zeros,
            "poa_direct":      zeros,
            "poa_sky_diffuse": zeros,
            "poa_ground_diff": zeros,
            "poa_global":      zeros,
        }), solar

    # --- SPECTRL2 ---
    raw = pvlib.spectrum.spectrl2(
        apparent_zenith=apparent_zenith,
        aoi=aoi,
        surface_tilt=surface_tilt,
        ground_albedo=ground_albedo,
        surface_pressure=surface_pressure,
        relative_airmass=relative_airmass,
        precipitable_water=precipitable_water,
        ozone=ozone,
        aerosol_turbidity_500nm=aerosol_turbidity_500nm,
        dayofyear=int(times[0].dayofyear),
    )

    # Squeeze out the length-1 time dimension → 1-D arrays
    spectrum = pd.DataFrame({
        "wavelength_nm":   raw["wavelength"],
        "dni_extra":       np.squeeze(raw["dni_extra"]),
        "dhi":             np.squeeze(raw["dhi"]),
        "dni":             np.squeeze(raw["dni"]),
        "poa_direct":      np.squeeze(raw["poa_direct"]),
        "poa_sky_diffuse": np.squeeze(raw["poa_sky_diffuse"]),
        "poa_ground_diff": np.squeeze(raw["poa_ground_diffuse"]),
        "poa_global":      np.squeeze(raw["poa_global"]),
    })

    return spectrum, solar


# ── Daily time series ──────────────────────────────────────────────────────────

def compute_daily_spectra(
    latitude,
    longitude,
    date,
    tz="UTC",
    time_resolution_min=15,
    **kwargs,
):
    """
    Compute the integrated irradiance (W/m²) over the full day at a given
    location, sampled at `time_resolution_min` intervals.

    Returns a DataFrame indexed by time with columns:
      poa_global, poa_direct, poa_sky_diffuse, poa_ground_diff, airmass, aoi
    """
    times = pd.date_range(
        start=f"{date} 00:00",
        end=f"{date} 23:59",
        freq=f"{time_resolution_min}min",
        tz=tz,
    )

    records = []
    for t in times:
        _, solar = compute_spectrum(
            latitude, longitude, t, tz=tz, **kwargs
        )
        # Re-run to get the integrated power (W/m²) by integrating spectrum over λ
        if solar["sun_above_horizon"]:
            spec, _ = compute_spectrum(
                latitude, longitude, t, tz=tz, **kwargs
            )
            # Integrate using the trapezoid rule: ∫ I(λ) dλ  [W/m²]
            wl = spec["wavelength_nm"].values
            records.append({
                "time":             t,
                "poa_global":       np.trapz(spec["poa_global"],       wl),
                "poa_direct":       np.trapz(spec["poa_direct"],       wl),
                "poa_sky_diffuse":  np.trapz(spec["poa_sky_diffuse"],  wl),
                "poa_ground_diff":  np.trapz(spec["poa_ground_diff"],  wl),
                "airmass":          solar["airmass"],
                "aoi":              solar["aoi"],
            })
        else:
            records.append({
                "time":             t,
                "poa_global":       0.0,
                "poa_direct":       0.0,
                "poa_sky_diffuse":  0.0,
                "poa_ground_diff":  0.0,
                "airmass":          np.nan,
                "aoi":              np.nan,
            })

    return pd.DataFrame(records).set_index("time")


# ── Plotting ───────────────────────────────────────────────────────────────────

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
    ax.plot(wl, spectrum["poa_ground_diff"], color="sienna",    lw=1.2, label="POA ground diffuse", ls="--")
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
    return fig


def plot_daily_power(daily_df, latitude, longitude, date):
    """
    Plot the integrated irradiance (W/m²) over the course of a day.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(daily_df.index, daily_df["poa_global"], alpha=0.2, color="gold")
    ax.plot(daily_df.index, daily_df["poa_global"],     color="gold",      lw=2,   label="POA global")
    ax.plot(daily_df.index, daily_df["poa_direct"],     color="orange",    lw=1.5, label="POA direct")
    ax.plot(daily_df.index, daily_df["poa_sky_diffuse"],color="steelblue", lw=1.5, label="POA sky diffuse")

    ax.set_xlabel("Time of day", fontsize=12)
    ax.set_ylabel("Irradiance (W/m²)", fontsize=12)
    ax.set_title(
        f"Daily irradiance on surface  —  {date}  |  lat={latitude}°, lon={longitude}°",
        fontsize=12
    )
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Single-moment spectrum ─────────────────────────────────────────────────
    # Antwerp, Belgium — summer solstice, solar noon
    LAT, LON = 51.22, 4.40
    TZ = "Europe/Brussels"
    DATETIME = "2024-06-21 13:00"    # local time (CEST = UTC+2, solar noon ≈ 13:30)

    print("Computing spectrum for Antwerp, 21 June 2024 13:00 CEST …")
    spectrum, solar = compute_spectrum(
        latitude=LAT,
        longitude=LON,
        datetime_input=DATETIME,
        tz=TZ,
        surface_tilt=90,       # vertical window
        surface_azimuth=180,   # south-facing
        # -- Atmospheric parameters (Belgian summer, typical) --
        precipitable_water=2.0,         # cm  — Belgium is fairly humid
        ozone=0.340,                    # atm-cm
        aerosol_turbidity_500nm=0.12,   # slightly urban
        ground_albedo=0.20,
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

    # ── Daily power curve ──────────────────────────────────────────────────────
    print("\nComputing daily irradiance curve …")
    daily = compute_daily_spectra(
        latitude=LAT,
        longitude=LON,
        date="2024-06-21",
        tz=TZ,
        surface_tilt=90,
        surface_azimuth=180,
        precipitable_water=2.0,
        ozone=0.340,
        aerosol_turbidity_500nm=0.12,
        ground_albedo=0.20,
        time_resolution_min=15,
    )

    daily_energy = np.trapz(daily["poa_global"], (daily.index - daily.index[0]).total_seconds() / 3600)
    print(f"  Daily energy on south window : {daily_energy:.1f} Wh/m²")

    fig2 = plot_daily_power(daily, LAT, LON, "2024-06-21")
    plt.show()

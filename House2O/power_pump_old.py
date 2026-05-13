# Code House2O project: energy consumption of the pumps.
# Authors: Mona Soors, Matti Cornille, Vincent Audenaert, Daan Grupping, Anis Dhewaju and Xander Scheyltjens
# Last updated: 5/05/2026



# ----- Calculating the power needed for the pump for underfloor heating ------------------------------------------------------

def heat_loss(T_in, T_out, U, A):
    """
    Compute the total transmission heat loss: Q = deltaT sum(U_i * A_i)

    Parameters:
    T_in  (float)            : Indoor temperature in °C
    T_out (float)            : Outdoor temperature in °C
    U     (float or list)     : U-value(s) in W/m²K, single value or list
    A     (float orlist)     : Area(s)   in m², single value or list

    Returns:
    float: Total heat loss in Watts
    """
    # Normalise scalars to lists
    U_list = U if isinstance(U, list) else [U]
    A_list = A if isinstance(A, list) else [A]

    if len(U_list) != len(A_list):
        raise ValueError(
            f"U and A must have the same length (got {len(U_list)} and {len(A_list)})."
        )

    ua_sum = sum(u * a for u, a in zip(U_list, A_list))
    return ua_sum * (T_in - T_out)

def flow_rate_heating(Q, deltaT=5, c=4186, rho=997):
    """
    Compute the flow rate of the pump needed to compensate for the heat loss.

    Parameters:
    Q (float)               : Heat loss in Watts
    deltaT    (float)       : Temperature difference of water entering and leaving the circulation in K (default: 5)
    c         (float)       : Specific heat capacity of water in J/kgK (default: 4186)
    rho       (float)       : Density of water in kg/m^3 (default: 997)

    Returns:
    float: Flow rate in m^3/s
    """
    return Q / (c * deltaT * rho)

def pump_power_heating(flow_rate, eta=0.7, R=100, length=4,width=4,pipe_spacing=0.15, ZF=2.2):
    """
    Compute the power consumption of the pump.

    Parameters:
    flow_rate (float)       : Flow rate in m^3/s
    eta       (float)       : Efficiency of the pump (default: 0.7)
    R         (float)       : pipe friction loss (default: 100 Pa/m)
    length, width (float)   : Dimensions of the house in meters (default: 4, 4)
    pipe_spacing (float)    : Spacing between pipes in meters (default: 0.15)
    ZF        (float)       : Zeta factor (default: 2.2)

    Returns:
    float: Power consumption in Watts
    """

    # Length of the longest loop approximted as:
    L = 2 * (length * width) / pipe_spacing

    # Compute the power consumption (simplified model)
    power = flow_rate * R * L * ZF / eta  
    return power


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


#--------Test-environment----------------------------------------------------------------
if __name__=="__main__":
    # heat_loss_value = heat_loss(T_in=20, T_out=0, U=[0.2, 1.1], A=[4*4+3*4*2.44+1.56*4+4*2.54, 4*2.44+4*2.54])
    # print("Heat loss =", heat_loss_value, "W")
    # print("Flow rate underfloor heating =", flow_rate_heating(heat_loss_value), "m^3/s")
    # print("Pump power underfloor heating =", pump_power_heating(flow_rate_heating(heat_loss_value)), "W")
    h_weir_value = h_weir()
    print("Head over the weir with Re=400:", h_weir_value, "m")

    print("Flow rate waterfall =", flow_rate_waterfall(h_weir=h_weir_value), "m^3/s")
    print("Pump power waterfall =", pump_power_waterfall(flow_rate_waterfall(h_weir=h_weir_value)), "W")
# mô phỏng được các kênh truyền từ LEO -> HAP -> UAV và UAV->UE 
'''
Following paper Deep Reinforcement Learning for UAV Placement
over Mixed FSO/RF-Based Non-terrestrial Networks

Using torch library to calculate CFSO. So, it can calculate CFSO for a tensor with more dimensions

'''
# from scipy.integrate import quad
import math
import torch 
from Parameters import *
import torch
#PARAM 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


pi = torch.tensor(torch.pi,device=device)
e = torch.tensor(torch.e,device=device)

PARAMS = {
    'lambda'             : lambda_fso_wavelength,         # FSO Wavelength (1.55 micrometer)
    'velocity_wind'     : velocity_wind,              # 21m/s (? It has to be varying because it is not constant )
    'Nc'                 : Nc ,           # cm-3
    'Hcl'                : Hcl,           # km =1e3 m 
    'B_FSO'              : B_FSO ,             # GHz 
    #Source Node = Satellite 
    'LEO_Altitude'      : leo_z,           # 500km = 500 e3 m
    'LEO_divergence_angle' : leo_divergence_angle,         # mu rad = 1e-6 rad
    'LEO_Transmitted_power': leo_transmitted_power ,         # dBm
    'LEO_theta_Sxy'       :leo_theta_sxy,
    
    'Stratospheric_Altitude'   : 28e3,            # Max Altitude of Stratospheric Layer = 50km = 50e3 m 
    'Troposhere_Altitude'      : 20e3,            # Max Altitude of Troposhere Layer = 20 km = 20e3 m 
    # Relay Node = HAP
    'HAP_Altitude'          : hap_z,          # Altitude of HAP 20 km = 20e3 m
    'HAP_divergence_angle'  : hap_divergence_angle,         # Divergence angle 1 mrad = 1e-3 rad
    'Radius_Curvature'      : 500.0,             # Radius of Curvature (bán kính cong)
    'LEO_2_HAP_Noise_standard_deviation_' : leo_hap_noise, # A/Hz 
    'HAP_Detector_responsivity' : hap_detector_responsivity , 
    'HAP_Aperture_Radius'       : hap_aperture_radius ,     # cm
    'HAP_Transmitted_power' : hap_tranmistted_power , #    dBm
    'HAP_muy_x'     : hap_muy_x,
    'HAP_muy_y'     : hap_muy_y,
    'HAP_sigma_x'   : hap_sigma_x,
    'HAP_sigma_y'   : hap_sigma_y,
    'HAP_theta_Rxy'   : hap_theta_Rxy, 

    #Destination Node = UAV 
    'UAV_Altitude'      : uav_z,               # m
    'UAV_Detector_responsivity' : 0.9,      
    'UAV_Aperture_Radius' : uav_Aperture_Radius ,             #cm
    'HAP_2_UAV_Noise_standard_deviation_' :hap_uav_noise, #A/Hz 
    'UAV_muy_x' : uav_muy_x,
    'UAV_muy_y' : uav_muy_y,
    'UAV_sigma_x'   : uav_sigma_x,
    'UAV_sigma_y'   : uav_sigma_y,
    'C2n_0'             : C2n_0,            # Ground-level turbulence varying from 10−17 m−2/3 to 10−13 m−2/3


}
#==========================================================================#==========================================================================
#==========================================================================#==========================================================================
#==========================================================================#==========================================================================




#                                                                Sharing functions that can be used by both channel

def compute_sec_xi(zenith_angle):
    # zenith_angle is computed in radian    
    cos_xi = torch.cos(zenith_angle)
    sec_xi = 1 / cos_xi

    return sec_xi

def beer_lambert(attenuation_coefficiency, H, zenith_angle):
    # attenuation_coefficiency
    # H is the vertical extent of corresponding layer
    # zenith_angle of satellite or UAV
    #print(f"attenuation_coef\n{attenuation_coefficiency.device}")
    sec_xi = compute_sec_xi(zenith_angle= zenith_angle)
    #print(f"sec_xi\n{sec_xi}")
    return torch.exp(-attenuation_coefficiency * H * sec_xi)
#==========================================================================#==========================================================================
#==========================================================================#==========================================================================
#==========================================================================#==========================================================================

#                                                                      LEO - HAP Channel Model
def stratospheric_attenuation(volcanic_level, zenith_angle):
    # 1 -> 4: Background, Moderate, High, Extreme volcanic

    alpha = 0
    if volcanic_level == 1:
        alpha= 1e-4
    elif volcanic_level == 2:
        alpha= 8e-4
    elif volcanic_level == 3:
        alpha= 2e-2
    elif volcanic_level == 4 :
        alpha= 1e-1 


    h_a_SR = beer_lambert(alpha, PARAMS['Stratospheric_Altitude'], compute_sec_xi(zenith_angle))
    return h_a_SR

def PDF_lambda_SR(
        mu_x, mu_y, sigma_x, sigma_y, 
        a_radius, 
        x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
        lambda_SR,

        link,
        volcanic_level=3,
        
):
    phi_Rmod, A_Rmod = compute_phi_mod_and_A_mod(
        mu_x, mu_y, sigma_x, sigma_y, 
        a_radius, 
        x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
        link)

    HAP_zenith_angle = compute_zenith_angle_torch(x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, x_lower_vehicle, y_lower_vehicle, z_lower_vehicle)

    psi_SR = (PARAMS['HAP_Detector_respontivity'] * PARAMS['LEO_Transmitted_power'] / PARAMS['LEO_2_HAP_Noise_standard_deviation_']) **2
    ha_SR = stratospheric_attenuation(volcanic_level, zenith_angle=HAP_zenith_angle)

    return (
        (phi_Rmod**2/ (2*lambda_SR*(A_Rmod*ha_SR)**(phi_Rmod**2))) * (lambda_SR/psi_SR)**(phi_Rmod**2/2) 
    )


#==========================================================================#==========================================================================
#==========================================================================#==========================================================================
#==========================================================================#==========================================================================
#                                                                      HAP - UAV Channel Model
#                                                                Problem 1: Cloud Attenuation
def cloud_attenuation(zenith_angle, Mc):
    # Mc (mg/m3) is Cloud Liquid Water Content 
    # Nc (cm-3) is the number cloud droplet concentration 
    """
    Compute cloud attenuation 
    """
    Mc_g_m3 = Mc * 1e-3 
    visibility = 1.002 /((Mc_g_m3 * PARAMS['Nc']) ** 0.6473)

    epsilon = 0  # size distribution of the scattering particles 
    # if visibility > 50:
    #     epsilon = 1.6
    # elif visibility > 6:
    #     epsilon = 1.3
    # elif visibility > 1:
    #     epsilon = 0.16 * visibility + 0.34
    # elif visibility >0.5:
    #     epsilon = visibility - 0.5 

    epsilon = torch.where(visibility > 50, 1.6,
                torch.where(visibility>6, 1.3,
                torch.where(visibility>1, 0.16 * visibility + 0.34,
                torch.where(visibility> 0.5, visibility-0.5, 
                torch.zeros_like(visibility)))))

    beta_db = (3.91 / visibility) * (PARAMS["lambda"]/ 550e-9) ** (-epsilon)
    beta    = beta_db / (1e4 * math.log10(e))
    h_a_RD =  beer_lambert(beta, PARAMS['Hcl'], zenith_angle) 
    return h_a_RD




#                                                                Problem 2: Atmospheric Turbulence 
#                                                               STEP 1: C2N: The variation of Refractive Index Structure Parameter 
def C2n(velocity_wind, h, C2n_i ):

    return ( 
        0.00594 * (velocity_wind/27)**2 *(1e-5*h)**10 * torch.exp(-h/ 1000) 
        + 2.7e-16 * torch.exp(-h/ 1500)
        + C2n_i* torch.exp(-h/ 100)
    )

C2n_lambda = lambda h : (
        0.00594 * (PARAMS['velocity_wind']/27)**2 *(1e-5*h)**10 * torch.exp(-h/ 1000) 
        + 2.7e-16 * torch.exp(-h/ 1500)
        + PARAMS['C2n_0']* torch.exp(-h/ 100)

)

def Rytov_integral(h):
     return (
         C2n_lambda(h) *  (h-PARAMS['UAV_Altitude'])** (5/6)
     )

#                                                               STEP 2: Rytov Variance  

def Rytov_variance(zenith_angle, integral_C2n):
    k = 2*pi / torch.tensor(PARAMS['lambda'],device=device)

    term1 = 2.25* k**(7/6) 
    sec_xi = compute_sec_xi(zenith_angle)
    term2 = sec_xi ** (11/6)

    #print(f"term1 {term1.device}")
    #print(f"term2 {term2.device}")
    #print(f"integral {integral_C2n.device}")

    return (    
        term1
        * term2
        * integral_C2n
    )


def atmospheric_turbulence(h_a, rytov_variance ):
    return (
        1 / (torch.sqrt(2*pi) * torch.sqrt(rytov_variance) * h_a )
        *  torch.exp(-( (torch.log(h_a) + rytov_variance/2 )** 2)/ (2*rytov_variance) )
    )



# STEP 3: Aperture Averaging Effect
def integrand_C2n_denominator(h):
        return C2n(PARAMS['velocity_wind'],h,PARAMS['C2n_0']) * (h-PARAMS['UAV_Altitude'] ) ** 2 


def compute_theta(integral_C2n_numerator, integral_C2n_denominator ):
    return (
        (integral_C2n_numerator/integral_C2n_denominator)** (6/7)
     )


def Aperture_averaging_effect(aperture_radius, lambda_, theta, zenith_angle_R):
    return (
        1 
        +(1.11* ( (4*aperture_radius) / (lambda_ * theta * compute_sec_xi(zenith_angle_R)) ) **(7/6) ) ** (-1)
    )


#                                                                Problem 3: Generalized Pointing Error Model / Hovering-Induced Misalignment
# Jitter Variance 
def compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y )  :  
    return (
        ( (3*(mu_x **2) *(sigma_x **4) 
           + 3 *(mu_y **2) * (sigma_y **4) 
           + sigma_x**6 + sigma_y**6 ) 
           / 2 )**(1/3)
    )

def compute_zenith_angle_torch(x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, x_lower_vehicle, y_lower_vehicle, z_lower_vehicle):
    """
    Compute zenith angle (Test OK)
    """

    distance = torch.sqrt((x_higher_vehicle-x_lower_vehicle)**2 + (y_higher_vehicle-y_lower_vehicle)**2 + (z_higher_vehicle-z_lower_vehicle)**2 + 1e-6) 
    cos_zenith =  (z_higher_vehicle - z_lower_vehicle) / distance 


    zenith_angle = torch.acos(cos_zenith) 
    return zenith_angle  #Radian

def mu2_integral_for_D(h):
     return (
         C2n_lambda(h) * ( (h-PARAMS['UAV_Altitude'])/ (PARAMS['HAP_Altitude']-PARAMS['UAV_Altitude']) ) ** (5/3)
     )

def mu2_integral_for_R(h):
     return (
         C2n_lambda(h) * ( (h-PARAMS['HAP_Altitude'])/ (PARAMS['LEO_Altitude']-PARAMS['HAP_Altitude']) ) ** (5/3)
     )

h_uav_hap = torch.linspace(uav_z, hap_z, steps = 512, device=device)
C2n_vals = C2n_lambda(h_uav_hap)
integrand = C2n_vals *((h_uav_hap-uav_z)/(hap_z-uav_z)) ** (5/3)
mu2_uav_hap = torch.trapz(integrand, h_uav_hap)

h_hap_leo = torch.linspace(hap_z, leo_z, steps = 512, device=device)
C2n_val_hap_leo = C2n_lambda(h_hap_leo)
integrand = C2n_val_hap_leo *((h_hap_leo-hap_z)/(leo_z-hap_z)) ** (5/3)
mu2_hap_leo = torch.trapz(integrand, h_hap_leo)

def compute_Gd(L,W,zenith_angle,link):
    k = 2 * pi / PARAMS['lambda']
    sec_xi = compute_sec_xi(zenith_angle)

    if link == 'LEO-HAP' :
        # mu2 = quad(mu2_integral_for_R, PARAMS['HAP_Altitude'], PARAMS['LEO_Altitude'])[0]

        return (
            4.35 * mu2_hap_leo 
            * ( (2 * L / (k * W**2)) ** (5/6) ) 
            * (k**(7/6)) 
            *  ( (PARAMS['LEO_Altitude']-PARAMS['HAP_Altitude'])** (5/6))
            *  (sec_xi**(11/6)) 
            )
        
    elif link == 'HAP-UAV':
        # mu2 = quad(mu2_integral_for_D, PARAMS['UAV_Altitude'], PARAMS['HAP_Altitude'])[0]

        return (
            4.35 *mu2_uav_hap
            * ( (2 * L / (k * W**2)) ** (5/6) ) 
            * (k**(7/6)) 
            *  ( (PARAMS['HAP_Altitude']-PARAMS['UAV_Altitude'])** (5/6))
            *  (sec_xi**(11/6)) 
            )

#Compute A_mod and phi_mod
def compute_phi_mod_and_A_mod(
        zenith_angle, L,
        link):
    
    k_w  = 2 * pi / PARAMS['lambda']

    if link == 'LEO-HAP' : # R stands for Relay (HAP) 
        w0 = 2 * PARAMS['lambda'] / (pi* PARAMS['LEO_divergence_angle'])  
        LEO_sigma_Sxy = PARAMS['LEO_theta_Sxy']  *   L
        sigma_x = torch.sqrt(LEO_sigma_Sxy** 2 + PARAMS['HAP_sigma_x']**2)
        sigma_y = torch.sqrt(LEO_sigma_Sxy** 2 + PARAMS['HAP_sigma_y']**2)
        mu_x, mu_y = PARAMS['HAP_muy_x'], PARAMS['HAP_muy_y']
        a_radius = PARAMS['HAP_Aperture_Radius']


    elif link == 'HAP-UAV' : #D stands for Destination (UAV)
        w0 = 2 * PARAMS['lambda'] / (pi* PARAMS['HAP_divergence_angle']) #Beam waist at L = 0
        HAP_sigma_Rxy = PARAMS['HAP_theta_Rxy']  *   L
        sigma_x = torch.sqrt(HAP_sigma_Rxy** 2 + PARAMS['UAV_sigma_x']**2)
        sigma_y = torch.sqrt(HAP_sigma_Rxy** 2 + PARAMS['UAV_sigma_y']**2)
        mu_x, mu_y = PARAMS['UAV_muy_x'], PARAMS['UAV_muy_y']
        a_radius = PARAMS['UAV_Aperture_Radius']

    else :
        # #print('No link was provided \n')
        return -999, -999

    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y )   
    term1 = 1 - L/ PARAMS['Radius_Curvature'] 
    term2 = (2 * L) / (k_w * w0**2)
    wL = w0 * torch.sqrt(term1** 2 + term2**2)
    Gd = compute_Gd(L, wL ,zenith_angle,link)
    wLe = wL * torch.sqrt(1+Gd)
    v = (a_radius/wLe) * torch.sqrt((pi/2))

    wLe_eq = torch.sqrt(            # wLe_eq is the equivalent beamwidth 
        wLe**2 * 
        (   (torch.sqrt(pi) * torch.erf(v)) 
            / (2 * v * torch.exp(- (v)**2))
        ) 
    )

    phi_mod = wLe_eq / (2*torch.sqrt(sigma_mod)) # phi_mod

    # Compute A_mod
    phi_x = wLe_eq/ (2*sigma_x)
    phi_y = wLe_eq/ (2*sigma_y)
    A_0 = torch.erf(v) **2 #  fraction of the collected power

    A_mod = A_0 * torch.exp(
        1/(phi_mod **2) 
        - 1/(2*phi_x**2) 
        - 1/(2*phi_y**2)
        - (mu_x**2 / (2*sigma_x**2 *phi_x**2) ) 
        - (mu_y**2/  (2*sigma_y**2 *phi_y**2) ) 
    )

    return phi_mod,  A_mod

rytov_integrand = C2n_vals *((h_uav_hap - uav_z) ** (5/6))
rytov_integral = torch.trapz(rytov_integrand, h_uav_hap)

def f_gamma(        
        x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, 
        x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
        gamma_it , #gamma_it : received SNR at UAV
        Mc,
        link):
    
    
    zenith_angle = compute_zenith_angle_torch(x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, x_lower_vehicle, y_lower_vehicle, z_lower_vehicle)
    ha_RD = cloud_attenuation(Mc=Mc,zenith_angle=zenith_angle).unsqueeze(1)
    #print(f"ha_Rd\n{ha_RD}")
    rytov_variance = Rytov_variance(zenith_angle,integral_C2n=rytov_integral).unsqueeze(1)
    #print(f"rytov_variance\n{rytov_variance}")
    #print(f"zenith_angle\n{zenith_angle}")
    psi_RD = compute_B0(link)

    cos_zenith = torch.cos(zenith_angle)
    L = (hap_z - uav_z) / cos_zenith 

    phi_Dmod, A_Dmod = compute_phi_mod_and_A_mod(zenith_angle, L, link)
    #print(f"phiDmod\n{phi_Dmod}")
    #print(f"A_Dmod\n{A_Dmod}")
    phi_Dmod = phi_Dmod.unsqueeze(1)
    A_Dmod = A_Dmod.unsqueeze(1)
    muy_it = rytov_variance * (1+2*phi_Dmod**2) / 2

    term1= phi_Dmod**2 / (4 * gamma_it)

    base_term= (torch.sqrt(gamma_it) / (A_Dmod* ha_RD* torch.sqrt(psi_RD) ) ) 
    term2 = base_term  ** (phi_Dmod**2)
    term3= torch.erfc( (torch.log(base_term) + muy_it)/(torch.sqrt(2 * rytov_variance)))
    term4= torch.exp(rytov_variance * phi_Dmod**2 * (1+phi_Dmod**2) /2)
    return(
        term1 * term2 * term3 * term4
    )
    
def compute_B0(link):
    if link == 'HAP-UAV':
        P_FSO_dBm = PARAMS['HAP_Transmitted_power'] 

    elif link =='LEO-HAP':
        P_FSO_dBm = PARAMS['LEO_Transmitted_power'] 

    P_FSO_W = 10 ** ((P_FSO_dBm-30)/10)
    return torch.tensor((P_FSO_W / PARAMS['HAP_2_UAV_Noise_standard_deviation_']) **2)


# def Ergodic_Capacity(
#         x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, 
#         x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
#         Mc,
#         link='HAP-UAV',
#         device= "cuda"
#     ):
#     integral = lambda gamma,x_lower_vehicle,y_lower_vehicle, z_lower_vehicle, Mc : (
#         PARAMS['B_FSO'] * 1e9 
#         * torch.log2(torch.tensor(1)+ gamma)
#         * f_gamma(         
#             x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, 
#             x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
#             gamma , Mc,
#             link)
#     )

#     C_FSO = []

#     for i in range(len(x_lower_vehicle)):

#         C_FSO.append(quad(integral,
#                           torch.tensor(0.01),
#                           torch.tensor(torch.inf),
#                           args=(x_lower_vehicle[i],
#                                  y_lower_vehicle[i],
#                                  z_lower_vehicle[i], 
#                                  Mc[i]))[0])

#     return torch.tensor(C_FSO, device=device) 

def Ergodic_Capacity_Trapz(
        x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, 
        x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
        Mc,
        link='HAP-UAV',
        device= "cuda"
    ):
    n_samples = 200 
    B_Hz = PARAMS['B_FSO'] * 1e9 
    gamma_it = torch.linspace(0.01, 5.0, n_samples,device=device)
    f_gamma_value = f_gamma(
        x_higher_vehicle, y_higher_vehicle, z_higher_vehicle, 
        x_lower_vehicle, y_lower_vehicle, z_lower_vehicle,
        gamma_it, Mc,
        link)

    log_term = torch.log2(1+gamma_it)
    integrand = B_Hz * log_term * f_gamma_value
    dx = (5.0 -0) / (n_samples-1)

    result = torch.trapz(integrand,dx=dx)


    return result

#==========================================================================#==========================================================================
#==========================================================================#==========================================================================
#==========================================================================#==========================================================================

if __name__ =='__main__':
    pass
    # uav_pos = torch.tensor([[100, 100, 250], [200,200,250]],device=device)
    # hap_pos = torch.tensor([800,800,20e3],device=device)


    # # #print(Ergodic_Capacity(hap_pos[0],hap_pos[1],hap_pos[2], 
    # #               uav_pos[:,0],uav_pos[:,1],uav_pos[:,2],
    # #               Mc=Mc,link='HAP-UAV'))
    # print(f"Ergodic")
    # print(Ergodic_Capacity_Trapz(hap_pos[0],hap_pos[1],hap_pos[2], 
    #               uav_pos[:,0],uav_pos[:,1],uav_pos[:,2],
    #               Mc=torch.tensor(0.5,device=device),link='HAP-UAV'))
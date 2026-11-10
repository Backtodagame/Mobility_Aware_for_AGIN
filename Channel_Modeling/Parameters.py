#   MAP
map_center = 800
map_maximum_xy_axis   = 800 * 2
map_cell_size  = 200
n_users = 200
n_uavs = 3
n_cells = map_maximum_xy_axis // map_cell_size

#   DQN :
STEP  = 20 



#   Free Space Optical Channel:
lambda_fso_wavelength = 1.55e-6
velocity_wind = 21 
Nc            = 250 
Hcl           = 2e3 
B_FSO         = 3 
C2n_0          = 1e-13 

#   Radio Frequency Channel:
xi_los = 1
xi_nlos = 20
a,b = 9.61, 0.16
N0 = -80.4      #      
UAV_P_RF = 35   #   dBm
HAP_P_RF = 40   #   dBm
c = 3e8         #   3e8 m/s


#   LEO
leo_z = 500e3
leo_divergence_angle  = 1e-6
leo_transmitted_power = 15
leo_theta_sxy  =  0.3e-6 
leo_hap_noise = 1e-7 

#   HAP 
hap_z = 20e3 
hap_divergence_angle = 1e-3
hap_aperture_radius = 5
hap_tranmistted_power =6 
hap_detector_responsivity = 0.9
hap_uav_noise = 1e-7

hap_muy_x, hap_muy_y = 0, 0
hap_sigma_x, hap_sigma_y = 1,1  # 1 meter
hap_theta_Rxy= 50e-6

#   UAV
uav_z = 250
uav_radius_coverage = 250  
uav_muy_x, uav_muy_y = 0,0
uav_sigma_x, uav_sigma_y = 0.8,1 # 1 meter
uav_Aperture_Radius = 4/100 # meter 
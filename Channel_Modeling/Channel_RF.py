# UAV -> UE: use Radio Frequency 
from Parameters import * 
PARAMS = {
    'HAP_P_RF' : HAP_P_RF,    # RF Transmitted Power 40(dBm)
    'UAV_P_RF' : UAV_P_RF,     # 35 dBm
    'a' :   a, #Environment Parameters
    'b'   : b,
    'xi_los': xi_los, #average excessive pathloss 1 (dB)
    'xi_nlos' : xi_nlos, # 20
    'N_0' : N0,      # Noise power -80.4 (dBm)
     'c'  : c,        # Speed of Light (m/s)
    'n_users' : n_users, # 
    'hap_altitude': hap_z, # 20 km

}

import torch 



def cal_probability_having_LoS(vertical_dist, horizontal_dist):
    return (
        1 / (1 + PARAMS['a']*torch.exp(-PARAMS['b']*(180/torch.pi* torch.arctan(vertical_dist/horizontal_dist)) -PARAMS['a']))
    )

def cal_avg_pathloss(dist_3d, p_LoS, current_fc):
    #cal path loss LoS and NLoS
    eta_LoS = 20* torch.log10(4*torch.pi*current_fc*dist_3d/ c) + xi_los
    eta_NLoS = 20* torch.log10(4*torch.pi*current_fc*dist_3d/ c) + xi_nlos

    return (
       p_LoS * eta_LoS + (1- p_LoS)*eta_NLoS
    )

def cal_dbW(dbm):
    return (
        10 ** ((dbm-30)/10)
    )
    


def cal_R_FP_batch(ue_pos_t, fp_pos,base_station='uav',device='cuda', 
                   B_RF=600e6, fc = 2e9):
    """
    Calculate Data Rate betweeen Flying Platforms (uav/hap) and users
    
    Agrs:
        ue_pos_t : user position at time step t
        fp_pos   : flying platform position
    """
    # Old: d = torch.norm(user_pos[:, :2].unsqueeze(1) - fp_pos[:, :2].unsqueeze(0), dim=2)  # (N, M)
    # New :
    if base_station == 'uav':
        vertical_dist = uav_z
        horizontal_dist = torch.linalg.norm(ue_pos_t[:,:2].unsqueeze(1) - fp_pos[:,:2],dim=2) # ([n_user, n_uav])

    elif base_station == 'hap':
        vertical_dist = hap_z
        horizontal_dist = torch.linalg.norm(ue_pos_t[:,:2].unsqueeze(1) - fp_pos[:2],dim=2) # ([n_user, 1])

    else :
        print(f"no base station was chosen")
        return -1
    
    dist_3d = torch.sqrt(horizontal_dist**2 + vertical_dist**2)

    P_los = cal_probability_having_LoS(vertical_dist , horizontal_dist )
    avg_eta = cal_avg_pathloss(dist_3d= dist_3d , p_LoS= P_los,current_fc= fc)

    avg_eta_exponent = 10** (-avg_eta / 10)

    N0_W_HZ = cal_dbW(PARAMS['N_0'])
    if base_station == 'uav':
        P_RF_W = cal_dbW(PARAMS['UAV_P_RF'])
    elif base_station == 'hap':
        P_RF_W = cal_dbW(PARAMS['HAP_P_RF'])
    else :
        print(f"no base station was chosen")
        return -1
    
    SNR = P_RF_W * avg_eta_exponent /  N0_W_HZ
    data_rate   = (B_RF / n_users) * torch.log2(1+SNR)

    return data_rate



if __name__ =="__main__":
    

    pass 
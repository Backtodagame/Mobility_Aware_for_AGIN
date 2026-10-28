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




# def cal_R_UAV_batch_mmWave(
#         uav_pos_t, uav_pos_prev, user_pos_t,user_pos_prev, t, 
#         B=400e6, fc=28e9, delta_f=120e3, c=3e8):

#     with torch.no_grad():
#         d_2d = torch.linalg.norm(user_pos_t[:, :2].unsqueeze(1) - uav_pos_t[:, :2].unsqueeze(0), dim=2)
#         d_3d = torch.sqrt(d_2d**2 + uav_z**2)
#         P_los = cal_probability_having_LoS(uav_z, d_2d )
#         avg_eta = cal_avg_pathloss(dist_3d=d_3d, p_LoS=P_los, current_fc=fc)
#         Prx_w = cal_dbW(UAV_P_RF) * 10 **(-avg_eta/10)

#         if t>0: 
#             user_v_3d = user_pos_t- user_pos_prev
#         else :
#             user_v_3d = torch.zeros_like(user_pos_t)        
#         uav_v_3d = uav_pos_t - uav_pos_prev
#         v_rel_vector_3d = uav_v_3d.unsqueeze(0) - user_v_3d.unsqueeze(1)
#         los_vector_3d = uav_pos_t.unsqueeze(0) - user_pos_t.unsqueeze(1)
#         d_3d_tensor = torch.linalg.norm(los_vector_3d,dim=2, keepdim=True)
#         unit_los_vector = los_vector_3d / (d_3d_tensor + 1e-9)
#         v_radial = torch.sum(v_rel_vector_3d * unit_los_vector, dim=2)
#         v_radial_mag = torch.abs(v_radial)
#         fd = v_radial_mag * fc  / c 
#         ici_factor = (torch.pi**2/3.0) * (fd/delta_f)**2
#         P_ici = Prx_w * ici_factor
#         N0_W = cal_dbW(N0)
#         SINR = Prx_w / (N0_W + P_ici)
#         rate = (B/n_users) * torch.log2(1+SINR)

#         return rate


if __name__ =="__main__":
    import System_Modeling_mmWave as SM 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_pos = SM.simulate_user_mobility(n_users,5,debug=True)
    uav_pos_prev = torch.tensor([[200,200,250],
                                 [500,500,250],
                                 [1200,1200,250]],device=device)   
    uav_pos_t = torch.tensor([[200,220,250],
                              [500,520,250],
                              [1180,1200,250]
                            ],device=device)
    all_uav_rates = cal_R_FP_batch(user_pos[:,0,:],uav_pos_t,base_station='uav')

    hap_pos = torch.tensor([800,800,20e3],device=device)

    hap_rate =cal_R_FP_batch(user_pos[:,0,:],hap_pos,base_station='hap')
    print(f"all uav rates\n{all_uav_rates.mean()}")
    # print(all_uav_rates - hap_rate)
    print(f"hap_rate {hap_rate.mean()}")

    # print(rate-hap_rate)


    a =torch.tensor([[2,2],
         [2,2]])

    b =torch.tensor([[1,3],
         [4,6]])

    print(a*b)

    print(cal_dbW(30)) 
    pass 
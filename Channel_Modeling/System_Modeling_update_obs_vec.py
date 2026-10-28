
from Channel_FSO import PARAMS as Channel_FSO_PARAMS
import Channel_FSO as ch_fso 
import Channel_RF as ch_rf
from Parameters import *

import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import numpy as np 
import torch 
import torch.nn as nn 

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

PARAMS = {
    # MAP 
    'map_center' : map_center,         
    'map_cell_size'  : map_cell_size,


    # User Positions
    'num_users' : n_users,
    'num_uavs'  : n_uavs,
    'STEP' : STEP,
}

class DQN(nn.Module):
    """
    Deep Q-Network for UAV model

    Agrs:
        n_observation (int): observation vector for input
        n_action (int)     : UAV action
    
    Returns:
        the best action in each observation vector 
    """
    def __init__(self, n_observation, n_action):
        super().__init__()
        self.linear_relu_stack = nn.Sequential (
            nn.Linear(n_observation, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_action)
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

def move_uav(uav_pos, action):
    """
    Action Space:
        Discrere space: the numver of possible action is finite (5)
    
    Args:
        uav_pos : uav position in tensor [(n_uavs , 2)]
        action  (int): specific number representing for action 
    
    Returns:
        uav_pos after performing action    
    """
    moved = uav_pos.clone()
    for i,a in enumerate (action):
        if a==0: moved[i][0] += STEP  # moving 1 step East 
        elif a == 1: moved[i][0] -= STEP # West
        elif a == 2: moved[i][1] += STEP # North
        elif a == 3: moved[i][1] -= STEP  # South
        else :
            pass
        moved[i][0] = torch.clip(moved[i][0],0, map_maximum_xy_axis - 1)
        moved[i][1] = torch.clip(moved[i][1],0, map_maximum_xy_axis - 1)
    return moved

def simulate_user_mobility(
        n_users, n_steps,
        distribution='gaussian', n_hotspot = n_uavs,
        ratio_mobile = 0.2,
        device ="cuda", 
        dtype = torch.float32
        ):

    """
    Simulate user mobility with 20% of them will move, while the rest
    is standing (not moving) 

    Args: 
        n_users (int) : Number of users
        n_steps (int) : Number of time steps
        map_center (float): Half of the area size 
        distribution (str) : User mobility models
        n_hotspot (int) : Number of hotspots 
        ratio_mobile (float) : Ratio of moving users and total one
    Returns:
        positions: np.ndarray of shape (n_users, n_steps, 2) 
    """

    positions = np.zeros((n_users, n_steps, 2))  
    n_mobile  = int(ratio_mobile * n_users)
    mobile_indices = np.random.choice(n_users, n_mobile, replace=False)
    static_indices = [i for i in range(n_users) if i not in mobile_indices]

    if distribution == 'gaussian':

        if n_hotspot > 1 :
            spot_centers = []

            for _ in range(n_hotspot):
                mean = np.random.uniform(300, map_center * 2 - 300,2)   
                spot_centers.append(mean)       
            spot_users = [0]

            while any(np.array(spot_users) <=0 ):
                nums = np.random.normal(n_users // n_hotspot,20,size= n_hotspot)
                spot_users = [int(num) for num in nums]
                spot_users[-1] = n_users - sum(spot_users[:-1])
            idx = 0

            for i in range(n_hotspot):
                count = spot_users[i]          
                group = np.arange(idx, idx+ count)  
                init_pos = np.random.normal(spot_centers[i], 200, (count, 2)) 
                positions[group, 0, :] = init_pos 
                idx +=count 

        else:
            mean = np.random.uniform(300, map_center * 2 -300, 2)
            positions[:,0,:] = np.random.normal(mean, 300, (n_users,2))
    elif distribution == 'uniform':
        positions[:,0,:] = np.random.uniform(0,map_center*2, (n_users, 2))


    # Clip initial positions to boundaries in map 
    positions[:,0,:] = np.clip(positions[:,0,:], 0, map_center * 2)
    positions[static_indices,:,:] = positions[static_indices, 0, :][:,None,:]

    alpha = 0.75
    avg_speed = 1.358 #m/s Men who is (20-29) age years old.  
    delta_t = 1.0 
    speeds = np.zeros(n_users)
    directions = np.random.uniform(low=0, high=360,size=n_users)
    avg_direction = directions.copy()

    
    for t in range(1,n_steps):

        s_x = np.random.normal(0,1, size=n_mobile)
        d_x = np.random.normal(0,1, size=n_mobile)

        speeds[mobile_indices]     = (alpha * speeds[mobile_indices]     + (1-alpha)* avg_speed                     + np.sqrt((1-alpha**2))*s_x)
        directions[mobile_indices] = (alpha * directions[mobile_indices] + (1-alpha)* avg_direction[mobile_indices] + np.sqrt((1-alpha**2))*d_x)

        pre_x = positions[mobile_indices, t-1, 0]
        pre_y = positions[mobile_indices, t-1, 1]

        new_x = np.round(pre_x + speeds[mobile_indices]* np.cos(np.deg2rad(directions[mobile_indices])) * delta_t ,2)
        new_y = np.round(pre_y + speeds[mobile_indices]* np.sin(np.deg2rad(directions[mobile_indices])) * delta_t ,2)
        
        limit_map = map_center * 2
        out_right = new_x > limit_map
        out_left  = new_x < 0
        out_top   = new_y > limit_map
        out_bot   = new_y < 0

        corner_upper_right = out_right & out_top        
        avg_direction[mobile_indices[corner_upper_right]] = 225
        new_x[corner_upper_right], new_y[corner_upper_right] = limit_map, limit_map

        corner_upper_left = out_left & out_top
        avg_direction[mobile_indices[corner_upper_left]] = 315
        new_x[corner_upper_left], new_y[corner_upper_left] = 0, limit_map

        corner_lower_right = out_right & out_bot
        avg_direction[mobile_indices[corner_lower_right]] = 135
        new_x[corner_lower_right], new_y[corner_lower_right] = limit_map, 0

        corner_lower_left = out_left & out_bot
        avg_direction[mobile_indices[corner_lower_left]] = 45
        new_x[corner_lower_left], new_y[corner_lower_left] = 0, 0

        edge_right = out_right & ~out_top & ~out_bot
        avg_direction[mobile_indices[edge_right]] = 180
        new_x[edge_right] = limit_map

        edge_left = out_left & ~out_top & ~out_bot
        avg_direction[mobile_indices[edge_left]] = 0 
        new_x[edge_left] = 0

        edge_top = out_top & ~out_right & ~out_left
        avg_direction[mobile_indices[edge_top]] = 270
        new_y[edge_top] = limit_map

        edge_bot = out_bot & ~out_right & ~out_left
        avg_direction[mobile_indices[edge_bot]] = 90
        new_y[edge_bot] = 0


        positions[mobile_indices, t, 0] = new_x
        positions[mobile_indices, t, 1] = new_y

    postions_3d = np.zeros((n_users, n_steps, 3))
    postions_3d[:,:,:2] = positions

    return torch.tensor(postions_3d,dtype=dtype,device=device)

def plot_user(user_positions, time_step=1):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(user_positions[:,time_step,0], user_positions[:,time_step,1], color='blue', s=8, label='Users')
    ax.set_xlim(0, map_maximum_xy_axis)
    ax.set_ylim(0, map_maximum_xy_axis)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_user_heatmap(user_pos_t, device= "cuda"):
    """
    Build user heatmap from their position at time step t 

    Args:
        user_pos_t ((num_users, t, 2))
    
    Returns:
        User heatmap at specified time step ((n_cells, n_cells))
    """
    xy = user_pos_t[:,:2]
    indices = (xy / map_cell_size).long()
    indices = torch.clamp(indices, min =0, max=n_cells-1) 
    cols = indices[:,0]
    rows = indices[:,1] 
    flat_indices = rows * n_cells + cols  
    heatmap_1d = torch.bincount(flat_indices, minlength=n_cells*n_cells).float()
    heatmap = heatmap_1d.view(n_cells, n_cells)
    max_val = heatmap.max()

    if max_val > 0 :
        heatmap = heatmap / max_val
        
    return heatmap
    

def torch_get_full_cloud(clwc_range=[0.5,7.5],device ="cuda"):
    """
    Build cloud map
    
    Agrs:
        None
    
    Returns:
        cloud map with size (area, area*4))
    """
    cloud_map = torch.empty(
        size=(n_cells, n_cells*4), dtype=torch.float32, device=device
        ).uniform_(0.5,7.5)
    cloud_map = F.interpolate(
        cloud_map.unsqueeze(0).unsqueeze(0),
        size = (map_maximum_xy_axis, map_maximum_xy_axis*4),
        mode="bicubic")
    
    cloud_map = torch.clamp(cloud_map, min=clwc_range[0], max=clwc_range[1])
    return cloud_map



def get_cloudmap_t(t, cloud_map,device="cuda"):
    """
    Get cloud map at time step t in the coordinate origin (1600, 1600)

    Args:
        t (int) : time step 
        concentrated_cloud_map (area): the cloud map which is stack by a number of its copy

    Returns:
        region_t ((area, area)): cloud map at time step t 
        cell_clwc_t ((n_cells, n_cells)): clwc value calculated in cells  
    """
    shift = int(4 * t)
    region_t = cloud_map[0,0,:,shift:shift+map_maximum_xy_axis]
    region_t_4D = region_t.unsqueeze(0).unsqueeze(0)

    cell_clwcl_t = F.avg_pool2d(
        region_t_4D,
        kernel_size=map_cell_size,
        stride=map_cell_size)
    
    cell_clwcl_t = cell_clwcl_t.squeeze() / 7.5

    return region_t, cell_clwcl_t

def perform_user_association(uav_pos, user_pos_t, region_t, HAP_position, device ="cuda"):
    """
    Perform user association based on valid UAV range and capacity.

    Args:
        uav_pos  tensor ((num_uav, 3))          # 3 stands for x,y,z value in coordinate origin
        user_pos_t tensor ((num_users, num_step, 3))  # 3 stands for x,y,z value in coordinate origin
        region_t tensor ((area, area))
        HAP_position tensor ((1 , 3))           # 3 stands for x,y,z value in coordinate origin

    Returns:
        alpha: UE/UAV association matrix ((num_users, num_uav))
        beta : UE/HAP association tensor (num_users)
        RUAV : Total Rate per UAV       (num_uav)    
        CFSO : FSO Capacity per UAV     (num_uav)
    """
    with torch.no_grad():
        x_idx = uav_pos[:, 0].long().clamp(0, region_t.shape[1]-1)
        y_idx = uav_pos[:, 1].long().clamp(0, region_t.shape[0]-1)
        Mc = region_t[y_idx, x_idx]
        CFSO = ch_fso.Ergodic_Capacity_Trapz(
            x_higher_vehicle=HAP_position[0], y_higher_vehicle=HAP_position[1], z_higher_vehicle=HAP_position[2],
            x_lower_vehicle=uav_pos[:,0], y_lower_vehicle=uav_pos[:,1], z_lower_vehicle=uav_pos[:,2],
            Mc=Mc,link='HAP-UAV'
        )

        CFSO_remaining = CFSO.clone()
        dists = torch.linalg.norm(user_pos_t[:,:2].unsqueeze(1) - uav_pos[:,:2],dim=2) #([n_users, n_uavs])
        is_user_in_uav_range = dists <= uav_radius_coverage
        all_uav_rates = ch_rf.cal_R_FP_batch(ue_pos_t=user_pos_t, fp_pos= uav_pos, base_station='uav')
        hap_rates     = ch_rf.cal_R_FP_batch(ue_pos_t=user_pos_t, fp_pos= HAP_position, base_station='hap')
        is_uav_rate_better_than_hap = all_uav_rates > hap_rates
        valid = is_user_in_uav_range & is_uav_rate_better_than_hap  #([n_user, n_uav])
        alpha = torch.zeros((n_users,n_uavs), dtype=torch.bool,device=device)
        beta  = torch.ones(n_users, dtype=torch.bool,device=device)
        RUAV =  torch.zeros(n_uavs, dtype=torch.float32,device=device)
        valid_rates = valid * all_uav_rates
        sorted_rates, sorted_uav = torch.sort(valid_rates,dim=1,descending=True)

        for user_idx in range(n_users):
            for i in range(n_uavs):
                rate = sorted_rates[user_idx,i].item() #return float instead of tensor with one element

                if rate <=0 :
                    break

                uav_idx = sorted_uav[user_idx, i]

                if CFSO_remaining[uav_idx].item() >=rate :
                    alpha[user_idx,uav_idx] =True
                    beta[user_idx] = False
                    RUAV[uav_idx] +=rate
                    CFSO_remaining[uav_idx] -=rate
                    break # Next user 


        return alpha, beta, RUAV, CFSO
                

def build_observation_vector(uav_pos, user_pos_t, cell_clwc_t, alpha, beta, device="cuda"):
    """
    Build the observation input vector (a.k.a Observation Space)

    Args: 
        uav_pos  (( num_uav, 3)) : Position of each UAV in 3 dimensions torch
        user_pos_t : Position of users in a specific time step
        cell_clwc_t (( n_cells, n_cells)): cell clwc value
        alpha    (( num_users, num_uav))
        beta     (num_uav) 
    Returns:
        Observation space vector with shape torch.Tensor ((3,130))
    """

    obs_list = []
    normalized_uav_xy = uav_pos[:,:2] / map_maximum_xy_axis
    
    for i in range(n_uavs):
        served_mask = (alpha[:, i] ==1 ) | (beta ==1)   # | is OR operator        
        served_users_xy = user_pos_t[served_mask]
        user_heatmap = build_user_heatmap(served_users_xy) 
        other_uav_pos = torch.cat([normalized_uav_xy[:i], normalized_uav_xy[i+1:]])
        obs_vector  = torch.concatenate([normalized_uav_xy[i],other_uav_pos.flatten(), user_heatmap.flatten(), cell_clwc_t.flatten()])
        obs_list.append(obs_vector)
    obs_batch = torch.stack(obs_list, axis=0)

    return obs_batch


def calculate_rewards(RUAV, CFSO, uav_pos, Rthres=0.6e9, w=0.3 , p=0.5, device ="cuda"):
    """
    Reward Function: 
        Calculate rewards based on user association results. This stage is fundemental 
        in RL because it is the only feedback for the agent. Thanks to it, out agent knows
        if the action taken was good or not.

    Args:
        RUAV: Total rate per UAV (tensor)
        CFSO: FSO Capacity per UAV (tensor)
        uav_pos: Position of each UAV in 3 dimensions tensor (( num_uav, 3))
        Rthres: Rate threshold
        w: Global reward weight 
        p: overlap penalty

    Returns:
        rewards: List of rewards for each UAV
    """

    with torch.no_grad():
        num_uav = CFSO.shape[0]
        if not isinstance(Rthres, torch.Tensor):
            Rthres = torch.tensor(Rthres, device=device)

        # UAV overlaps
        delta = uav_pos[:,:2].unsqueeze(1) - uav_pos[:,:2].unsqueeze(0)
        dists_uav = torch.linalg.norm(delta, dim =2)
        overlaps = (dists_uav < 2* uav_radius_coverage) & (~torch.eye(n=num_uav, device=device, dtype=torch.bool))
        nc = overlaps.sum(dim=1)

        # global reward 
        total_rate = torch.sum(RUAV)
        total_capacity = torch.sum(CFSO)
        thres = num_uav * Rthres
        sqrt_sum = torch.sqrt((total_rate * total_capacity/ thres**2))
        g_reward = sqrt_sum - (0.0 if torch.all(RUAV > Rthres) else 1.0)

        # local reward 
        sqrt_term = torch.sqrt((RUAV / Rthres) * (CFSO/Rthres))
        l_reward  = torch.where(RUAV > Rthres, sqrt_term -nc*p, sqrt_term - nc*p -1) 

        #total reward
        total_reward = w * g_reward + (1-w) * l_reward

        return total_reward.tolist()


def user_association_and_calculate_reward(uav_pos, user_pos_t, region_t, HAP_position ,device ="cuda"):
    """
    Perform user association, then calculate reward (combining perform_user_association and calculating rewards functions) 

    Args:
        uav_pos: UAV position (tensor)
        user_pos_t: user position at time step t (tensor)
        region_t: clwc map at time step t 
        HAP_position: HAP position (tensor)

    Returns:
        rewards : 
        alpha: UE/UAV association matrix ((num_users, num_uav))
        beta : UE/HAP association tensor (num_users)
        RUAV : Total Rate per UAV       (num_uav)
        CFSO : FSO Capacity per UAV (num_uav)    
    """

    alpha, beta, RUAV, CFSO = perform_user_association(uav_pos, user_pos_t, region_t,HAP_position,device=device)
    rewards = calculate_rewards(RUAV, CFSO,uav_pos,device=device)

    return rewards, alpha, beta, RUAV, CFSO

def evaluate_trained_policy(
        DQN, action_size, n_uavs, n_steps, uav_pos_init, user_pos, 
        cloud_map, hap_pos, episode_to_load, device="cuda", save_dir = "/kaggle/working/", method ="single"
):
    """
    Evaluate trained policy

    Args:
        DQN: Deep Q Network that is used to train 
        action_size: action size of UAV
        n_uavs: number of uav
        n_steps: total steps
        uav_pos_init: initial uav position
        user_pos: user position
        cloud_map: cloud map 
        hap_pos: hap position 
        episode to load: which episode to evaluate 
        save_dir: save directory in kaggle 
    
    Returns:
        RUAV_log: Total Data Rate served by UAV
        CFSO_log: Allocated FSO to UAV
        uav_traj: uav trajectory 
    """
    if method == "multi":
        obs_vec_length = 2 * n_uavs + n_cells**2*2  #multi means that observation vector will include other uav position
    else :
        obs_vec_length = 2 + n_cells**2 * 2 # single means that we consider only each uav position + ( user_heatmap + cloud_map (flatten both of them))

    Q_nets = [DQN(obs_vec_length, action_size).to(device) for _ in range(n_uavs)]
    
    for i in range(n_uavs):
        path = f"{save_dir}/q_net_{i}_ep{episode_to_load}.pt"
        Q_nets[i].load_state_dict(torch.load(path, map_location = device))
        Q_nets[i].eval()

    alpha, beta = None, None
    RUAV_log, CFSO_log, uav_traj = [],[],[]
    uav_pos = uav_pos_init.copy()

    for t in range(n_steps):
        region_t, cell_clwc_t = get_cloudmap_t(t, cloud_map)
        user_pos_t = user_pos[:,t,:]

        if t == 0 :
            _, alpha, beta, _ = user_association_and_calculate_reward(
                uav_pos, user_pos_t, region_t, hap_pos
            )

        obs_vec = build_observation_vector(uav_pos, user_pos_t, cell_clwc_t,alpha, beta)
        
        with torch.no_grad():
            actions = [Q_nets[i](obs_vec[i]).argmax() for i in range(n_uavs)]

        uav_pos = move_uav(uav_pos, actions)
        rewards, alpha, beta, RUAV, CFSO = user_association_and_calculate_reward(
            uav_pos, user_pos_t, region_t, hap_pos
        )
        RUAV_log.append(RUAV)
        CFSO_log.append(CFSO)
        uav_traj.append(uav_pos.copy())

    return RUAV_log, CFSO_log, uav_traj



if __name__ =='__main__':
    device = "cuda"
    cloud_map = torch_get_full_cloud()
    region_t, cell_clwc_t = get_cloudmap_t(0,cloud_map)
    n_users = 200
    user_pos = simulate_user_mobility(n_users, n_steps)
    # print(f"user_pos {user_pos[:,0,:].shape}")
    uav_pos =torch.tensor([[400,  400,  250],
                          [400,420, 250],
                          [1200,1200,250]], dtype=torch.float32, device="cuda")
    
    hap_pos = torch.tensor([800,800,20e3],dtype=torch.float32,device=device)
    alpha, beta, RUAV, CFSO =perform_user_association(uav_pos,user_pos[:,0,:],region_t,hap_pos)

    obs_vec=build_observation_vector(uav_pos,user_pos[:,0,:],cell_clwc_t,alpha,beta)

    print(f"obs_vec\n{obs_vec}")
    print(f"obs_vec\n{obs_vec.shape}")

    # print(f"len_obs_vec {obs_vec.shape}")
    # stacked_obs_vec = torch.cat((obs_vec,obs_vec),dim=-1)
    # print(f"stacked_obs\n{stacked_obs_vec}")
    # print(f"len {stacked_obs_vec.shape}")
    # print(f"alpha\n{alpha}")
    print(f"served people: UAV 1:{torch.sum(alpha[:,0])} UAV 2: {torch.sum(alpha[:,1])} UAV 3: {torch.sum(alpha[:,2])}")
    # print(f"beta\n{beta}")
    print(f"RUAV\n{RUAV}")
    print(f"CFSO\n{CFSO}")
    # simulate_user = torch.tensor([[800,800,0],[150,150,0]],device=device)

    # # cfso_by_uav = ch_fso.Ergodic_Capacity_Trapz()

    # uav_pos1= torch.tensor([[0,0,250]],dtype=torch.float32,device=device)
    # CFSO = ch_fso.Ergodic_Capacity_Trapz(
    #     hap_pos[0],hap_pos[1], hap_pos[2],
    #     uav_pos1[:,0], uav_pos1[:,1],uav_pos1[:,2],
    #     Mc=torch.tensor(0.5,device=device))

    # R_uav = ch_rf.cal_R_FP_batch(ue_pos_t=simulate_user,fp_pos=uav_pos1)
    # R_hap = ch_rf.cal_R_FP_batch(ue_pos_t=user_pos[:,0,:],fp_pos=hap_pos,link=2)
    # # print(f"R_UAV\n{R_uav}\nR_HAP\n{R_hap}")

    # print(f"total Rate by R_hap {torch.sum(R_hap,dtype=torch.float32)}")


    # print(f"CFSO {CFSO}")
    pass

    
    """
    >python range.py
    float16: Max = 65504.0, Min = -65504.0, Resolution = 0.0010004043579101562
    float32: Max = 3.4028234663852886e+38, Min = -3.4028234663852886e+38, Resolution = 9.999999974752427e-07
    float64: Max = 1.7976931348623157e+308, Min = -1.7976931348623157e+308, Resolution = 1e-15
    --------------------------------------------------
    int8: Max = 127, Min = -128
    int16: Max = 32767, Min = -32768
    int32: Max = 2147483647, Min = -2147483648
    int64: Max = 9223372036854775807, Min = -9223372036854775808
    uint8: Max = 255, Min = 0
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # user_pos = simulate_user_mobility(20,5)
    # user_pos_0 = user_pos[:,0,:]
    # uav_pos = torch.tensor([[200,200, uav_z],
    #                         [100,100, uav_z],
    #                         [300,300,uav_z]], dtype= torch.float32,device=device)

    # hap_pos = torch.tensor([map_maximum_xy_axis/2, map_maximum_xy_axis/2, hap_z], dtype= torch.float32, device=device)
    # cloudmap = torch_get_full_cloud()
    # region_t, cell_clwc_t = get_cloudmap_t(0, cloudmap)
    # alpha, beta,RUAV, CFSO = perform_user_association(uav_pos, user_pos_0, region_t, hap_pos)
    # obs_vector = build_observation_vector(uav_pos,user_pos_0, cell_clwc_t,alpha,beta)


    # print(f"obs_vector {obs_vector}")

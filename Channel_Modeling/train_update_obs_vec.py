import torch
import torch.nn as nn
import torch.optim as optim
from System_Modeling_update_obs_vec import * 
from collections import deque
from tqdm import trange 
import random 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_size = 5
STEP  = 20 
batch_size = 64
gamma = 0.99 # discounted factor
epsilon_start = 1.0
epsilon_min = 0.05
n_episodes = 2001
n_steps = 300
n_steps_test=1000
epsilon_decay  = (epsilon_start - epsilon_min) / (2/3 * n_episodes * n_steps)  
# 2/3 total step is exploration, the rest of step is exploitation
# epsilon_start - (2/3 * n_episodes * n_steps * x ) < epsilon_min => x = 2.375e-5
learning_rate = 1e-3

def train(device=device):

    total_cells = (map_maximum_xy_axis // map_cell_size) **2
    Q_nets = [DQN(2*3+total_cells*2,action_size).to(device) for i in range(n_uavs)]   #    2 + total_cells**2 is equal the number of length [x_uav, y_uav, user heatmap, cloud heatmap]
    target_Q_nets = [DQN(2*3+total_cells*2,action_size).to(device) for i in range(n_uavs)]

    for i in range(n_uavs):
        target_Q_nets[i].load_state_dict(Q_nets[i].state_dict())
        for param in target_Q_nets[i].parameters():
            param.requires_grad = False

    optimizers = [optim.Adam(Q_nets[i].parameters(), lr=learning_rate) for i in range(n_uavs)] 
    replay_buffers = [deque(maxlen=1_000_000) for _ in range(n_uavs)]
    hap_pos = torch.tensor([map_maximum_xy_axis/2, map_maximum_xy_axis/2,hap_z], dtype= torch.float32,device=device)
    total_step = 0
    episode_return = []

    for episode in trange(n_episodes):
        cloud_map = torch_get_full_cloud() # tensor ([1600, 12800])
        user_pos = simulate_user_mobility(n_users,n_steps)
        uav_pos = torch.randint(0,map_maximum_xy_axis, (n_uavs, 3),dtype=torch.float32, device=device )
        uav_pos[:,2] = uav_z
        total_rewards = torch.zeros(n_uavs)
        total_loss    = torch.zeros(n_uavs)
        loss_counts   = torch.zeros(n_uavs)
        
        # Gemini feedback: initialize time step 0 here 
        region_0, cell_clwc_0 = get_cloudmap_t(0,cloud_map)
        user_pos_0 = user_pos[:,0,:]
        _, alpha0, beta0, _ ,_= user_association_and_calculate_reward(
            uav_pos, user_pos_0, region_0, hap_pos,device
        )
        obs_vec = build_observation_vector(
            uav_pos, user_pos_0, cell_clwc_0, alpha0, beta0
            ) # ([3,134]) # 3 stands for n_uavs, 134 stands for obs_vec length

        for t in range(n_steps):
            total_step +=1
            actions = []
            epsilon = max(epsilon_min, epsilon_start - total_step * epsilon_decay)
            for i in range(n_uavs):
                # if random.random()
                if random.random() < epsilon :
                    actions.append(random.randint(0, action_size -1)) 
                else :
                    with torch.no_grad():
                        q_values = Q_nets[i](obs_vec[i])
                        actions.append(q_values.argmax().item())

            uav_pos = move_uav(uav_pos, actions)
            next_t = min(t+1,n_steps-1)
            next_region_t, next_cell_clwc_t = get_cloudmap_t(next_t, cloud_map)
            next_user_pos_t = user_pos[:,next_t,:]

            rewards, alpha, beta, _,_ = user_association_and_calculate_reward(
                uav_pos, next_user_pos_t, next_region_t, hap_pos,device
                )
            next_obs_vec = build_observation_vector(
                uav_pos, next_user_pos_t,next_cell_clwc_t, alpha,beta
                ) # ([3,130]) # 3 stands for n_uavs, 130 stands for obs_vec length 

            for i in range(n_uavs):
                done = 1.0 if t == n_steps -1 else 0.0
                replay_buffers[i].append((obs_vec[i].detach(), actions[i], rewards[i], next_obs_vec[i].detach(), done))

            total_rewards += torch.tensor(rewards)

            for i in range(n_uavs):
                if (len(replay_buffers[i]) >= batch_size ):
                    minibatch = random.sample(replay_buffers[i], batch_size) # lấy mẫu
                    states = torch.stack([s[0] for s in minibatch]).to(device) # thêm các hàm trạng thái
                    actions_batch = torch.tensor([s[1] for s in minibatch], dtype=torch.long, device=device) # 
                    rewards_batch = torch.tensor([s[2] for s in minibatch], dtype=torch.float32, device=device)
                    next_states = torch.stack([s[3] for s in minibatch]).to(device)
                    dones_batch = torch.tensor([s[4] for s in minibatch], dtype=torch.float32, device=device)

                    with torch.no_grad():
                        next_q = target_Q_nets[i](next_states)  # torch 
                        max_next_q_values = next_q.max(1).values
                        y = rewards_batch +  (1-dones_batch) * gamma * max_next_q_values
                        y = y.unsqueeze(1)
                        
                    Q_values = Q_nets[i](states).gather(1,actions_batch.unsqueeze(1))
                    loss = nn.SmoothL1Loss()(Q_values, y)
                    optimizers[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(Q_nets[i].parameters(), clip_value=1.0)
                    optimizers[i].step()      
                    total_loss[i] +=loss.item()
                    loss_counts[i] +=1
                    
            # Reset target Q == Q for every fix 100 steps
            if total_step % 100 ==0:
                for i in range(n_uavs):
                    target_Q_nets[i].load_state_dict(Q_nets[i].state_dict()) 

        
            obs_vec = next_obs_vec

        episode_return.append(total_rewards)

        if episode % 500 == 0:
            avg_reward = torch.mean(total_rewards).item()
            avg_loss = torch.mean(total_loss/ loss_counts +1e-9).item()
            print(f"Episode: {episode}/{n_episodes} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f}")
            
            save_dir = '/kaggle/working/save_models'
            os.makedirs(save_dir, exist_ok= True)

            for i in range(n_uavs):
                torch.save(Q_nets[i].state_dict(), os.path.join(save_dir,f"q_net_uav{i}_ep{episode}.pt"))
                # torch.save(target_Q_nets[i].state_dict(), os.path.join(save_dir, f"target_q_net_uav{i}_ep{episode}.pt"))
                # torch.save(optimizers[i].state_dict(), os.path.join(save_dir,f"optimizer_uav{i}_ep{episode}.pt"))

            print(f"Model and training state saved at episode {episode}\n")
    print(f"OK Models saved successfully\n")
    return episode_return

def evaluate_trained_policy(
    device=device, 
    save_dir = "/kaggle/input/datasets/trisleevawnminh/madrl-sagin/save_models",
    episode_to_load =str(n_episodes-1),
    n_steps_test= n_steps_test
):
    """
    Evaluate trained policy

    Args:
        DQN: Deep Q Network that is used to train 
        action_size: action size of UAV
        n_uavs: number of uav
        n_steps_test: total steps to test
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
    total_cells = (map_maximum_xy_axis // map_cell_size) **2
    Q_nets = [DQN(2*3+total_cells*2,action_size).to(device) for i in range(n_uavs)]   #    2 + total_cells**2 is equal the number of length [x_uav, y_uav, user heatmap, cloud heatmap]

    for i in range(n_uavs):
        path = f"{save_dir}/q_net_uav{i}_ep{episode_to_load}.pt"
        Q_nets[i].load_state_dict(torch.load(path, map_location = device))
        Q_nets[i].eval()

    # torch.manual_seed(1)
    # Some parameter used for testing only 
    hap_pos = torch.tensor([map_maximum_xy_axis/2, map_maximum_xy_axis/2,hap_z], dtype= torch.float32,device=device)
    cloud_map = torch_get_full_cloud()
    user_pos = simulate_user_mobility(n_users, n_steps_test)
    uav_pos = torch.randint(0,map_maximum_xy_axis, (n_uavs, 3),dtype=torch.float32, device=device )
    uav_pos[:,2] = uav_z

    region_0, cell_clwc_0 = get_cloudmap_t(0,cloud_map)
    user_pos_0 = user_pos[:,0,:]
    rewards0, alpha0, beta0, RUAV0, CFSO0 = user_association_and_calculate_reward(
        uav_pos, user_pos_0, region_0, hap_pos,device
    )
    obs_vec = build_observation_vector(
        uav_pos, user_pos_0, cell_clwc_0, alpha0, beta0
        ) # ([3,130]) # 3 stands for n_uavs, 130 stands for obs_vec length

    uav_rewards, RUAV_log, CFSO_log, uav_traj, alpha_log = [],[],[],[],[]
    uav_rewards.append(rewards0)
    RUAV_log.append(RUAV0)
    CFSO_log.append(CFSO0)
    uav_traj.append(uav_pos.cpu().numpy().copy())
    alpha_log.append(alpha0)
    
    for t in range(n_steps_test):
        with torch.no_grad():
            actions = [Q_nets[i](obs_vec[i]).argmax().item() for i in range(n_uavs)]

        uav_pos = move_uav(uav_pos, actions)
        next_t = min(t+1, n_steps_test -1)
        next_region_t, next_cell_clwc_t = get_cloudmap_t(next_t, cloud_map)
        next_user_pos_t = user_pos[:,next_t,:]
        rewards, alpha, beta, RUAV, CFSO = user_association_and_calculate_reward(
            uav_pos, next_user_pos_t, next_region_t, hap_pos, device
        )

        obs_vec = build_observation_vector(
            uav_pos, next_user_pos_t, next_cell_clwc_t,alpha, beta
            )
        uav_rewards.append(rewards)
        RUAV_log.append(RUAV)
        CFSO_log.append(CFSO)
        uav_traj.append(uav_pos.cpu().numpy().copy())
        alpha_log.append(alpha)
    return RUAV_log, CFSO_log, uav_traj, uav_rewards, cloud_map, hap_pos, user_pos, alpha_log

def draw_gif(t):

    # ==== Màu sắc đại diện ====
    uav_colors = ['blue', 'yellow', 'green']
    hap_color = 'red'

    # ==== Vẽ animation ====

    ax.clear()
    ax.set_xlim(0, map_maximum_xy_axis)
    ax.set_ylim(0, map_maximum_xy_axis)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.set_title(f"Time step {t}")
    shift = min(int(4 * t), cloud_map.shape[3] - map_maximum_xy_axis)
    region = cloud_map[0,0,:, shift:shift + map_maximum_xy_axis]
    discrete_cmap = plt.get_cmap('Blues', 7)
    
    img = ax.imshow(
    region,
    cmap=discrete_cmap,              # Hoặc thử 'bone', 'Blues'
    origin='lower',
    extent=[0, 1600, 0, 1600],
    alpha=0.5,
    vmin=0.5,
    vmax=7.5
    )

    # ==== Vẽ HAP ====
    ax.scatter(hap_pos[0], hap_pos[1], marker='s', color=hap_color, label='HAP')
    current_uav_pos = uav_traj[t]
    # ==== Vẽ UAVs và vùng phủ sóng ====
    for i, uav in enumerate(current_uav_pos):
        ax.scatter(uav[0], uav[1], marker='^', color=uav_colors[i], label=f"UAV {i+1}")
        circle = patches.Circle((uav[0], uav[1]), radius=uav_radius_coverage, fill=False, color=uav_colors[i], linestyle='--')
        ax.add_patch(circle)
        
    current_alpha = alpha_log[t]
    if torch.is_tensor(current_alpha):
        current_alpha = current_alpha.cpu().numpy()
    # ==== Vẽ người dùng ====
    for n in range(n_users):
        x, y= user_pos[n, t,:2]
        served = -1
        for i in range(len(current_uav_pos)):
            if current_alpha[n,i]:
                served = i
                break
        if served!= -1:
            user_color = uav_colors[i]
        else :
            user_color = 'gray'
        
        ax.scatter(x, y, marker='o', color=user_color, edgecolors='k', s=20)

    # ==== Gọn legend ====
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

def plot_training_rewards(episode_return, window_size =50):
    print("Đang vẽ biểu đồ quá trình huấn luyện (Training Rewards)...")
    returns_arr = []
    for r in episode_return:
        if torch.is_tensor(r):
            returns_arr.append(r.cpu().numpy())
        else:
            returns_arr.append(r)
    returns_arr = np.array(returns_arr, dtype=np.float32)
    n_episodes = returns_arr.shape[0]
    n_uavs = returns_arr.shape[1]
    
    # 2. Khởi tạo đồ thị
    plt.figure(figsize=(10, 6), dpi=120)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 3. Vẽ đường phần thưởng cho từng UAV
    for i in range(n_uavs):
        uav_rewards = returns_arr[:, i]
        
        # Nếu số episode đủ lớn, áp dụng Moving Average để làm mượt
        if n_episodes >= window_size:
            # Tính trung bình trượt
            weights = np.ones(window_size) / window_size
            smoothed_rewards = np.convolve(uav_rewards, weights, mode='valid')
            
            # Vẽ đường dữ liệu gốc (mờ đi để làm nền)
            plt.plot(uav_rewards, color=colors[i % len(colors)], alpha=0.25)
            # Vẽ đường đã làm mượt (đậm và nổi bật)
            plt.plot(range(window_size - 1, n_episodes), smoothed_rewards, 
                     label=f'UAV {i+1} (Avg {window_size})', 
                     color=colors[i % len(colors)], linewidth=2)
        else:
            # Nếu ít episode quá thì vẽ bình thường
            plt.plot(uav_rewards, label=f'UAV {i+1}', color=colors[i % len(colors)], linewidth=2)

    # 4. Trang trí theo chuẩn bài báo
    plt.title('Training Learning Curve (Episode Rewards)', fontsize=15, fontweight='bold')
    plt.xlabel('Training Episode', fontsize=13)
    plt.ylabel('Average Reward', fontsize=13)
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    # 5. Lưu và hiển thị
    save_dir = '/kaggle/working/results'
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/training_learning_curve.png', bbox_inches='tight')
    plt.show()
    

def plot_evaluation_logs(RUAV_log, CFSO_log, uav_rewards):
    """
    Vẽ 3 biểu đồ đánh giá: Phần thưởng, Dung lượng FSO (CFSO), và Tốc độ phục vụ User (RUAV)
    """
    print("Đang xử lý dữ liệu để vẽ biểu đồ...")
    
    # 1. Chuyển đổi dữ liệu về dạng NumPy mảng 2D: kích thước (n_steps_test, n_uavs)
    # Xử lý trường hợp dữ liệu bên trong vẫn là PyTorch Tensor
    def to_numpy_array(log_data):
        clean_log = []
        for step_data in log_data:
            if torch.is_tensor(step_data):
                clean_log.append(step_data.cpu().numpy())
            elif isinstance(step_data, list) and torch.is_tensor(step_data[0]):
                clean_log.append([x.cpu().item() for x in step_data])
            else:
                clean_log.append(step_data)
        return np.array(clean_log, dtype=np.float32)

    reward_arr = to_numpy_array(uav_rewards)
    cfso_arr = to_numpy_array(CFSO_log)
    ruav_arr = to_numpy_array(RUAV_log)

    n_steps_test = reward_arr.shape[0]
    n_uavs = reward_arr.shape[1]
    steps = np.arange(n_steps_test)

    # 2. Khởi tạo khung vẽ (3 biểu đồ xếp dọc)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), dpi=100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Bảng màu chuẩn

    # --- Biểu đồ 1: Phần thưởng (Rewards) ---
    for i in range(n_uavs):
        axs[0].plot(steps, reward_arr[:, i], label=f'UAV {i+1}', color=colors[i%len(colors)], linewidth=2)
    axs[0].set_title('UAV Rewards Over Time', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Reward')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(loc='upper right')

    # --- Biểu đồ 2: Dung lượng quang học (CFSO) ---
    for i in range(n_uavs):
        axs[1].plot(steps, cfso_arr[:, i], label=f'UAV {i+1}', color=colors[i%len(colors)], linewidth=2)
    axs[1].set_title('FSO Link Capacity', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('CFSO Capacity (bps/Hz)')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(loc='upper right')

    # --- Biểu đồ 3: Thông lượng người dùng (RUAV) ---
    for i in range(n_uavs):
        axs[2].plot(steps, ruav_arr[:, i], label=f'UAV {i+1}', color=colors[i%len(colors)], linewidth=2)
    axs[2].set_title('Total Access Link Data Rate', fontsize=14, fontweight='bold')
    axs[2].set_xlabel('Time Step (s)', fontsize=12)
    axs[2].set_ylabel('RUAV Data Rate (bps/Hz)')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(loc='upper right')

    # 3. Chỉnh khoảng cách và hiển thị
    plt.tight_layout()
    
    # Nếu chạy trên Kaggle, lưu ảnh ra file trước khi show để phòng bị mất
    plt.savefig('/kaggle/working/results/evaluation_metrics.png', bbox_inches='tight')
    plt.show()

def plot_ruav_vs_cfso(RUAV_log, CFSO_log):
    print("Đang vẽ biểu đồ chồng RUAV lên CFSO...")
    
    # 1. Hàm chuyển đổi dữ liệu an toàn sang NumPy
    def to_numpy_array(log_data):
        clean_log = []
        for step_data in log_data:
            if torch.is_tensor(step_data):
                clean_log.append(step_data.cpu().numpy())
            elif isinstance(step_data, list) and torch.is_tensor(step_data[0]):
                clean_log.append([x.cpu().item() for x in step_data])
            else:
                clean_log.append(step_data)
        return np.array(clean_log, dtype=np.float32)

    cfso_arr = to_numpy_array(CFSO_log)
    ruav_arr = to_numpy_array(RUAV_log)

    n_steps_test = cfso_arr.shape[0]
    n_uavs = cfso_arr.shape[1]
    steps = np.arange(n_steps_test)

    # 2. Khởi tạo khung vẽ (n_uavs biểu đồ xếp dọc)
    fig, axs = plt.subplots(n_uavs, 1, figsize=(12, 4 * n_uavs), dpi=100, sharex=True)
    
    # Ép kiểu list nếu n_uavs = 1 để tránh lỗi khi duyệt vòng lặp
    if n_uavs == 1:
        axs = [axs]

    # 3. Vẽ chồng 2 đường cho từng UAV
    for i in range(n_uavs):
        # Đường CFSO (Trần) - Nét đứt (--), màu đỏ gạch
        axs[i].plot(steps, cfso_arr[:, i], linestyle='--', linewidth=2.5, 
                    color='#d62728', label='CFSO (Ceiling / Backhaul)')
        
        # Đường RUAV (Thực tế) - Nét liền (-), màu xanh lam
        axs[i].plot(steps, ruav_arr[:, i], linestyle='-', linewidth=2, 
                    color='#1f77b4', label='RUAV (Served Data Rate)')
        
        # TÔ MÀU KHOẢNG TRỐNG: Giúp nhìn rõ khoảng cách giữa Trần và Thực tế
        axs[i].fill_between(steps, ruav_arr[:, i], cfso_arr[:, i], 
                            color='gray', alpha=0.15, label='Wasted Capacity')

        # Trang trí biểu đồ
        axs[i].set_title(f'UAV {i+1}: CFSO vs RUAV', fontsize=14, fontweight='bold')
        axs[i].set_ylabel('Data Rate (bps/Hz)')
        axs[i].grid(True, linestyle=':', alpha=0.7)
        axs[i].legend(loc='upper right')

    # Đặt nhãn trục X ở biểu đồ dưới cùng
    axs[-1].set_xlabel('Time Step (s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/results/ruav_vs_cfso.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    episode_return = train()
    import os # Nhớ import os cho hàm makedirs
    # # ==== Lưu thành GIF ====
    save_dir = '/kaggle/working/results'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Bắt đầu vẽ biểu đồ phần thưởng")

    plot_training_rewards(episode_return, window_size=50) # Cửa sổ 50 episodes để làm mượt
    
    print("Bắt đầu đánh giá mô hình AI...")
    # BỎ COMMENT DÒNG NÀY ĐỂ KẾT NỐI VỚI BỘ NÃO AI
    RUAV_log, CFSO_log, uav_traj, uav_rewards, cloud_map_tensor, hap_pos_tensor, user_pos_tensor, alpha_log = evaluate_trained_policy()
    
    # Ép dữ liệu Tensor từ GPU/CPU về mảng Numpy để Matplotlib vẽ
    cloud_map = cloud_map_tensor.cpu().numpy()
    hap_pos = hap_pos_tensor.cpu().numpy()
    user_pos = user_pos_tensor.cpu().numpy()
    
    # Ghi chú: uav_traj đã là list của numpy arrays ở trong hàm evaluate_trained_policy rồi
    
    print("Đang tạo file Mp4 (vui lòng chờ)...")
    fig, ax = plt.subplots(figsize=(8, 8)) # n_cells có thể gây lỗi nếu chưa định nghĩa, dùng số cụ thể
    
    dummy_cmap = plt.get_cmap('Blues', 7)
    dummy_img = ax.imshow([[0,0],[0,0]], cmap=dummy_cmap, vmin=0.5, vmax=7.5, visible=False)
    cbar = fig.colorbar(dummy_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('CLWC Value (g/m³)')
    
    
    ani = animation.FuncAnimation(fig, draw_gif,  frames=range(0, n_steps_test, 5))
    
    name_gif = "trained_uav_policy"
    ani.save(f"{save_dir}/{name_gif}.mp4", writer='ffmpeg', fps=10)
    print(f"✅ Đã lưu Video: {save_dir}/{name_gif}.mp4")
    
    # BỎ COMMENT để vẽ các biểu đồ phân tích
    plot_evaluation_logs(RUAV_log, CFSO_log, uav_rewards)
    plot_ruav_vs_cfso(RUAV_log, CFSO_log)
    print("Hoàn tất đánh giá!")

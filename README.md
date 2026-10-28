# Mobility_Aware_for_AGIN
Mobility-aware for Air-Ground Integrated Network (AGIN). Phương pháp nhận thức tính di động cho mạng truyền thông tích hợp bầu trời - mặt đất.

# Reference github: 
Repository này có tham khảo code từ link: https://github.com/minhnguyen27823/SAGIN-problem-optimization. Ngoài ra, chúng tôi đã tinh chỉnh một vài hàm để tối ưu việc huấn luyện trên GPU (CUDA). This code is adapted from link: https://github.com/minhnguyen27823/SAGIN-problem-optimization. Moreover, we refine some functions for faster training on GPU (CUDA)

# Some oustanding information/ results 
1. Kiến trúc về AGIN (AGIN Architecture)
![Architecture](assets/AGIN%20architecture.png)

2. Biểu đồ phần thưởng 2000 episode (Episode Reward)
![Architecture](assets/training_learning_curve.png)

3. Đánh giá mô hình qua kịch bản test (Test phase)
![Architecture](assets/trained_uav_policy.mp4)
![Architecture](assets/evaluation_metrics.png)
![Architecture](assets/ruav_vs_cfso.png)

# Tệp train và đánh giá tại: Channel_Modeling\train_update_obs_vec.py



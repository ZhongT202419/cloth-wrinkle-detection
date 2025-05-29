# cloth-wrinkle-detection
Implementation of Real-Time Cloth Wrinkle Detection and Optimization via Physics-Informed CNNs and LSTM-Reinforcement Learning" submitted to The Visual Computer. Includes CNN-based wrinkle detection, LSTM-RL prediction, RGBD integration, and reinforcement-based optimization.

# Cloth Wrinkle Detection and Optimization using Physics-Informed CNNs and LSTM-RL

**Official implementation of**  
📄 Real-Time Cloth Wrinkle Detection and Optimization via Physics-Informed CNNs and LSTM-Reinforcement Learning  
Submitted to The Visual Computer (Springer)

---

## 🔍 Overview

This repository contains the full codebase and assets for a hybrid framework that combines computer vision, physics-based modeling, and reinforcement learning (RL) to enable real-time wrinkle detection, prediction, and correction across various fabrics using static RGBD imagery.

---

## 📦 Features

- CNN-based wrinkle feature extraction (VGG16/ResNet)
- Mask R-CNN for occlusion handling
- RGBD fusion for depth and geometry estimation
- LSTM-based sequential wrinkle prediction
- Physics-based mass-spring model
- RL optimization of stiffness and damping (using PPO/DQN)
- 📸 45,876 high-resolution garment images (cotton, silk, polyester, blends)
- 🧩 Annotated with wrinkle maps: location, width, depth (shallow/medium/deep)
- 🤖 RGBD-supported stereo formats
  
## 🔧 Installation

```bash
git clone https://github.com/ZhongT202419/cloth-wrinkle-detection-LSTM-RL.git
cd cloth-wrinkle-detection-LSTM-RL
pip install -r requirements.txt



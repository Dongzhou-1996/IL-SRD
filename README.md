# IL-SRD
Imitation learning-based spacecraft rendezvous and docking method with Expert Demonstration
# Requirements
1. python >= 3.8.0
2. torch >= 2.7.1+cuda12.8
3. tensorboard >= 2.20.0
4. gymnasium >= 1.1.1
5. h5py >= 3.14.0
## 🎥 Demo Video

The video demonstrates spacecraft rendezvous and docking maneuvers under different control methods.
We compare the proposed **IL-SRD** controller with **Model Predictive Control (MPC)**, which is also
used to generate the expert demonstrations for imitation learning. Other methods include **PID**, **SAC**, **D4PG** and **Vanilla BC** are also compared.

As shown in the video, IL-SRD achieves performance comparable to MPC in terms of rendezvous and
docking accuracy, while successfully reproducing the expert behavior learned from demonstrations.

https://github.com/user-attachments/assets/6848844e-0361-4ef7-b151-5a388fa10f5e
## 🚀 Training and Evaluation

### Code Structure
- `train.py`  
  Contains both **training** and **evaluation** code.
- `dataset_2/`  
  Directory containing the dataset.
- `model_set/`  
  Directory containing a **pre-trained model**.

---

### 🔧 Training
To train the model from scratch, simply run:
```bash
python train.py

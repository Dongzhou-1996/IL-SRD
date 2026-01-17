# IL-SRD
Imitation learning-based spacecraft rendezvous and docking method with Expert Demonstration
# Requirements
1. python >= 3.8.0
2. torch >= 2.7.1+cuda12.8
3. tensorboard >= 2.20.0
4. gymnasium >= 1.1.1
5. h5py >= 3.14.0
# Guidance
The plots are the demonstration of spacecraft rendezvous and docking error diagrams for different control methods. It can be seen that the proposed IL-SRD algorithm achieves comparable results with MPC, which is the same controller we used to generate expert demonstrations. 

https://github.com/user-attachments/assets/6848844e-0361-4ef7-b151-5a388fa10f5e

Both the train code and evaluation code are included in train.py, dataset is included in dataset_2. There is already a trained model in model_set, just change the directory in the eval code is able to directly run evaluation. 

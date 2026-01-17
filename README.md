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
<div align="center">
  <!-- 核心视频嵌入标签 -->
  <video 
    controls 
    width="80%"  # 自适应README宽度，也可设固定值如640
    height="auto" 
    muted        # GitHub要求静音才能自动播放
    autoplay     # 自动播放（需配合muted）
    loop         # 循环播放（适合演示视频）

https://github.com/user-attachments/assets/6848844e-0361-4ef7-b151-5a388fa10f5e


    preload="metadata"  # 仅预加载元数据，加快加载速度
  >
    <source 
      src="https://github.com/user-attachments/assets/6848844e-0361-4ef7-b151-5a388fa10f5" 
      type="video/mp4"  # 声明视频格式（GitHub拖拽上传默认是MP4）
    >
    您的浏览器不支持HTML5视频播放，请复制链接在新窗口打开：https://github.com/user-attachments/assets/a74a17e4-64e5-4703-8c25-3ed9f014a6aa
  </video>
</div>
Both the train code and evaluation code are included in train.py, dataset is included in dataset_2. There is already a trained model in model_set, just change the directory in the eval code is able to directly run evaluation. 

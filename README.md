# Recfm-Planner

## ⚠️ 注意事项（Important Notes）

- **ReFlowMatching** 为模型训练算法所在目录，请在使用前确保已正确配置 **PyTorch** 及相关深度学习运行环境。
- 模型训练完成后，请将生成的 **`latest.ckpt`** 模型文件移动至 **`SimDrone_WS/src/planner/recfm_planner/trained_models/latest.ckpt`**。
- 训练数据位于 **`ReFlowMatching/training_data/my_train_data.pkl`**（受 GitHub 文件大小限制，当前仅提供少量示例数据用于流程演示；如需将训练后的模型实际应用于规划算法中，请自行扩充训练数据，以获得更优的规划性能与泛化能力）。
- **SimDrone_WS** 为规划算法工程目录，已在 **Ubuntu 20.04 + ROS Noetic** 环境下完成编译与运行测试。

---

## 📥 下载与解压

```bash
git clone https://github.com/ZYY-gif-cloud/Recfm-Planner.git
cd Recfm-Planner
unzip ReFlowMatching.zip && rm ReFlowMatching.zip
unzip SimDrone_WS.zip && rm SimDrone_WS.zip
```

---

## 🧠 ReFlowMatching：环境配置与模型训练

```bash
cd ReFlowMatching
pip install -r requirements.txt
```

默认已正确安装 PyTorch、CUDA 及相关深度学习依赖，运行 **`CFM_train.py`** 脚本即可启动模型训练：
```bash
python CFM_train.py
```

---

## 🚁 SimDrone_WS：编译与运行

```bash
sudo apt-get install libarmadillo-dev
cd SimDrone_WS
catkin_make
source devel/setup.bash
roslaunch recfm_planner simple_run.launch
```

---

## 🙏 致谢（Acknowledgements）

本项目的规划算法框架基于 **Ego-Planner**，并在此基础上进行了功能扩展与适配。  
Ego-Planner 在四旋翼局部规划领域展现了优异的性能，为本工作的实现提供了重要参考。

---

## 📚 引用（BibTeX）

若本项目对你的研究或工程实践有所帮助，欢迎引用我们的相关工作，BibTeX 信息将于后续补充。



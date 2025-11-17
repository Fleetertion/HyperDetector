# HyperDetector

HyperDetector 是一个结合超图神经网络（HGNN）与块式自注意力（Block Self-Attention）的 APT（Advanced Persistent Threat）检测研究原型。项目中的脚本覆盖了 DARPA TC E3、ClearScope、SCVIC-APT-2021 等多源数据，提供了从原始日志解析、图构建、模型训练到告警生成的完整链路。

---

## 目录概览

```
.
├─dataset/                # 数据缓存（apt2021、darpatc、wget）
├─groundtruth/            # DARPA TC 标签（cadets/theia/trace/...）
├─models/                 # 训练过程中导出的模型与特征
├─scripts/                # 训练、推理及数据处理脚本
├─readme.md
└─requirements.txt
```

---

## 环境准备

1. **Python/Conda**
   ```bash
   conda create -n hyperdetector python=3.8
   conda activate hyperdetector
   ```
2. **依赖安装**
   ```bash
   pip install -r requirements.txt
   ```
   `torch-geometric` 相关依赖（`torch-scatter`、`torch-sparse` 等）需要与本机 CUDA/PyTorch 版本匹配，若安装失败请参考 [PyG 官方说明](https://pytorch-geometric.readthedocs.io/).
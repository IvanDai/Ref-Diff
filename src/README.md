# RFStyle Baseline

## 想做什么

这个项目尝试从受干扰的单载波复基带信号中，恢复更接近原始发射端的信号。

输入是带有噪声、频偏、相偏、时钟偏差和多径的接收信号 `y_rx`。模型以接收信号的频域作为条件，通过 1D diffusion 生成干净发射信号的频域，最后可以用 IFFT 还原时域波形。

```text
随机信息
  -> 调制和 PulseShaping
  -> 干净发射信号 x_tx
  -> 同步偏差、多径和噪声
  -> 接收信号 y_rx

训练目标：FFT(y_rx) -> diffusion -> FFT(x_tx)
```

当前只研究干扰剥离，不做调制识别。

## 数据生成目标

生成器按照 RML2018 论文公开的流程实现，计划覆盖以下内容：

- 24 种调制：ASK、PSK、APSK、QAM、AM、FM、GMSK 和 OQPSK；
- RRC PulseShaping；
- 时间偏移和符号率偏移；
- 载波相偏和频偏；
- Rayleigh 多径；
- AWGN，复基带采样点 `SNR` 范围为 `-20～30 dB`；
- 每段信号长度为 1024 个复数采样点。

论文没有公开足以逐样本复刻官方 HDF5 的全部参数和实现细节，因此这里称为 **RML2018-compatible generator**，不声称生成结果与官方数据完全相同。

## 每条数据包含什么

- `x_canonical`：固定 PulseShaping 参数的白模信号；
- `x_tx`：使用本样本 PulseShaping 参数的干净发射信号；
- `y_rx`：施加全部选定损伤后的接收信号；
- `channel_taps`：本样本的多径系数；
- manifest 元数据：调制、SNR、损伤 profile 和各项生成参数。

信号保存在 HDF5 分片中，索引保存在 `manifest.csv`。训练时可以按照 `profile`、调制类型、SNR 和时延扩展筛选，不需要复制或重新排列数据。

## 当前已经写好的部分

- 24 种调制的波形生成接口；
- RRC 成型、同步偏差、相偏、频偏、多径和 AWGN 接口；
- HDF5 分片和 manifest 生成逻辑；
- 可按配置筛选数据的 PyTorch Dataset；
- 频域条件 1D U-Net；
- Gaussian diffusion 训练加噪和 DDIM 采样；
- 训练、验证、checkpoint 和基础评估入口；
- 完整数据配置、小规模配置和基础测试代码。

## 当前状态

信号生成已拆分为独立的 `rfsig` 包，并完成波形、损伤、统一 SNR、复现性和 HDF5 小规模生成测试。诊断图位于 `rfsig/tests/results/`。模型训练和大规模数据生成仍需单独验证。

后续验证顺序应为：

1. 检查全部 24 种调制的时域、频域和功率；
2. 单独验证每种损伤的数值和物理含义；
3. 生成小规模 HDF5 数据并测试筛选；
4. 运行模型前向和一次反向传播；
5. 完成小数据过拟合测试；
6. 再扩大数据量和损伤范围。

## 项目结构

```text
configs/            数据生成和训练配置
rfsig/              独立的调制、损伤和信号仿真包
rfstyle/data/       HDF5 数据生成、读取和 rfsig 兼容入口
rfstyle/models/     条件 1D U-Net
rfstyle/training/   diffusion 过程
scripts/            生成、训练和评估入口
tests/              信号链、数据和模型测试
pyproject.toml      项目依赖与打包配置
```

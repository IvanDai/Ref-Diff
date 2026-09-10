# RID2026

`rid2026` 包含两个层次：

- 信号生成 core：生成理想信号、实际发射信号和受损接收信号；
- RID2026 数据集生成器：调用 core，流式写入单个 HDF5 文件。

该 package 拥有生成链路的全部可复用实现，不依赖下游训练或分类包。
数据集生成的配置和运行入口属于根目录下的编号实验。

## 信号定义

每个 `SimulationResult` 只有四个顶层部分：

1. `y_rx`：经过发射成形、同步偏差、载波偏差、多径和 AWGN 的接收 IQ；
2. `modulation`：调制类别；
3. `x_ref`：无信道、无接收机损伤、无干扰和无噪声的参考目标；
4. `parameters`：波形参数、损伤参数、`x_tx`、信源和 seed。

不同调制族的 `x_ref` 定义如下：

- ASK、PSK、APSK、QAM：符号映射后进行矩形脉冲成形；
- OQPSK：I/Q 符号进行矩形脉冲成形，并保留半符号 Q 路偏移；
- GMSK：Gaussian 成形是调制定义的一部分，因此保留固定 BT 的理想 GMSK；
- AM/FM：完成模拟调制但尚未经过信道和接收机损伤的理想复基带波形。

`x_tx` 是施加损伤前的实际发射波形。数字线性调制的 `x_tx` 使用 RRC，
所以 `x_ref` 与 `x_tx` 的区别不是噪声，而是参考矩形脉冲与实际发射成形。
所有返回的纯净波形都归一化为单位平均功率。

## 信号链与裁剪

损伤顺序遵循 arXiv:1712.04578 Figure 2：

```text
信源 -> 调制 -> 发射成形 -> timing/SRO interpolation
     -> phase/CFO mixer -> Rayleigh multipath convolution -> AWGN
```

core 先生成包含前后 guard symbols 的完整帧，在完整帧上施加损伤，最后从
`x_ref`、`x_tx` 和 `y_rx` 的相同位置裁出观测帧。不会把帧尾周期延拓到帧头。

`crop_signal(x, start, length)` 可以对一维信号进行无填充、无回绕的显式裁剪。
训练划分时，同一条长信号裁出的窗口必须落在同一个样本区间，
避免数据泄漏。

## 单样本调用

```python
import numpy as np

from rid2026 import ImpairmentParameters, RFSignalGenerator, SignalConfig

channel = np.array([1.0, 0.3j], dtype=np.complex64)
channel /= np.linalg.norm(channel)

generator = RFSignalGenerator(
    SignalConfig(
        frame_length=1024,
        samples_per_symbol=8,
        guard_symbols=32,
        rrc_alpha=0.25,
    ),
    seed=233,
)
impairments = ImpairmentParameters(
    esn0_db=10.0,
    timing_offset=0.5,          # samples
    symbol_rate_offset=1e-4,   # fractional rate
    phase_offset=0.2,          # radians
    carrier_offset=2e-4,       # cycles/sample
    delay_spread=1.0,          # samples
    channel_taps=channel,
)

result = generator.generate("QPSK", impairments, seed=1234)

print(result.y_rx.shape)
print(result.modulation)
print(result.x_ref.shape)
print(result.parameters.x_tx.shape)
print(result.parameters.waveform)
print(result.parameters.impairments)
```

给定同一个显式 `seed`、配置和参数会得到逐点一致的结果。大规模生成使用
`derive_seed(root_seed, modulation, esn0, example)`，使样本 seed
不依赖生成顺序。这些 seed 只用于生成过程，不写入数据集。

## RML2018 参数分布

RID2026 的默认完整配置位于
`experiments/00_generate_rid2026/configs/full.yaml`。论文中每个
样本独立初始化各随机变量，RID2026 使用相同分布：

| 量 | RID2026 配置或分布 | 代码单位 |
|---|---|---|
| RRC roll-off `alpha` | `U(0.1, 0.4)` | 无量纲 |
| timing offset `delta_t` | `U(0, 16)` | samples |
| symbol-rate offset `delta_fs` | `N(0, sigma_clk)` | fractional rate |
| carrier phase `theta_c` | `U(0, 2*pi)` | radians |
| carrier-frequency offset `delta_fc` | `N(0, sigma_clk)` | cycles/sample |
| multipath `H` | `sum_i delta(t - Rayleigh_i(tau))` | 离散复基带 taps |
| delay spread `tau` | `{0, 0.5, 1.0, 2.0}` | samples |
| noise level | `Es/N0 = -20, -18, ..., 30` | dB |

完整配置使用论文实验中的 `sigma_clk = 0.0001`。`delta_fs` 和 `delta_fc`
共享同一个标准差，但对每个样本独立抽取。

`esn0_db` 不是采样点 SNR。对于单位平均采样功率、每符号 `sps` 个采样的数字
信号，噪声功率按 `Es = sample_power * sps` 计算。模拟调制返回的
`samples_per_symbol=1`，对应使用等效采样功率比。

论文给出了 Rayleigh 路径时延分布，但没有在 Table I 中指定有限离散实现所需
的路径数、最大 tap 数和复系数相位。RID2026 明确采用：

- `channel_paths = 16`；
- `max_channel_taps = 32`；
- 连续 Rayleigh 时延四舍五入到采样 bin，超出范围的路径截到最后一个 bin；
- 每条路径使用独立 `U(0, 2*pi)` 相位；
- 合并同一 bin 后将信道归一化为单位能量。

这些选择是论文公式的离散复基带实现参数，不冒充论文未给出的数值。

## 数据集规模

完整 RID2026 与 RML2018.01A 的公开宏观规格对齐：

```text
24 modulation classes
26 Es/N0 levels
4096 examples / modulation / Es/N0
1024 complex samples / example
2,555,904 examples total
```

数据集只生成每个 `(modulation, Es/N0)` 下的 4096 条独立样本，
不划分 train、valid 或 test。下游训练代码按比例和 seed 进行可复现的
分层划分。

线性数字调制采用 `samples_per_symbol=8`，每条 1024 点记录对应 128 个符号；
前后各生成 32 个 guard symbols 后再裁剪。过采样率和 guard 是 RID2026 明确
记录的实现参数，不是 Table I 中的随机变量。

24 类为：OOK、4ASK、8ASK、BPSK、QPSK、8PSK、16PSK、32PSK、16APSK、
32APSK、64APSK、128APSK、16QAM、32QAM、64QAM、128QAM、256QAM、
AM-SSB-WC、AM-SSB-SC、AM-DSB-WC、AM-DSB-SC、FM、GMSK、OQPSK。

## 生成数据集

在仓库根目录运行小规模验证：

```bash
conda run -n rfsig python experiments/00_generate_rid2026/run.py \
  --config experiments/00_generate_rid2026/configs/smoke.yaml \
  --output datasets/rid2026/rid2026_smoke.h5
```

确认磁盘容量和生成时间后，再运行完整配置：

```bash
conda run -n rfsig python experiments/00_generate_rid2026/run.py \
  --config experiments/00_generate_rid2026/configs/full.yaml \
  --output datasets/rid2026/rid2026.h5
```

生成器按 batch 流式写入一个 HDF5 文件，不会在内存中保存
2,555,904 条 record。文件结构为：

```text
rid2026.h5
├── modulation       UTF-8   [M]
├── esn0_db          float32 [S]
├── y_rx             float32 [M, S, E, 2, 1024]
├── x_ref            float32 [M, S, E, 2, 1024]
└── parameters/
    ├── samples_per_symbol
    ├── rrc_alpha
    ├── timing_offset
    ├── symbol_rate_offset
    ├── phase_offset
    ├── carrier_offset
    ├── delay_spread
    ├── channel_path_count
    └── channel_taps
```

`modulation` 和 `esn0_db` 是信号数组前两个维度的坐标轴。每条信号与
调制类别、Es/N0 的对应关系为：

```text
y_rx[m, s, e]  <-> modulation[m], esn0_db[s], 条件内第 e 条样本
x_ref[m, s, e] <-> modulation[m], esn0_db[s], 条件内第 e 条样本
```

例如 `y_rx[3, 5, 12]` 的调制类别是 `modulation[3]`，Es/N0 是
`esn0_db[5]`，并与 `x_ref[3, 5, 12]` 逐样点配对。`parameters/`
中每个数组的前三维同样是 `[M, S, E]`，因此
`parameters/name[m, s, e]` 就是该对信号的参数。

如果下游将前三维展平，第 `index` 条数据的坐标可以还原为：

```python
m, remainder = divmod(index, S * E)
s, e = divmod(remainder, E)
modulation_name = modulation[m]
sample_esn0_db = esn0_db[s]
```

`M=24`、`S=26`、`E=4096`。两个 IQ 数组未压缩时约占 `39.0 GiB`；
另有约 `0.61 GiB` 的 channel taps 和少量标量参数。文件不保存
`x_tx`、split、seed、profile、attrs 或生成配置副本。数据划分由下游使用者完成。

## 测试与诊断

在仓库根目录运行：

```bash
conda run -n rfsig python -m pytest tests/00_rid2026_waveforms \
  tests/01_rid2026_impairments tests/02_rid2026_generation \
  tests/03_rid2026_package_boundaries
conda run -n rfsig python experiments/01_rid2026_diagnostics/run.py
```

诊断图和 `Es/N0` 实测结果写入
`experiments/01_rid2026_diagnostics/results/`。测试覆盖 24 类波形、
矩形 `x_ref`、GMSK 特例、复现性、Table I 分布、Rayleigh 路径时延、损伤
顺序、无周期回绕、`Es/N0` 数值和 HDF5 小规模生成。

论文依据：Tim O'Shea, Tamoghna Roy, T. Charles Clancy, "Over the Air Deep
Learning Based Radio Signal Classification," arXiv:1712.04578，尤其是 Table I、
Figure 1 和 Figure 2。

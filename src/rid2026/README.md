# rfsig

`rfsig` 是一个不依赖 `rfstyle` 的复基带信号仿真包。它负责生成干净波形、施加接收链损伤，以及返回可复现的完整仿真结果。

## 信号链

```text
随机信源 -> 调制 -> RRC/Gaussian 成形 -> timing/SRO
         -> 单位能量多径 -> 接收机 phase/CFO -> AWGN
```

`snr_db` 对所有数字和模拟信号使用同一定义：

```text
SNR = 10 log10(mean(|deterministic_rx|^2) / mean(|noise|^2))
```

这是复基带采样点 SNR。数字信号的 `Es/N0` 可用
`snr_db + 10*log10(samples_per_symbol)` 近似换算。

## RML 兼容范围

- BPSK、QPSK、8PSK、16QAM、64QAM 和 RRC 成形参考 RML2016 的 GNU Radio 生成流程。
- 32QAM 和 128QAM 使用常见 cross-QAM 几何。
- APSK 使用 DVB-S2/S2X 风格的环点数和代表性半径比。
- RML2018 未发布完整生成代码，因此 APSK、扩展 QAM 和模拟调制不是官方逐点复刻。
- 模拟信号使用低通随机过程代替 RML2016 中受版权约束的音频文件。

参考代码：<https://github.com/radioML/dataset>

## 基本用法

```python
import numpy as np

from rfsig import ImpairmentParameters, RFSignalGenerator, SignalConfig

config = SignalConfig(length=1024, samples_per_symbol=8)
channel = np.array([1.0, 0.3j], dtype=np.complex64)
channel /= np.linalg.norm(channel)
impairments = ImpairmentParameters(
    snr_db=10.0,
    timing_offset=0.5,
    symbol_rate_offset=1e-4,
    phase_offset=0.2,
    carrier_offset=2e-4,
    delay_spread=1.0,
    channel_taps=channel,
)

result = RFSignalGenerator(config, seed=233).generate("QPSK", impairments)
print(result.waveform.transmitted.shape, result.received.shape)
```

## 测试与诊断图

在 `src` 目录运行：

```bash
conda run -n rfsig pytest rfsig/tests
conda run -n rfsig python -m rfsig.tests.generate_diagnostics
```

诊断图写入 `rfsig/tests/results/`。

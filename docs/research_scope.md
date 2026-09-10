# RefDiff 第一阶段研究 Backbone

## 研究目标

第一阶段的目标很明确：

> 利用 RID2026 中成对保存的干净调制信号和完整受损信号，训练 RefDiff 将受损 IQ 信号恢复到对应的参考波形，并验证这种恢复是否能够改善 AMC 分类结果。

当前阶段只围绕下面这条链路展开：

```text
RID2026
   ↓
RefDiff
   ↓
Restored IQ
   ↓
AMC
```

先把这条链路做通，再考虑后续扩展。

---

# 一、RID2026

## 1. 数据集定位

RID2026 已经完成。

它沿用完整的 RML2018 信号生成流程，但除了保存最终经过干扰流程后的 IQ 信号外，还额外保存了矩形成形后的干净调制信号。

因此，每个样本都同时具有：

- 干净参考波形；
- 完整受损后的接收波形；
- 调制类别；
- 对应的生成参数。

可以记为：

\[
(x_{\mathrm{ref}}, y_{\mathrm{rx}}, m, \theta)
\]

其中：

- \(x_{\mathrm{ref}}\) 为矩形成形后的干净调制信号；
- \(y_{\mathrm{rx}}\) 为经过完整 RML2018 干扰流程后的信号；
- \(m\) 为调制类别；
- \(\theta\) 为信号生成过程中对应的参数记录。

## 2. RID2026 在本研究中的作用

RID2026 最重要的意义是给 RefDiff 提供一一对应的监督数据：

\[
y_{\mathrm{rx}}
\rightarrow
x_{\mathrm{ref}}
\]

也就是说，对于每一个复杂接收信号，我们都知道它在进入完整干扰流程之前是什么样子。

这一点是后续 RefDiff 能够进行 waveform restoration 的基础。

---

# 二、RefDiff

## 1. 模型定位

RefDiff 的名称包含两层含义：

- **Reference Restoration Diffusion**
- **Waveform Refinement Diffusion**

它的任务不是调制分类，而是波形恢复。

输入：

\[
y_{\mathrm{rx}}
\]

输出：

\[
\hat{x}_{\mathrm{ref}}
\]

目标是：

\[
\hat{x}_{\mathrm{ref}}
\approx
x_{\mathrm{ref}}
\]

也就是让模型从完整受损后的 IQ 信号中恢复出对应的参考波形。

## 2. RefDiff 要解决的问题

RML2018 的完整生成流程会同时引入多种影响，最终的接收 IQ 已经不再只是原始调制波形。

传统 AMC 直接在这个信号上做分类，相当于让分类模型自己适应所有这些变化。

RefDiff 先做一步恢复：

\[
y_{\mathrm{rx}}
\rightarrow
\hat{x}_{\mathrm{ref}}
\]

希望尽量把这些额外影响从波形中压掉，同时保留原始调制结构。

第一阶段不要求 RefDiff 判断具体是哪一种损伤，也不要求它输出 CFO、相位、信道等参数。

它只需要完成一个任务：

> **把受损信号恢复得尽可能接近 RID2026 中对应的干净参考信号。**

---

# 三、AMC

## 1. AMC 的定位

AMC 在这一阶段主要作为 RefDiff 的下游验证模型。

它负责：

\[
x
\rightarrow
m
\]

也就是根据 IQ 信号判断调制方式。

## 2. AMC 要解决的问题

只看波形误差还不能完全说明 RefDiff 是否真的保留了调制相关信息。

因此需要把 RefDiff 的输出送入 AMC，进一步验证：

> RefDiff 恢复后的波形，是否比原始受损波形更容易被正确识别。

AMC 本身不需要成为这一阶段的主要创新点。

只要选择一个结构稳定、结果可靠的分类模型即可。

---

# 四、训练安排

RefDiff 和 AMC 分开训练。

## AMC

AMC 使用 RID2026 中的干净参考波形训练：

\[
x_{\mathrm{ref}}
\rightarrow
m
\]

训练完成后固定参数。

这样可以先建立一个对参考波形有稳定识别能力的分类器。

## RefDiff

RefDiff 使用 RID2026 中成对的数据训练：

\[
(y_{\mathrm{rx}},x_{\mathrm{ref}})
\]

其中：

- \(y_{\mathrm{rx}}\) 作为条件输入；
- \(x_{\mathrm{ref}}\) 作为恢复目标。

RefDiff 不使用调制标签作为输入。

第一阶段也不和 AMC 联合训练。

这样可以保证后续的分类提升来自 RefDiff 对波形本身的恢复，而不是两个模型联合优化后的结果。

---

# 五、测试与判断

第一阶段只保留两个最核心的验证。

## 1. 波形恢复

使用 NMSE 比较 RefDiff 输出和真实参考波形之间的误差：

\[
NMSE(\hat{x}_{\mathrm{ref}},x_{\mathrm{ref}})
\]

同时与原始受损信号相比：

\[
NMSE(y_{\mathrm{rx}},x_{\mathrm{ref}})
\]

第一阶段首先要证明：

\[
NMSE(\hat{x}_{\mathrm{ref}},x_{\mathrm{ref}})
<
NMSE(y_{\mathrm{rx}},x_{\mathrm{ref}})
\]

说明 RefDiff 的确把信号往参考波形方向恢复了。

## 2. AMC 分类提升

使用同一个已经训练并固定的 AMC，分别测试：

\[
y_{\mathrm{rx}}
\rightarrow
AMC
\]

以及：

\[
y_{\mathrm{rx}}
\rightarrow
RefDiff
\rightarrow
AMC
\]

比较两者分类准确率。

需要证明：

\[
Acc_{\mathrm{RefDiff+AMC}}
>
Acc_{\mathrm{Raw+AMC}}
\]

如果这一结果成立，就说明 RefDiff 的恢复不仅降低了波形误差，也确实恢复了对调制识别有用的信息。

---

# 六、当前阶段的要求

这一阶段只做基础链路，不扩展问题。

需要坚持以下几点：

1. **RID2026 直接作为现有数据基础，不重新简化信号生成流程。**
2. **RefDiff 只做 received-to-reference waveform restoration。**
3. **RefDiff 不输入 modulation label。**
4. **AMC 和 RefDiff 分开训练。**
5. **第一阶段只用 NMSE 判断波形恢复，用 Accuracy 判断 AMC 提升。**
6. **暂时不做参数估计、条件生成、OOD、SDR、端到端联合训练等扩展。**
7. **所有后续改动都建立在 RID2026 → RefDiff → AMC 这条主线上。**

---

# 七、第一阶段完成标准

第一阶段只需要确认两件事：

\[
\boxed{
RefDiff(y_{\mathrm{rx}})
\text{ 比 }
y_{\mathrm{rx}}
\text{ 更接近 }
x_{\mathrm{ref}}
}
\]

以及：

\[
\boxed{
AMC(RefDiff(y_{\mathrm{rx}}))
\text{ 的准确率高于 }
AMC(y_{\mathrm{rx}})
}
\]

如果这两点稳定成立，就说明第一阶段的基本思路可行。

之后再在这个基础上逐步讨论 RefDiff 的结构优化、训练方式改进、不同损伤的恢复能力以及更进一步的生成和反演问题。

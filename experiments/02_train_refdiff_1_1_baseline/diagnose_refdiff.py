from pathlib import Path
import sys

import torch


# ============================================================
# 1. 项目路径
# ============================================================

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "Ref-Diff")
sys.path.insert(0, str(ROOT / "src"))

from refdiff import GaussianDiffusion
from refdiff_1_1 import ConditionalUNet1D, RID2026TimeDomainPairs


# ============================================================
# 2. 只需要修改这里！！！
#
# 改成你自己的 checkpoint 路径
# ============================================================

CHECKPOINT = Path(
    "experiments/02_train_refdiff_1_1_baseline/"
    "outputs/20260911_105842/refdiff_1_1_last.pt"
)


# ============================================================
# 下面不用改
# ============================================================


def rms(x):
    """
    计算 IQ 信号的 RMS。
    正常 target 应该大约是 1。
    """
    return (
        x.square()
        .sum(dim=1)
        .mean()
        .sqrt()
        .item()
    )


def nmse(estimate, target):
    return (
        (estimate - target).square().sum()
        / target.square().sum()
    ).item()


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("device =", device)

    # --------------------------------------------------------
    # 加载 checkpoint
    # --------------------------------------------------------

    checkpoint_path = ROOT / CHECKPOINT

    print("checkpoint =", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    print("checkpoint epoch =", checkpoint["epoch"])

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model_config = config["model"]

    model = ConditionalUNet1D(
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(
            model_config["channel_multipliers"]
        ),
    ).to(device)

    # 使用 EMA 模型
    model.load_state_dict(checkpoint["ema_model"])

    model.eval()

    # --------------------------------------------------------
    # 创建 diffusion
    # --------------------------------------------------------

    diffusion = GaussianDiffusion(
        **config["diffusion"]
    ).to(device)

    # --------------------------------------------------------
    # 加载 validation 数据
    # --------------------------------------------------------

    data_config = config["dataset"]

    dataset_path = Path(data_config["path"])

    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    valid_set = RID2026TimeDomainPairs(
        dataset_path,
        "valid",
        data_config["split_counts"],
        modulations=data_config.get("modulations"),
        esn0_values=data_config.get("esn0_values"),
    )

    # 为了诊断快一点，只取前 4 个样本
    samples = [valid_set[i] for i in range(4)]

    condition = torch.stack(
        [x["condition"] for x in samples]
    ).to(device)

    target = torch.stack(
        [x["target"] for x in samples]
    ).to(device)

    print()
    print("==============================")
    print("输入数据")
    print("==============================")

    print(f"condition RMS = {rms(condition):.6f}")
    print(f"target RMS    = {rms(target):.6f}")
    print(f"raw NMSE      = {nmse(condition, target):.6f}")

    # --------------------------------------------------------
    # 从随机噪声开始
    # --------------------------------------------------------

    torch.manual_seed(123)

    sample = torch.randn_like(condition)

    print()
    print("==============================")
    print("开始 DDIM")
    print("==============================")

    print(f"initial noise RMS = {rms(sample):.6f}")

    steps = 50

    schedule = torch.linspace(
        diffusion.timesteps - 2,
        0,
        steps,
        device=device,
    ).round().long()

    schedule = torch.unique_consecutive(schedule)

    # --------------------------------------------------------
    # 手动运行 DDIM，观察每一步的数值
    # --------------------------------------------------------

    with torch.no_grad():

        for index, scalar_timestep in enumerate(schedule):

            t = int(scalar_timestep.item())

            timestep = torch.full(
                (condition.shape[0],),
                t,
                dtype=torch.long,
                device=device,
            )

            predicted_noise = model(
                sample,
                condition,
                timestep,
            )

            clean = diffusion.predict_clean(
                sample,
                predicted_noise,
                timestep,
            )

            # 前几步全部打印，
            # 后面每 5 步打印一次
            if index < 6 or index % 5 == 0:

                print(
                    f"step={index:02d} "
                    f"t={t:03d} | "
                    f"sample RMS={rms(sample):10.4f} | "
                    f"pred noise RMS={rms(predicted_noise):10.4f} | "
                    f"clean RMS={rms(clean):10.4f}"
                )

            if index == len(schedule) - 1:
                sample = clean
                continue

            next_timestep = torch.full_like(
                timestep,
                int(schedule[index + 1].item()),
            )

            alpha_next = diffusion._extract(
                diffusion.alpha_bar,
                next_timestep,
                sample.ndim,
            )

            sample = (
                alpha_next.sqrt() * clean
                + (1 - alpha_next).sqrt()
                * predicted_noise
            )

    restored = sample

    print()
    print("==============================")
    print("最终结果")
    print("==============================")

    print(f"target RMS   = {rms(target):.6f}")
    print(f"restored RMS = {rms(restored):.6f}")
    print(f"raw NMSE     = {nmse(condition, target):.6f}")
    print(f"restored NMSE= {nmse(restored, target):.6f}")


if __name__ == "__main__":
    main()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the accompanying recorded test data; this does not rerun tests."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN = "#2563a6", "#cb641b", "#258467"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "savefig.dpi": 160,
    }
)


def finish(fig, filename, title, subtitle):
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, subtitle, ha="center", fontsize=11, color="#4b5563")
    for ax in fig.axes:
        ax.set_box_aspect(1)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.85), w_pad=3)
    fig.savefig(ROOT / filename, facecolor="white")
    plt.close(fig)


def bridge_numerics():
    data = json.loads((ROOT / "bridge-data.json").read_text())
    prompts = data["bf16"]["prompts"]
    assert [p["scored_tokens"] for p in prompts] == [16, 33, 64]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    ax = axs[0]
    x = np.arange(3)
    for offset, key, color, label in [
        (-0.18, "all_vocab_max_base", BLUE, "Base"),
        (0.18, "all_vocab_max_adapter", ORANGE, "Nonzero LoRA"),
    ]:
        values = [p[key] * 1e4 for p in prompts]
        bars = ax.bar(x + offset, values, 0.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=[p["scored_tokens"] for p in prompts],
        xlabel="Scored positions per prompt",
        ylabel="Max absolute logprob error (×10⁻⁴ nats)",
        ylim=(0, 7.2),
        title="A  Maximum logprob error",
    )
    ax.legend(fontsize=10, loc="upper right")

    ax = axs[1]
    lim = max(
        abs(v)
        for p in prompts
        for key in ("native_target_adapter_delta", "vllm_target_adapter_delta")
        for v in p[key]
    ) * 1e4 * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="#9ca3af", label="Exact agreement")
    for p, color in zip(prompts, (BLUE, ORANGE, GREEN), strict=True):
        x = np.asarray(p["native_target_adapter_delta"]) * 1e4
        y = np.asarray(p["vllm_target_adapter_delta"]) * 1e4
        assert len(x) == len(y) == p["scored_tokens"]
        assert np.isclose(
            np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
            p["adapter_effect_cosine"],
        )
        ax.scatter(x, y, s=20, alpha=0.75, color=color, label=f"{p['scored_tokens']} positions")
    ax.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="Native: adapted − base (×10⁻⁴ nats)",
        ylabel="vLLM: adapted − base (×10⁻⁴ nats)",
        title="B  Nonzero LoRA effect",
        aspect="equal",
    )
    ax.legend(fontsize=10, loc="upper left")
    finish(
        fig,
        "bridge-numerics.png",
        "Inkling · tiny-model BF16 comparison with stock vLLM",
        "2 random-weight layers · hidden size 512 · 113 positions × 120 scored vocabulary entries",
    )


def native_qualification():
    data = json.loads((ROOT / "e2e-data.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    ax = axs[0]
    scatter = data["g1"]["update_scatter"]
    x, y = np.asarray(scatter["trainer_delta"]), np.asarray(scatter["sampler_delta"])
    assert x.shape == y.shape == (433,) and np.isfinite(x).all() and np.isfinite(y).all()
    lim = max(abs(x).max(), abs(y).max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "--", color="#9ca3af", label="Exact agreement")
    ax.scatter(x, y, s=13, alpha=0.55, color=BLUE, label="433 positions · 3 prompts")
    ax.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="Trainer logprob change (nats)",
        ylabel="Sampler logprob change (nats)",
        title="A  Update agreement",
        aspect="equal",
    )
    ax.legend(fontsize=10, loc="upper left")
    agreement = data["g1"]["updates"]["first_repeat"]
    ax.text(
        0.97,
        0.05,
        f"Cosine similarity: {agreement['cosine_similarity']:.4f}\nDiagnostic LR: 0.01",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color="#4b5563",
    )

    ax = axs[1]
    rows = data["tau10"]["evaluation"]
    assert [r["policy_step"] for r in rows] == [0, 5, 10]
    assert all(r["n"] == 100 for r in rows)
    x = np.arange(len(rows))
    for offset, values, color, label in [
        (-0.19, [r["mean_reward"] for r in rows], BLUE, "Mean reward"),
        (0.19, [r["strict_successes"] / r["n"] for r in rows], GREEN, "Strict success fraction"),
    ]:
        bars = ax.bar(x + offset, values, 0.38, color=color, label=label)
        labels = (
            [f"{v:.2f}" for v in values]
            if offset < 0
            else [f"{r['strict_successes']}/100" for r in rows]
        )
        ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=[r["policy_step"] for r in rows],
        xlabel="Policy step",
        ylim=(0, 1),
        ylabel="Reward / success fraction",
        title="B  Heldout evaluation",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.text(
        0.5,
        0.64,
        "One run · descriptive results\nSame 100 heldout tasks",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    finish(
        fig,
        "native-qualification.png",
        "Inkling-Small · full-model training",
        "276B total / 12B active parameters",
    )


if __name__ == "__main__":
    bridge_numerics()
    native_qualification()

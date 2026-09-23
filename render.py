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
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "savefig.dpi": 160,
    }
)


def finish(fig, filename, title, subtitle, footer):
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.939, subtitle, ha="center", fontsize=10, color="#4b5563")
    fig.text(0.5, 0.017, footer, ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0.01, 0.065, 0.99, 0.905), h_pad=2.5, w_pad=3)
    fig.savefig(ROOT / filename, facecolor="white")
    plt.close(fig)


def bridge_numerics():
    data = json.loads((ROOT / "bridge-data.json").read_text())
    prompts = data["bf16"]["prompts"]
    assert [p["scored_tokens"] for p in prompts] == [16, 33, 64]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    bound = (
        max(
            abs(v)
            for p in prompts
            for key in ("target_base_delta", "target_adapter_delta")
            for v in p[key]
        )
        * 1.25
    )
    for ax, key, title, color in [
        (
            axs[0, 0],
            "target_base_delta",
            "A  Base · target-token logprob difference",
            BLUE,
        ),
        (
            axs[0, 1],
            "target_adapter_delta",
            "B  Nonzero LoRA · target-token difference",
            ORANGE,
        ),
    ]:
        values = [v for p in prompts for v in p[key]]
        assert len(values) == 113 and np.isfinite(values).all()
        ax.scatter(np.arange(1, 114), values, color=color, s=15, alpha=0.8)
        ax.axhline(0, color="#9ca3af", lw=1)
        for end in (16.5, 49.5):
            ax.axvline(end, color="#9ca3af", ls=":")
        for center, text in [(8, "16"), (33, "33"), (82, "64 positions")]:
            ax.text(
                center, bound * 0.85, text, ha="center", fontsize=8, color="#4b5563"
            )
        ax.set(
            xlabel="Scored position (three prompts concatenated)",
            ylabel="Native − vLLM (nats)",
            title=title,
            ylim=(-bound, bound),
            xlim=(-1, 115),
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axs[1, 0]
    x = np.arange(3)
    for offset, key, color, label in [
        (-0.18, "all_vocab_max_base", BLUE, "Base"),
        (0.18, "all_vocab_max_adapter", ORANGE, "Nonzero LoRA"),
    ]:
        bars = ax.bar(
            x + offset, [p[key] for p in prompts], 0.36, color=color, label=label
        )
        ax.bar_label(
            bars, labels=[f"{p[key]:.2e}" for p in prompts], padding=4, fontsize=8
        )
    ax.set(
        xticks=x,
        xticklabels=[p["scored_tokens"] for p in prompts],
        xlabel="Scored positions per prompt",
        ylabel="Maximum absolute logprob difference (nats)",
        ylim=(0, 0.00072),
        title="C  All 120 vocabulary entries · maximum error",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=8, loc="upper right")

    ax = axs[1, 1]
    lim = (
        max(
            abs(v)
            for p in prompts
            for key in ("native_target_adapter_delta", "vllm_target_adapter_delta")
            for v in p[key]
        )
        * 1.1
    )
    ax.plot([-lim, lim], [-lim, lim], "--", color="#9ca3af")
    for p, color in zip(prompts, (BLUE, ORANGE, GREEN), strict=True):
        x, y = p["native_target_adapter_delta"], p["vllm_target_adapter_delta"]
        assert len(x) == len(y) == p["scored_tokens"]
        assert np.isclose(
            np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
            p["adapter_effect_cosine"],
        )
        ax.scatter(
            x,
            y,
            s=15,
            alpha=0.75,
            color=color,
            label=f"{p['scored_tokens']} positions · cosine {p['adapter_effect_cosine']:.3f}",
        )
    ax.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="Native: adapted − base (nats)",
        ylabel="vLLM: adapted − base (nats)",
        title="D  Nonzero adapter effect · target tokens",
    )
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=8, loc="upper left")
    finish(
        fig,
        "bridge-numerics.png",
        "Inkling · tiny-model BF16 numerical diagnostics",
        "Two random-weight layers, hidden size 512 · 113 positions × 120 vocabulary entries · native vs stock vLLM",
        "Dense-fusion candidate that became 7cd9a887; not an exact-commit GPU rerun.\n"
        "113 positions are from three prompts, not independent trials. Absolute errors are diagnostic; published-model parity remains separate.",
    )


def native_qualification():
    data = json.loads((ROOT / "e2e-data.json").read_text())
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axs[0, 0]
    scatter = data["g1"]["update_scatter"]
    x, y = np.asarray(scatter["trainer_delta"]), np.asarray(scatter["sampler_delta"])
    assert (
        x.shape == y.shape == (433,) and np.isfinite(x).all() and np.isfinite(y).all()
    )
    lim = max(abs(x).max(), abs(y).max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "--", color="#9ca3af", label="Exact agreement")
    ax.scatter(x, y, s=10, alpha=0.55, color=BLUE, label="433 token positions")
    ax.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="Trainer logprob change (nats)",
        ylabel="Sampler logprob change (nats)",
        title="A  Full-model update agreement · G1",
    )
    ax.legend(fontsize=8, loc="upper left")
    agreement = data["g1"]["updates"]["first_repeat"]
    ax.text(
        0.97,
        0.05,
        f"Cosine: {agreement['cosine_similarity']:.4f}\nDiagnostic update: LR 0.01",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#4b5563",
    )

    ax = axs[0, 1]
    ranks = data["g2"]["ranks"]
    x = np.arange(len(ranks))
    ax.bar(
        x - 0.19,
        [r["allocated_peak_gib"] for r in ranks],
        0.38,
        color=BLUE,
        label="Peak allocated",
    )
    ax.bar(
        x + 0.19,
        [r["reserved_peak_gib"] for r in ranks],
        0.38,
        color=GREEN,
        label="Peak reserved",
    )
    capacity = data["g2"]["device_capacity_gib"]
    ax.axhline(
        capacity, ls="--", color="#6b7280", label=f"Device capacity: {capacity:.1f} GiB"
    )
    ax.set(
        xticks=x,
        xticklabels=[r["rank"] for r in ranks],
        xlabel="Trainer rank",
        ylabel="CUDA allocator memory (GiB)",
        ylim=(0, capacity * 1.13),
        title="B  32,768-position capacity check · G2",
    )
    ax.legend(fontsize=8, loc="upper right")

    ax = axs[1, 0]
    for key, label, color in [
        ("tau2", "Tau2: 2/2 updates", BLUE),
        ("tau10", "Tau10: 10/10 updates", ORANGE),
    ]:
        rows = data[key]["updates"]
        assert all(
            r["gradient_norm"] > 0 and np.isfinite(r["gradient_norm"]) for r in rows
        )
        ax.plot(
            [r["optimizer_update"] for r in rows],
            [r["gradient_norm"] for r in rows],
            "o-",
            color=color,
            label=label,
            markersize=5,
        )
    ax.set(
        yscale="log",
        xticks=range(1, 11),
        xlim=(0.6, 10.4),
        xlabel="Acknowledged optimizer update",
        ylabel="Gradient norm (log scale)",
        title="C  Native training updates · separate runs",
    )
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1, 1]
    rows = data["tau10"]["evaluation"]
    assert [r["policy_step"] for r in rows] == [0, 5, 10]
    assert all(r["n"] == 100 for r in rows)
    x = np.arange(len(rows))
    for offset, values, color, label in [
        (-0.19, [r["mean_reward"] for r in rows], BLUE, "Mean reward"),
        (
            0.19,
            [r["strict_successes"] / r["n"] for r in rows],
            GREEN,
            "Strict success fraction",
        ),
    ]:
        bars = ax.bar(x + offset, values, 0.38, color=color, label=label)
        labels = (
            [f"{v:.3f}" for v in values]
            if offset < 0
            else [f"{r['strict_successes']}/100" for r in rows]
        )
        ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=[f"Policy step {r['policy_step']}" for r in rows],
        ylim=(0, 1),
        ylabel="Score / fraction",
        title="D  Tau10 heldout evaluation · same 100 tasks",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.5,
        0.63,
        "One run; descriptive result\nNo significance or full-MELT claim",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    finish(
        fig,
        "native-qualification.png",
        "Inkling-Small · recorded native qualification",
        "276B total / 12B active · 16 H200s · Bridge 7cd9a887 / SkyRL 99bf425d · integration evidence",
        "G1: 1057142   ·   G2: 1057146   ·   Tau2: 1057417   ·   Tau10: 1057648\n"
        "Recorded runs, not a rerun of cleanup head f2e39f10. Memory is allocator usage; full MELT sign-off remains separate.",
    )


if __name__ == "__main__":
    bridge_numerics()
    native_qualification()

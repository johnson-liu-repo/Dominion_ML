#!/usr/bin/env python
"""
Simulate the tabular Q-learning example from README section 2.4.1
("Example of Tabular Q-Learning in a Simple Deckbuilder") and plot the
Q-values over the update index k.

The environment has two non-terminal states (L = low hand value, H = high hand
value) and two actions (B = buy treasure card, S = buy score card). Rewards and
transitions follow Table 1 of the README. The agent replays the fixed
four-episode trajectory documented there, and each transition produces one
Q-value update

    Q_k(s,a) = (1 - alpha) Q_{k-1}(s,a) + alpha [ r + gamma max_a' Q_{k-1}(s',a') ] .

Examples:
  python scripts/simulate_tabular_q_learning.py
  python scripts/simulate_tabular_q_learning.py --alpha 0.5 --gamma 0.9 --out docs/tabular_q_learning.png
  python scripts/simulate_tabular_q_learning.py --q0 "L,B=0.5 H,S=1.0" --show
  python scripts/simulate_tabular_q_learning.py --q0 '{"L": {"B": 0, "S": 0}, "H": {"B": 0, "S": 0}}'
  python scripts/simulate_tabular_q_learning.py --repeat 15 --csv q_values.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

# --- Environment (README Table 1) ---------------------------------------------

STATES = ("L", "H")
ACTIONS = ("B", "S")
TERMINAL = "T"

PAIRS = tuple((s, a) for s in STATES for a in ACTIONS)

# (state, action) -> (reward, next_state)
DYNAMICS: dict[tuple[str, str], tuple[float, str]] = {
    ("L", "B"): (0.0, "H"),
    ("L", "S"): (1.0, "L"),
    ("H", "B"): (1.0, "H"),
    ("H", "S"): (3.0, TERMINAL),
}

# The trajectory the agent follows in the README: one list of (state, action)
# per episode, each episode starting in L and ending in the terminal state T.
EPISODES: tuple[tuple[tuple[str, str], ...], ...] = (
    (("L", "B"), ("H", "S")),
    (("L", "B"), ("H", "S")),
    (("L", "S"), ("L", "B"), ("H", "S")),
    (("L", "B"), ("H", "B"), ("H", "S")),
)

# --- Plot styling (light surface; see docs for the palette rationale) ---------

STATE_COLOR = {"L": "#2a78d6", "H": "#eb6834"}  # categorical slots 1 and 2
ACTION_STYLE = {"B": "-", "S": "--"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


@dataclass
class UpdateRecord:
    """One Q-learning update (one environment transition)."""

    k: int
    episode: int
    step: int
    state: str
    action: str
    reward: float
    next_state: str
    max_next: float
    target: float
    old_value: float
    new_value: float
    q_table: dict[tuple[str, str], float]

    @property
    def td_error(self) -> float:
        """Q_k = Q_{k-1} + alpha * (TD error), the equivalent form of the update."""
        return self.target - self.old_value


def _split_pair_key(key: str) -> tuple[str, str]:
    """Parse 'L,B', 'LB', 'L:B', '(L, B)', 'Q(L,B)' into ('L', 'B')."""
    letters = re.sub(r"[^A-Za-z]", "", key).upper()
    if letters.startswith("Q"):
        letters = letters[1:]
    if len(letters) != 2:
        raise ValueError(f"Cannot parse state-action key: {key!r}")
    state, action = letters[0], letters[1]
    if state not in STATES or action not in ACTIONS:
        raise ValueError(
            f"Unknown state-action pair {state},{action}. "
            f"States are {STATES} and actions are {ACTIONS}."
        )
    return state, action


def parse_initial_q(spec: str | None) -> dict[tuple[str, str], float]:
    """Build the initial Q-table from a CLI spec, defaulting every pair to 0.

    Accepts a JSON object (inline or a path to a .json file), either nested
    ``{"L": {"B": 0, "S": 0}}`` or flat ``{"L,B": 0}``, or a plain list of
    assignments such as ``"L,B=0.5 H,S=1.0"``. Pairs left out stay at 0.
    """
    q = {pair: 0.0 for pair in PAIRS}
    if not spec:
        return q

    text = spec.strip()
    path = Path(spec)
    if path.suffix.lower() == ".json" and path.exists():
        text = path.read_text(encoding="utf-8").strip()

    if text.startswith("{"):
        data = json.loads(text)
        for key, value in data.items():
            if isinstance(value, dict):
                for action, inner in value.items():
                    q[_split_pair_key(f"{key}{action}")] = float(inner)
            else:
                q[_split_pair_key(key)] = float(value)
        return q

    assignment = re.compile(
        r"(Q?\(?\s*[A-Za-z]\s*[,:]?\s*[A-Za-z]\s*\)?)\s*=\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    matches = list(assignment.finditer(text))
    if not matches:
        raise ValueError(
            f"Could not parse any assignments from {spec!r}. "
            "Expected something like \"L,B=0.5 H,S=1.0\" (quote the whole spec)."
        )
    # Anything outside the matches must be separators only.
    leftover = assignment.sub(" ", text)
    if re.sub(r"[,;\s]+", "", leftover):
        raise ValueError(
            f"Unrecognized text in the initial Q-value spec: {leftover.strip()!r}"
        )
    for match in matches:
        q[_split_pair_key(match.group(1))] = float(match.group(2))
    return q


def max_q(q: dict[tuple[str, str], float], state: str) -> float:
    """Best Q-value available in a state; the terminal state is worth 0."""
    if state == TERMINAL:
        return 0.0
    return max(q[(state, action)] for action in ACTIONS)


def run_simulation(
    initial_q: dict[tuple[str, str], float],
    alpha: float,
    gamma: float,
    episodes: tuple[tuple[tuple[str, str], ...], ...] = EPISODES,
    repeat: int = 1,
) -> list[UpdateRecord]:
    """Replay the trajectory, returning one record per update (k = 1, 2, ...)."""
    q = dict(initial_q)
    history: list[UpdateRecord] = []
    k = 0
    episode_number = 0

    for _ in range(repeat):
        for episode in episodes:
            episode_number += 1
            for step, (state, action) in enumerate(episode, start=1):
                reward, next_state = DYNAMICS[(state, action)]
                max_next = max_q(q, next_state)
                target = reward + gamma * max_next
                old_value = q[(state, action)]
                new_value = (1 - alpha) * old_value + alpha * target

                q[(state, action)] = new_value
                k += 1
                history.append(
                    UpdateRecord(
                        k=k,
                        episode=episode_number,
                        step=step,
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        max_next=max_next,
                        target=target,
                        old_value=old_value,
                        new_value=new_value,
                        q_table=dict(q),
                    )
                )

    return history


def series_over_k(
    initial_q: dict[tuple[str, str], float], history: list[UpdateRecord]
) -> tuple[list[int], dict[tuple[str, str], list[float]]]:
    """Q-value of every state-action pair at each k, including k = 0."""
    ks = [0] + [record.k for record in history]
    series = {
        pair: [initial_q[pair]] + [record.q_table[pair] for record in history]
        for pair in PAIRS
    }
    return ks, series


def print_trace(
    initial_q: dict[tuple[str, str], float],
    history: list[UpdateRecord],
    alpha: float,
    gamma: float,
) -> None:
    print(f"alpha = {alpha}, gamma = {gamma}")
    print("initial Q-table: " + "  ".join(f"Q({s},{a}) = {initial_q[(s, a)]:.5g}" for s, a in PAIRS))
    print()

    max_next_header = "max Q(s',a')"
    header = (
        f"{'k':>3}  {'ep':>2}  {'transition':<22}  {'r':>4}  "
        f"{max_next_header:>12}  {'target':>8}  {'Q_old':>8}  {'TD err':>8}  {'Q_new':>8}"
    )
    print(header)
    print("-" * len(header))

    for record in history:
        transition = (
            f"{record.state} --({record.action}, r={record.reward:g})--> {record.next_state}"
        )
        print(
            f"{record.k:>3}  {record.episode:>2}  {transition:<22}  {record.reward:>4.4g}  "
            f"{record.max_next:>12.5g}  {record.target:>8.5g}  "
            f"{record.old_value:>8.5g}  {record.td_error:>8.5g}  {record.new_value:>8.5g}"
        )

    print()
    print("final Q-table:")
    final = history[-1].q_table if history else initial_q
    print(f"{'state':>7} | {'B':>10} | {'S':>10}")
    print("-" * 33)
    for state in STATES:
        print(f"{state:>7} | {final[(state, 'B')]:>10.5g} | {final[(state, 'S')]:>10.5g}")


def write_csv(
    path: Path, ks: list[int], series: dict[tuple[str, str], list[float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"Q({s},{a})" for s, a in PAIRS]
    lines = ["k," + ",".join(columns)]
    for index, k in enumerate(ks):
        values = ",".join(f"{series[pair][index]:.10g}" for pair in PAIRS)
        lines.append(f"{k},{values}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions apart so end-of-line labels do not overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = list(values)
    for position, index in enumerate(order):
        if position == 0:
            continue
        previous = adjusted[order[position - 1]]
        if adjusted[index] - previous < min_gap:
            adjusted[index] = previous + min_gap
    return adjusted


def plot_history(
    ks: list[int],
    series: dict[tuple[str, str], list[float]],
    history: list[UpdateRecord],
    alpha: float,
    gamma: float,
    title: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    max_k = ks[-1]
    compact = max_k <= 24

    # Episode spans, drawn behind the data.
    if compact:
        spans: list[tuple[int, int, int]] = []  # (episode, first k - 1, last k)
        for record in history:
            if spans and spans[-1][0] == record.episode:
                spans[-1] = (record.episode, spans[-1][1], record.k)
            else:
                spans.append((record.episode, record.k - 1, record.k))
        for episode, start, end in spans:
            if start > 0:
                ax.axvline(start, color=GRIDLINE, linewidth=1.0, zorder=0)
            ax.text(
                (start + end) / 2,
                1.02,
                f"Episode {episode}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_MUTED,
            )

    # A trailing half-step so the value held after the last update stays visible.
    tail = max(0.5, 0.03 * max_k)
    for pair in PAIRS:
        state, action = pair
        values = series[pair]
        ax.step(
            ks + [max_k + tail],
            values + [values[-1]],
            where="post",
            color=STATE_COLOR[state],
            linestyle=ACTION_STYLE[action],
            linewidth=2.0,
            label=f"Q({state},{action})",
            zorder=2,
        )
        # Mark the updates that actually touched this pair.
        updated_k = [record.k for record in history if (record.state, record.action) == pair]
        if updated_k:
            ax.plot(
                updated_k,
                [values[ks.index(k)] for k in updated_k],
                linestyle="none",
                marker="o",
                markersize=5,
                markerfacecolor=STATE_COLOR[state],
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                zorder=3,
            )

    # Direct labels at the right edge, nudged apart when they crowd.
    finals = [series[pair][-1] for pair in PAIRS]
    y_low, y_high = ax.get_ylim()
    label_positions = _spread_labels(finals, 0.05 * (y_high - y_low))
    for pair, y_label, value in zip(PAIRS, label_positions, finals):
        state, action = pair
        ax.annotate(
            f"Q({state},{action}) = {value:.5g}",
            xy=(max_k + tail, y_label),
            fontsize=9,
            color=INK_SECONDARY,
            va="center",
            ha="left",
            annotation_clip=False,
        )

    ax.set_xlabel("update index $k$", color=INK_SECONDARY)
    ax.set_ylabel("Q-value", color=INK_SECONDARY)
    ax.set_title(
        title or f"Tabular Q-learning Q-values over updates  ($\\alpha$ = {alpha:g}, $\\gamma$ = {gamma:g})",
        color=INK_PRIMARY,
        fontsize=13,
        pad=28 if compact else 12,
    )

    if compact:
        ax.set_xticks(ks)
    ax.set_xlim(-0.2, max_k + tail)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)

    legend = ax.legend(
        loc="upper left", fontsize=9, facecolor=SURFACE, edgecolor="none", framealpha=0.9
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0, 0, 0.82, 1))  # reserve room for the direct labels
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the tabular Q-learning example from README 2.4.1 and plot "
            "the Q-values over the update index k."
        )
    )
    parser.add_argument(
        "--q0",
        "--initial-q",
        dest="initial_q",
        default=None,
        help=(
            "Initial Q-values. Assignments like \"L,B=0.5 H,S=1.0\", inline JSON, "
            "or a path to a .json file. Unlisted pairs default to 0."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate (default: 0.5).")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (default: 0.9).")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Replay the four-episode trajectory this many times (default: 1).",
    )
    parser.add_argument("--out", default=None, help="Optional output image path.")
    parser.add_argument("--csv", default=None, help="Optional CSV path for the Q-values per k.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument("--no-plot", action="store_true", help="Print the trace only.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    args = parser.parse_args()

    if not 0 < args.alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {args.alpha}.")
    if not 0 <= args.gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {args.gamma}.")
    if args.repeat < 1:
        raise ValueError(f"--repeat must be at least 1, got {args.repeat}.")

    initial_q = parse_initial_q(args.initial_q)
    history = run_simulation(initial_q, args.alpha, args.gamma, repeat=args.repeat)
    ks, series = series_over_k(initial_q, history)

    print_trace(initial_q, history, args.alpha, args.gamma)

    if args.csv:
        csv_path = Path(args.csv)
        write_csv(csv_path, ks, series)
        print(f"\nwrote {csv_path}")

    if args.no_plot:
        return 0

    fig = plot_history(ks, series, history, args.alpha, args.gamma, title=args.title)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"wrote {out_path}")

    if args.show or not args.out:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

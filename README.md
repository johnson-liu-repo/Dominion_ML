# Dominion ML — Reinforcement Learning for Dominion

Dominion ML is a work-in-progress reinforcement-learning project for training agents to play the deck-building game **Dominion**. The repository layers a Gymnasium-style buy-phase environment, a Dueling/Double DQN training loop, diagnostics tooling, and replay visualization on top of a bundled copy of [Pyminion](https://github.com/evanofslack/pyminion).

## Current status

The current codebase is an experimental buy-phase RL stack rather than a finished Dominion-playing agent.

- **Game engine:** the repository vendors Pyminion under `pyminion_master/` and uses it to run two-player Dominion games.
- **Supported RL environment:** `DominionBuyPhaseEnv` exposes only the RL agent's buy decision. Action, treasure, cleanup, and the opponent turn are handled internally between environment steps.
- **Player count:** training currently supports exactly one scripted opponent, so games are always two-player: the RL agent plus one opponent bot.
- **Card set:** the training entry point currently exposes `BASE_CARDS` as the configured card-set key.
- **Opponent bots:** the supported opponent names in the training config are `BigMoney` and `BigMoneyUltimate`.
- **Agent/trainer:** `src/agent_rl/train_dqn.py` implements a Dueling DQN with legal-action masking, replay buffer learning, target-network updates, checkpointing, resume/continuation support, and optional diagnostics.
- **Repository tests:** the checked-in tests cover environment card-count behavior and replay-state serialization/reconstruction.
- **Known limitations:** the project does not yet expose a full action-phase RL policy, does not yet evaluate against a broad benchmark suite, and does not currently include a checked-in `config/train_agent.json`; use `config/train_agent_example.txt` as the template/reference when creating one locally.

## Repository layout

```text
.
├── README.md
├── config/
│   └── train_agent_example.txt        # Documented JSON config examples
├── data/training/training_010/        # Checked-in sample diagnostics from an earlier run
├── docs/                              # Research notes, reports, and images
├── pyminion_master/                   # Vendored Pyminion engine and tests
├── replay_viewer/
│   └── dominion_replay_viewer.html    # Browser-based JSONL replay viewer
├── scripts/
│   ├── train_agent.py                 # Training CLI; reads a JSON config
│   ├── analyze_midgame_policy.py      # Policy-analysis helper
│   ├── generate_diagnostics_report.py # Builds text summaries from diagnostics CSVs
│   ├── plot_diagnostics.py            # Builds diagnostics plots
│   ├── plot_episode_metrics.py        # Plots reward/score-diff episode metrics
│   ├── plot_final_deck_card_trends.py # Analyzes deck-composition trends
│   └── reorganize_run_artifacts.py    # Moves run outputs into tiered folders
├── src/agent_rl/
│   ├── card_catalog.py                # Base-card action/observation catalog
│   ├── diagnostics.py                 # Training diagnostics collectors/rollups
│   ├── dominion_env.py                # Gymnasium buy-phase environment
│   ├── dominion_env_factory.py        # Environment factory
│   ├── dummie_bot.py                  # RL-agent bot shim used by the environment
│   ├── replay_state.py                # Replay-state snapshot/reconstruction helpers
│   ├── train_dqn.py                   # Dueling/Double DQN trainer
│   ├── training_io.py                 # Run directories, CSV/JSONL/checkpoint I/O
│   └── wrappers.py                    # Environment alias/future wrapper hook
├── tests/
│   ├── test_dominion_env_counts.py
│   └── test_replay_state.py
└── writeup.md                         # Project writeup/notes
```

## Environment and dependencies

There is no root-level dependency lockfile yet. The current Python code imports the following major packages:

- `gymnasium`
- `numpy`
- `torch`
- `pandas`
- `matplotlib`
- `pytest` for tests

A typical local setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install gymnasium numpy torch pandas matplotlib pytest
```

The vendored Pyminion package is imported directly from the repository (`pyminion_master`) by the project scripts and tests.

## Running tests

From the repository root:

```bash
python -m pytest tests
```

You can also run the vendored Pyminion test suite separately if you are changing Pyminion itself:

```bash
python -m pytest pyminion_master/tests
```

## Training workflow

The training CLI expects a JSON config file. By default it looks for `config/train_agent.json`, which is intentionally not checked in at the moment. Create it from the examples documented in `config/train_agent_example.txt`.

Minimal example:

```json
{
  "cards": "BASE_CARDS",
  "seed": 4991,
  "opponent_bots": ["BigMoney"],
  "output_dir": "data/training",
  "run_dir": null,
  "resume_from": null,
  "continue_training": {
    "enabled": false,
    "checkpoint_path": null,
    "output_dir": null,
    "run_dir": null,
    "inherit_optimizer_state": true,
    "inherit_epsilon_schedule": true
  },
  "training": {
    "episodes": 100,
    "turn_limit": 250,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon": 1.0,
    "eps_decay": 0.9995,
    "eps_min": 0.05,
    "target_update": 1000,
    "checkpoint_every": 200,
    "latest_every": 1,
    "save_turns": true,
    "save_turns_every": 100,
    "progress_bar": true,
    "enable_diagnostics": true
  }
}
```

Run training with:

```bash
python scripts/train_agent.py --config config/train_agent.json
```

### Resuming and continuing training

The trainer supports two checkpoint workflows:

1. **In-place resume** with the top-level `resume_from` field. This continues writing artifacts into the source run directory inferred from the checkpoint path.
2. **Forked continuation** with `continue_training.enabled: true`. This loads an old checkpoint but writes new artifacts to a new run directory, optionally inheriting optimizer state and/or epsilon schedule.

See `config/train_agent_example.txt` for complete examples of both workflows.

## Training artifacts

Runs are written under `data/training/training_XXX/` unless `run_dir` chooses a specific location. The writer creates or updates these artifacts:

| Path | Purpose |
| --- | --- |
| `episode_data_over_time.csv` | Per-episode reward, score, epsilon, loss, and timing summary. |
| `model_weights_over_time.csv` | Index of saved checkpoints by episode. |
| `final_decks.json` | End-of-episode deck snapshots. |
| `episodes/.../episode_XXXXXX_turns.jsonl` | Optional turn/replay logs, stored in tiered episode folders. |
| `checkpoints/checkpoint_latest.pt` | Latest checkpoint for resume. |
| `checkpoints/checkpoint_ep_*/checkpoint_ep_XXXXXX.pt` | Periodic checkpoints in tiered checkpoint folders. |
| `continuation_metadata.json` | Provenance metadata for forked continuation runs. |
| `diagnostics/*.csv` | Optional diagnostic CSVs when diagnostics are enabled. |

The repository currently includes sample diagnostics from `data/training/training_010/` but does not include full model checkpoints.

## Diagnostics and analysis

Enable diagnostics with:

```json
{
  "training": {
    "enable_diagnostics": true
  }
}
```

Then use the analysis scripts against a run's diagnostics folder:

```bash
python scripts/plot_diagnostics.py data/training/training_010/diagnostics
python scripts/generate_diagnostics_report.py data/training/training_010/diagnostics
```

The diagnostics pipeline can produce/consume:

- `episode_summary.csv`
- `buy_decisions.csv`
- `rolling_metrics.csv`
- `coin_bucket_summary.csv`
- `training_step_metrics.csv`
- generated plots under `results/plots/`
- a generated text report under `results/diagnostics_report.txt`

For short runs, treat diagnostics as sanity checks rather than evidence of stable learning.

## Replay viewer

`replay_viewer/dominion_replay_viewer.html` is a standalone browser viewer for turn JSONL logs written by training runs. Open it in a browser and load a generated `episode_XXXXXX_turns.jsonl` file from a run's `episodes/` tree.

## Implementation notes

### Environment step semantics

One RL `env.step(action)` corresponds to one buy decision by the RL agent. The environment then advances the game through cleanup and the opponent's turn until the next RL buy decision or terminal state. Observations combine supply counts, agent hand counts, agent owned-zone counts, opponent owned-zone counts, and scalar turn/economy/score features.

### Action legality

The action space has one action per configured card plus one pass action. Legal-action masking prevents the trainer from selecting or bootstrapping against impossible buys; pass is always legal.

### DQN design

The current neural-network baseline is a Dueling DQN MLP with:

- input `LayerNorm`
- three shared fully connected ReLU layers
- separate value and advantage heads
- epsilon-greedy exploration over legal actions only
- replay-buffer sampling
- Double DQN target computation
- periodic hard target-network syncs

## Near-term project direction

- Add a checked-in dependency specification for the root project.
- Add a checked-in starter JSON config or config-generation helper.
- Expand automated tests around trainer I/O, checkpoint resume, and diagnostics.
- Evaluate learned policies against scripted baselines over repeated seeded games.
- Extend beyond buy-phase-only control toward action/buy multi-phase agents.

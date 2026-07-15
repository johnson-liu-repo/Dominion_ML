<div align="justify">

*— Work in progress —*

# Reinforcement Learning: Deep Q-Learning for Dominion


### Contents



## Reinforcement Learning
Reinforcement learning is a subtype of machine learning where an agent learns to optimize its behavior in a given environment through learning by trial and error.
The agent learns from its experience what it should and shouldn't do within the state that it currently occupies.

### Historical Background

blah blah blah

### Sequential Decision Making

Given states $S_t$ at time $t$, future states $S_{t+1}$, the rewards $R_t$ for state transitions, and the actions $A_t$ that map $S_t$ into $S_{t+1}$, the reinforcement agent seeks to maximize its return

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\
G_t &= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\end{aligned}
$$

where $\gamma \in [0,1]$ is a discount factor for future returns.

===================================

===================================

### Markov Decision Processes

A Markov decision process (MDP) is a 4-tuple defined as

$$
(𝒮, 𝒜, 𝒫, ℛ)
$$

where $𝒮$ is the state space, $𝒜$ is the action space, $𝒫 = P(s' \mid s, a)$ is the probability of transitioning from state $s$ to $s'$ through action $a$, and $ℛ(s,a,s')$ is the reward for the transition from state $s$ to $s'$ through action $a$.
Generally the reward is written as

$$
r(s,a,s') = 𝔼\left[ R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \right]
$$

MDPs have the property where the probability of an event happening in the present $(s, a)$ is independent of the process's history:

$$
P \big( S_{t_1} \mid S_t, A_t \big) = P \big( S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots \big) \ .
$$

### Policies, Returns, and Value Functions

Following a deterministic policy, the agent's behavior is defined by

$$
a = \pi(s) \in 𝒜
$$

where $\pi(s)$ is a policy mapping each state $s$ to an action $a$.
Acting on a deterministic policy, the agent picks the action that gives it the greatest reward.
A stochastic policy $\pi(a \mid s)$ maps states to distributions over actions.

The state-value function

<b id="eq-quadratic"></b>

$$
V^\pi(s) = 𝔼\left[ G_t \mid S_t = s \right]  \qquad \text{(1)} \%\%\ MAGIT_PARSER_PROTECT%%
$$

gives the expected return for the agent if it starts in state $s$ and follows policy $\pi$ in all future steps.
The state-action value function

$$
Q^\pi(s,a) = 𝔼\left[ G_t \mid S_t = s, A_t = a\right ]
$$

gives the expected return for the agent if it starts in state $s$ and follows action $a$ and then policy $\pi$ forwards.
The optimal policy is one in which the action-value function is maximized:

$$
\pi_\*(s) \in \mathop{\rm argmax}_{a} \ q_\*(s,a)
$$

### Bellman Equations
The Bellman Expectation Equations are recursive equations used for policy evaluation of fixed policies.
Starting with the [state-value equation](#policies-returns-and-value-functions) and substituting in the definition of $G_t$, we have

$$
V^\pi(s) = 𝔼\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} │ S_t = s \right] \ .
$$

Separating the first term from the rest of the summation, factoring out a $\gamma$ from the remaining sum, and reindexing, we have

$$
V^\pi(s) = 𝔼\left[ \gamma^0 R_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} │ S_t = s\right ] \ .
$$

And applying linearity of expectation, this becomes

$$
V^\pi(s) = 𝔼\left[ \gamma^0 R_{t+1} │ S_t = s \right] + \gamma 𝔼\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} │ S_t = s \right]
$$

$$
V^\pi(s) = 𝔼\left[ R_{t+1} │ S_t = s \right] + \gamma 𝔼\left[ G_{t+1} │ S_t = s \right] \ .
$$

This equation says that the value of the current state under policy $\pi$ is the sum of the expected immediate reward of transitioning from state $s$ to $s'$ and the discounted expected future return $G_{t+1}$ starting from the next time step.
The expected immediate reward for being in the current state is computed as the summation over all possible actions and all possible successor states while multiplying together the probability that the agent will take action $a$ ($\pi(a|s)$) while in state $s$, the probability that the environment will transistion to state $s'$ ($P(s' | s, a)$), and the reward for transitioning to state $s'$ through action $a$ ($r(s, a, s')$):

$$
𝔼\left[ R_{t+1} │ S_t = s \right] = \sum_{a \in 𝒜} \pi(a | s) \sum_{s' \in 𝒮} P(s' | s, a) r(s, a, s') \ .
$$
The expected future return for being in state $s$ is found through summing over all actions and all states $s'$ and multiplying $\pi(a | s)$, $P(s' | s, a)$, and the state-value of the next state ($V^\pi (s')$):

$$
𝔼\left[ G_{t+1} │ S_t = s \right] = \sum_{a \in 𝒜} \pi(a | s) \sum_{s' \in 𝒮} P(s' | s, a) V^\pi(s') \ .
$$

Put together, we have

$$
V^\pi(s) = \sum_{a \in 𝒜} \left( \pi(a | s) \sum_{s' \in 𝒮} \left[ P(s,a,s') \left( r(s,a,s') + \gamma V^\pi(s') \right) \right] \right) \ .
$$

The $Q-value$ of a state-action pair is defined as the reward received for taking action $a$ in state $s$ plus the discounted expected value of being in the new state:

$$
Q(s, a) = r(s, a) + \gamma \sum_{s' \in 𝒮} P(s' | s, a) V^\pi (s') \ . \qquad \text{(2)} \%\%\ MAGIT_PARSER_PROTECT%%
$$

If the agent follows an optimal policy $\pi^*$, then the value of being in some state is equal to the return that the agent gets from taking the optimal action within that state corresponding to the largest possible $Q$-value out of all of the agent's options:

$$
V^\*(s) = \max_a Q^\*(s,a) \ .
$$

The [state-action value equation](#bellman-equations) is then

$$
Q^\*(s, a) = r(s, a) + \gamma \sum_{s' \in 𝒮} P(s' | s, a) \max_{a'} Q^\*(s',a') \ .
$$

Although the agent is taking its best possible action, there is some uncertainty in the state transition within the environment, expressed in the summation over $P(s' | s, a)$.
Typically, the agent does not know the environment's probability distribution for next state transitions.
Rather than summing over all possible actions allowed within the current state to compute the expected return, the agent uses the current value for $Q(s,a)$ and some policy, such as $\epsilon$-greedy, to decide how to act.
The agent then observes the immediate reward for its action and the value of the new state to compute a target value to update the $Q$-value of the state-action pair that lead to the new state.

### Temporal-Difference Learning

Because the agent does not know the probabilities of the state transitions when taking action $a$ in state $s$, $Q(s,a)$ cannot be directly computed.
Instead, the model uses Monte Carlo sampling and bootstrapping to iteratively update the $Q$-values of state-action pairs.
After initializing $Q$-values (usually to 0), the agent repeatedly samples the environment to make updates to the estimation of $Q(S,A)$.
Starting from some state and acting through policy $\pi$, the agent will visit the possible next states with frequencies that match the actual next state probability distribution.
Over many iterations, this sampling takes the place of the summation over probabilities in $Q(S,A)$ through averaging over many outcomes.
Adapting the formula for computing $Q(S,A)$ to use in temporal-difference updates, we have

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r(s,a,s') + \gamma\max_{a'}Q(s',a') - Q(s,a) \right]
$$

or

$$
Q(s,a) \leftarrow (1 - \alpha)Q(s,a) + \alpha\left[ r(s,a,s') + \gamma\max_{a'}Q(s',a') \right]
$$

where $Q(s,a)$ is the old estimated value of the state-action pair, $r(s,a,s') + \gamma\max_{a'}Q(s',a')$ is the target for the update, and $\alpha$ is the learning rate.
Gradually, $Q(S,A)$ converges onto the expectation in the Bellman optimality equation.

## Q-Learning

... some text here? ...

### Exploration and $ϵ$-Greedy Action Selection

The agent uses a model-free algorithm, it does not have an internal model of the environment's dynamics (the transition probabilities $P(s'|s,a)$ and reward function $R(s,a)$ for every given state-action pair).
Instead of explicitly learning the entirety of the environment's dynamics, it must learn its target policy by estimating state-action values through empirical sampling (trial and error).
The agent's ultimate goal is to find a target policy that, when deterministically followed, maximizes its total return.
But since the agent lacks a model to look ahead to find the optimal action path, it must randomly sample different actions to discover which trajectories yield the best empirical results.

This exploitation-exploration tradeoff during learning is possible because the agent's behavior policy (its strategy when interacting with the environment) is separate from its target policy (the strategy that the agent is actually trying to optimize).
In Q-learning, one behavior policy that an agent can have is $ϵ$-Greedy Action Selection, where the agent chooses to either exploit or explore the environment.
The agent has a 1-ϵ chance of exploiting the environment through action $a = \argmax_a Q(s,a)$ that yields the greatest expected future return and a $ϵ$ chance of uniformly picking any valid action at random.
This splits the agent's efforts between exploitation (performing the action that is currently predicted to lead to the best outcome) and exploration (taking actions that seem to yield suboptimal results in an attempt to find an action path that leads to rewards greater than those produced by the currently predicted best action).
The target policy uses the same Q-values that the behavior policy uses, but always acts in a greedy manner.

Through many episodes, the agent updates its Q-values for state-action pairs by acting suboptimally to gather data (exploration through its epsilon-greedy behavior policy) and evaluating that data assuming optimal future play (derived from the current estimated Q-values).


### Tabular Q-Learning

In tabular Q-learning, the Q-values for every possible state-action pair are stored in a lookup table.
Each row $r_i$ in the table corresponds to a specific state of the environment and each column $c_j$ represents a different action.
Each non-empty entry $Q_{ij}$ in the table holds the current Q-value (expected future return) for the state-action pair $(s_i, a_j)$.
When the agent decides to exploit the environment from its current state $s$, it reads the Q-table and chooses the action $a$ that has the largest Q-value for that state's row.
Otherwise, the agent explores, randomly sampling from among all valid actions within that state.
Once an action is executed, the agent transitions to a new state and observes the environment.
It receives its immediate reward, retrieves the current Q(s,a) value and the highest possible Q-value of the next state, computes the target value and new Q-value using the [temporal-difference formula](#temporal-difference-learning).
Since Q-learning is a Temporal-Difference method, the Q-table is immediately updated with this new Q-value.
The episode then continues on to the next step, continually updating the Q-table in this bootstrapped manner.

### Example of Tabular Q-Learning — A Simple Deckbuilder

This is an example of tabular Q-learning in an overly simplified deck builder.
The environment has two states (the value of the agent's hand): low (L) and high (H).
For both states, the possible actions are buying a treasure card (B) and buying s score card (S).
The rewards and transitions for each state-action pair is given in Table (1).

<div align="center"><b>Table 1:</b> Reward/Transition Table for a Simple Deckbuilder</div>

<div align="center">

|State (s)|Action (a)|Reward (r)|New State (s')|
| :---: | :---: | :---: | :---: |
|L <br> (low hand value)|B <br> (buy treasure card)|0|H|
|L|S <br> (buy score card)|1|L|
|H <br> (high hand value)|B|1|H|
|H|S|3|T <br> (terminal)|
</div>

#### Agent Training
The agent in this example follows the transitions shown below, always starting in state L and ending in state T.


1. [Episode 1]()\
    1.1 [Update 1](#ep1-update1): L --> H\
    1.2 [Update 2](): L $\xrightarrow{B, r = 0}$ H
2. [Episode 2]()\
    2.1 [Update 1](): L --> H\
    2.2 [Update 2](): H --> T
3. [Episode 3]()\
    3.1 [Update 1](): L --> L\
    3.2 [Update 2](): L --> H\
    3.3 [Update 3](): H --> T
4. [Episode 4]()\
    4.1 [Update 1](): L --> H\
    4.2 [Update 2](): H --> H\
    4.3 [Update 3](): H --> T




<a id="ep1-update1"></a>
##### Episode 1 — Update 1
In the first update of Episode 1, the agent transitions from state L to state H through action B and receives reward r = 0


<a id="ep1-update2"></a>
##### Episode 1 — Update 2



<a id="ep2-update1"></a>
##### Episode 2 — Update 1



<a id="ep2-update2"></a>
##### Episode 2 — Update 2



<a id="ep3-update1"></a>
##### Episode 3 — Update 1



<a id="ep3-update2"></a>
##### Episode 3 — Update 2



<a id="ep3-update3"></a>
##### Episode 3 — Update 3



<a id="ep4-update1"></a>
##### Episode 2 — Update 1



<a id="ep4-update2"></a>
##### Episode 4 — Update 2


---

... Create an animation/gif for the Q-table ...

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <!-- ROW 0 -->
  <tr style="border-bottom: 1px solid #ccc;">
    <!-- Text aligned to the bottom to match Row 1 -->
    <td rowspan="2" style="border-right: 4px solid black; font-weight: bold; vertical-align: bottom; padding: 12px;">State (s)</td>
    <!-- Cells 0,1 and 0,2 merged horizontally -->
    <td colspan="2" style="padding: 12px;">Action (a)</td>
  </tr>
  
  <!-- ROW 1 -->
  <!-- This row has the THICK black bottom border -->
  <tr style="border-bottom: 4px solid black;">
    <!-- Cell 1,0 is omitted here because 0,0 stretches down into it -->
    <td style="padding: 12px;">B</td>
    <td style="padding: 12px;">S</td>
  </tr>
  
  <!-- ROW 2 -->
  <tr style="border-bottom: 1px solid #ccc;">
    <td style="border-right: 4px solid black; padding: 12px;">L</td>
    <td style="padding: 12px;">0</td>
    <td style="padding: 12px;">0</td>
  </tr>

  <!-- ROW 3 -->
  <tr>
    <td style="border-right: 4px solid black; padding: 12px;">H</td>
    <td style="padding: 12px;">0</td>
    <td style="padding: 12px;">0</td>
  </tr>
</table>



### Strengths and Limitations of the Tabular Q-learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

---

## Deep Q-Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### Neural Networks

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### State Value and Action Value Approximation Using Neural Networks

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Deep Q-Learning for Dominion

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.



</div>



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

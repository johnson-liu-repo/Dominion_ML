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
r(s,a,s') = 𝔼\big[ R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \big]
$$

MDPs have the property where the probability of an event happening in the present $(s, a)$ is independent of the process's history:

$$
P \big(S_{t_1} \mid S_t, A_t \big) = P \big( S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots \big) \ .
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

$$
V^\pi(s) = 𝔼\big[G_t \mid S_t = s\big]
$$

gives the expected return for the agent if it starts in state $s$ and follows policy $\pi$ in all future steps.
The state-action value function

$$
Q^\pi(s,a) = 𝔼\big[G_t \mid S_t = s, A_t = a\big]
$$

gives the expected return for the agent if it starts in state $s$ and follows action $a$ and then policy $\pi$ forwards.
The optimal policy is one in which the action-value function is maximized:

$$
\pi_\*(s) \in \mathop{\rm argmax}_{a} \ q_\*(s,a)
$$

### Bellman Equations

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### Temporal-Difference Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Q-Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### Exploration and ϵ-Greedy Action Selection

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### Tabular Q-Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### Example of Tabular Q-Learning — A Simple Deckbuilder





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

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

### Exploration and ϵ-Greedy Action Selection


Using an $\epsilon$-greedy policy, the agent takes action $\text{argmax}_{a}Q*(s,a)$



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

"""
PPO-based Allocator Agent

Uses Proximal Policy Optimization for meta-level capital allocation.
"""

import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Dict, List
import chex


class AllocationNetwork(nn.Module):
    """
    Policy network for capital allocation.

    Outputs allocation probabilities over agents [π_HFT, π_MFT, π_LFT].
    """

    hidden_dims: Tuple[int, ...] = (128, 64)
    num_agents: int = 3

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass through allocation network.

        Args:
            x: Feature vector (agent performances, latent factors, macro indicators)

        Returns:
            allocation: Capital allocation probabilities [num_agents]
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Output allocation logits
        logits = nn.Dense(self.num_agents)(x)

        # Softmax to get allocation probabilities
        allocation = nn.softmax(logits)

        return allocation


class ValueNetwork(nn.Module):
    """
    Value network for state value estimation.

    Estimates V(s) for advantage calculation.
    """

    hidden_dims: Tuple[int, ...] = (128, 64)

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass through value network.

        Args:
            x: Feature vector

        Returns:
            value: State value estimate
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        value = nn.Dense(1)(x)

        return jnp.squeeze(value, axis=-1)


class PPOAllocatorAgent:
    """
    PPO-based Allocator Agent for meta-level capital allocation.

    Learns to allocate capital across HFT/MFT/LFT agents to:
    - Maximize long-term wealth
    - Maximize Sharpe ratio
    - Minimize CVaR
    """

    def __init__(
        self,
        feature_dim: int,
        num_agents: int = 3,
        policy_hidden: Tuple[int, ...] = (128, 64),
        value_hidden: Tuple[int, ...] = (128, 64),
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 42
    ):
        """
        Initialize PPO allocator agent.

        Args:
            feature_dim: Dimension of input features
            num_agents: Number of sub-agents to allocate to
            policy_hidden: Hidden layer sizes for policy network
            value_hidden: Hidden layer sizes for value network
            policy_lr: Learning rate for policy network
            value_lr: Learning rate for value network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Initialize random key
        self.key = jax.random.PRNGKey(seed)

        # Initialize networks
        self.key, policy_key, value_key = jax.random.split(self.key, 3)

        # Policy network
        self.policy_network = AllocationNetwork(
            hidden_dims=policy_hidden,
            num_agents=num_agents
        )

        dummy_input = jnp.zeros((1, feature_dim))
        policy_params = self.policy_network.init(policy_key, dummy_input)

        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(policy_lr)
        )

        self.policy_state = train_state.TrainState.create(
            apply_fn=self.policy_network.apply,
            params=policy_params,
            tx=policy_optimizer
        )

        # Value network
        self.value_network = ValueNetwork(
            hidden_dims=value_hidden
        )

        value_params = self.value_network.init(value_key, dummy_input)

        value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(value_lr)
        )

        self.value_state = train_state.TrainState.create(
            apply_fn=self.value_network.apply,
            params=value_params,
            tx=value_optimizer
        )

        # Training statistics
        self.train_steps = 0

        # Trajectory buffer
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []

    def select_action(
        self,
        state: chex.Array,
        deterministic: bool = False
    ) -> Tuple[chex.Array, float, float]:
        """
        Select allocation using current policy.

        Args:
            state: Current state (features)
            deterministic: If True, use argmax allocation

        Returns:
            allocation: Allocation probabilities [num_agents]
            log_prob: Log probability of selected allocation
            value: State value estimate
        """
        if state.ndim == 1:
            state = state[None, :]

        # Get allocation probabilities
        allocation_probs = self.policy_state.apply_fn(
            self.policy_state.params,
            state
        )[0]

        if deterministic:
            # Use argmax for deterministic policy
            best_agent = int(jnp.argmax(allocation_probs))  # Convert to Python int
            allocation = jnp.zeros(self.num_agents)
            allocation = allocation.at[best_agent].set(1.0)
        else:
            # Sample from categorical distribution
            self.key, subkey = jax.random.split(self.key)
            allocation = jax.random.categorical(subkey, jnp.log(allocation_probs))

            # Convert to one-hot (or use continuous allocation directly)
            # For now, use the probabilities as continuous allocation
            allocation = allocation_probs

        # Calculate log probability
        log_prob = jnp.sum(allocation * jnp.log(allocation_probs + 1e-8))

        # Get value estimate
        value = self.value_state.apply_fn(
            self.value_state.params,
            state
        )[0]

        return allocation, float(log_prob), float(value)

    def store_transition(
        self,
        state: chex.Array,
        action: chex.Array,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store transition in trajectory buffer."""
        self.states_buffer.append(state)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)
        self.dones_buffer.append(done)
        self.log_probs_buffer.append(log_prob)
        self.values_buffer.append(value)

    def compute_gae(
        self,
        rewards: chex.Array,
        values: chex.Array,
        dones: chex.Array,
        next_value: float
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value of next state

        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        T = len(rewards)
        advantages = jnp.zeros(T)
        returns = jnp.zeros(T)

        gae = 0.0
        next_value_temp = next_value

        for t in reversed(range(T)):
            if dones[t]:
                next_value_temp = 0.0
                gae = 0.0

            delta = rewards[t] + self.gamma * next_value_temp - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae

            advantages = advantages.at[t].set(gae)
            returns = returns.at[t].set(gae + values[t])

            next_value_temp = values[t]

        return advantages, returns

    @staticmethod
    @jax.jit
    def _ppo_loss(
        policy_params: flax.core.FrozenDict,
        value_params: flax.core.FrozenDict,
        policy_apply_fn,
        value_apply_fn,
        states: chex.Array,
        actions: chex.Array,
        old_log_probs: chex.Array,
        advantages: chex.Array,
        returns: chex.Array,
        clip_epsilon: float,
        entropy_coef: float,
        value_coef: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute PPO loss.

        Args:
            policy_params: Policy network parameters
            value_params: Value network parameters
            states, actions, old_log_probs, advantages, returns: Batch data
            clip_epsilon, entropy_coef, value_coef: PPO hyperparameters

        Returns:
            total_loss: Combined loss
            info: Dictionary of loss components
        """
        # Policy loss
        allocation_probs = policy_apply_fn(policy_params, states)

        # Compute log probabilities of actions
        log_probs = jnp.sum(actions * jnp.log(allocation_probs + 1e-8), axis=-1)

        # Ratio for PPO
        ratio = jnp.exp(log_probs - old_log_probs)

        # Surrogate losses
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Entropy bonus (encourage exploration)
        entropy = -jnp.mean(jnp.sum(allocation_probs * jnp.log(allocation_probs + 1e-8), axis=-1))
        entropy_loss = -entropy_coef * entropy

        # Value loss
        values = value_apply_fn(value_params, states)
        value_loss = jnp.mean((returns - values) ** 2)

        # Total loss
        total_loss = policy_loss + entropy_loss + value_coef * value_loss

        info = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((log_probs - old_log_probs) ** 2)
        }

        return total_loss, info

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update using collected trajectories.

        Args:
            next_value: Value of next state (for bootstrapping)

        Returns:
            info: Dictionary of training metrics
        """
        if len(self.states_buffer) == 0:
            return {}

        # Convert buffers to arrays
        states = jnp.array(self.states_buffer)
        actions = jnp.array(self.actions_buffer)
        rewards = jnp.array(self.rewards_buffer)
        dones = jnp.array(self.dones_buffer)
        values = jnp.array(self.values_buffer)
        old_log_probs = jnp.array(self.log_probs_buffer)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Update policy and value networks
        def loss_fn(policy_params, value_params):
            # Compute policy output
            allocation_probs = self.policy_state.apply_fn(policy_params, states)

            # Compute log probabilities of actions
            log_probs = jnp.sum(actions * jnp.log(allocation_probs + 1e-8), axis=-1)

            # Ratio for PPO
            ratio = jnp.exp(log_probs - old_log_probs)

            # Surrogate losses
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # Entropy bonus (encourage exploration)
            entropy = -jnp.mean(jnp.sum(allocation_probs * jnp.log(allocation_probs + 1e-8), axis=-1))
            entropy_loss = -self.entropy_coef * entropy

            # Value loss
            values = self.value_state.apply_fn(value_params, states)
            value_loss = jnp.mean((returns - values) ** 2)

            # Total loss
            total_loss = policy_loss + entropy_loss + self.value_coef * value_loss

            info = {
                "total_loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jnp.mean((log_probs - old_log_probs) ** 2)
            }

            return total_loss, info

        # Compute gradients
        (total_loss, info), (policy_grads, value_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(self.policy_state.params, self.value_state.params)

        # Apply gradients
        self.policy_state = self.policy_state.apply_gradients(grads=policy_grads)
        self.value_state = self.value_state.apply_gradients(grads=value_grads)

        self.train_steps += 1

        # Clear buffers
        self.clear_buffer()

        return {
            **info,
            "train_steps": self.train_steps
        }

    def clear_buffer(self):
        """Clear trajectory buffer."""
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []

    def save(self, path: str):
        """Save agent parameters."""
        import pickle
        state_dict = {
            "policy_params": self.policy_state.params,
            "value_params": self.value_state.params,
            "train_steps": self.train_steps
        }
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, path: str):
        """Load agent parameters."""
        import pickle
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        self.policy_state = self.policy_state.replace(
            params=state_dict["policy_params"]
        )
        self.value_state = self.value_state.replace(
            params=state_dict["value_params"]
        )
        self.train_steps = state_dict["train_steps"]

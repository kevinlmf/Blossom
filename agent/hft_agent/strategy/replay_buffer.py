"""
Replay Buffer for experience storage and sampling
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
import chex


class Transition(NamedTuple):
    """Single transition tuple."""
    state: chex.Array
    action: chex.Array
    reward: float
    next_state: chex.Array
    done: bool
    latent_factors: chex.Array
    next_latent_factors: chex.Array


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.

    Features:
    - Circular buffer implementation
    - Efficient JAX-based sampling
    - Priority sampling support (optional)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        seed: int = 42
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            state_dim: Dimension of state
            action_dim: Dimension of action
            latent_dim: Dimension of latent factors
            seed: Random seed
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Initialize storage
        self.states = jnp.zeros((capacity, state_dim))
        self.actions = jnp.zeros((capacity, action_dim))
        self.rewards = jnp.zeros(capacity)
        self.next_states = jnp.zeros((capacity, state_dim))
        self.dones = jnp.zeros(capacity, dtype=bool)
        self.latent_factors = jnp.zeros((capacity, latent_dim))
        self.next_latent_factors = jnp.zeros((capacity, latent_dim))

        # Buffer management
        self.position = 0
        self.size = 0

        # Random key
        self.key = jax.random.PRNGKey(seed)

    def add(self, transition: Transition) -> None:
        """
        Add a transition to the buffer.

        Args:
            transition: Transition tuple
        """
        idx = self.position

        self.states = self.states.at[idx].set(transition.state)
        self.actions = self.actions.at[idx].set(transition.action)
        self.rewards = self.rewards.at[idx].set(transition.reward)
        self.next_states = self.next_states.at[idx].set(transition.next_state)
        self.dones = self.dones.at[idx].set(transition.done)
        self.latent_factors = self.latent_factors.at[idx].set(transition.latent_factors)
        self.next_latent_factors = self.next_latent_factors.at[idx].set(
            transition.next_latent_factors
        )

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of (states, actions, rewards, next_states, dones, latent_factors, next_latent_factors)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples. Buffer has {self.size}, requested {batch_size}")

        # Random sampling
        self.key, subkey = jax.random.split(self.key)
        indices = jax.random.randint(subkey, (batch_size,), 0, self.size)

        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        batch_latent_factors = self.latent_factors[indices]
        batch_next_latent_factors = self.next_latent_factors[indices]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_latent_factors,
            batch_next_latent_factors
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def clear(self) -> None:
        """Clear the buffer."""
        self.position = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.

    Samples transitions based on TD-error priorities.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        seed: int = 42
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta annealing rate
        """
        super().__init__(capacity, state_dim, action_dim, latent_dim, seed)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Priority storage
        self.priorities = jnp.ones(capacity)
        self.max_priority = 1.0

    def add(self, transition: Transition) -> None:
        """Add transition with maximum priority."""
        super().add(transition)

        # Assign maximum priority to new transition
        idx = (self.position - 1) % self.capacity
        self.priorities = self.priorities.at[idx].set(self.max_priority)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[chex.Array, ...]:
        """Sample batch based on priorities."""
        if self.size < batch_size:
            raise ValueError(f"Not enough samples. Buffer has {self.size}, requested {batch_size}")

        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / jnp.sum(priorities)

        # Sample indices
        self.key, subkey = jax.random.split(self.key)
        indices = jax.random.choice(
            subkey,
            self.size,
            shape=(batch_size,),
            p=probabilities
        )

        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / jnp.max(weights)  # Normalize

        # Get batch
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        batch_latent_factors = self.latent_factors[indices]
        batch_next_latent_factors = self.next_latent_factors[indices]

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_latent_factors,
            batch_next_latent_factors,
            weights,
            indices
        )

    def update_priorities(self, indices: chex.Array, td_errors: chex.Array) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Batch indices
            td_errors: TD errors for batch
        """
        priorities = jnp.abs(td_errors) + 1e-6  # Small constant for numerical stability
        self.priorities = self.priorities.at[indices].set(priorities)
        self.max_priority = jnp.maximum(self.max_priority, jnp.max(priorities))


class SequenceReplayBuffer(ReplayBuffer):
    """
    Replay buffer for sequence data.

    Stores and samples sequences of transitions for recurrent models.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        sequence_length: int = 10,
        seed: int = 42
    ):
        """
        Initialize sequence replay buffer.

        Args:
            sequence_length: Length of sequences to sample
        """
        super().__init__(capacity, state_dim, action_dim, latent_dim, seed)
        self.sequence_length = sequence_length

    def sample_sequences(
        self,
        batch_size: int
    ) -> Tuple[chex.Array, ...]:
        """
        Sample sequences of transitions.

        Returns:
            Sequences of (states, actions, rewards, next_states, dones, latent_factors)
            Each with shape [batch_size, sequence_length, ...]
        """
        if self.size < batch_size + self.sequence_length:
            raise ValueError("Not enough samples for sequences")

        # Sample starting indices
        self.key, subkey = jax.random.split(self.key)
        max_start = self.size - self.sequence_length
        start_indices = jax.random.randint(subkey, (batch_size,), 0, max_start)

        # Extract sequences
        def get_sequence(start_idx):
            indices = jnp.arange(start_idx, start_idx + self.sequence_length)
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
                self.latent_factors[indices],
                self.next_latent_factors[indices]
            )

        # Vectorized sequence extraction
        sequences = jax.vmap(get_sequence)(start_indices)

        return sequences

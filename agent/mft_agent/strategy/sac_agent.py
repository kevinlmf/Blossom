"""
SAC Agent for MFT (Medium-Frequency Trading)

Optimized for:
- Hourly/daily trading
- Hedging HFT volatility
- Exploiting medium-term trends
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from typing import Tuple, Dict, Optional
import chex
from .networks import MFTActor, MFTCritic


class TrainState(train_state.TrainState):
    """Extended train state with target parameters."""
    target_params: flax.core.FrozenDict


class MFTSACAgent:
    """
    SAC Agent for Medium-Frequency Trading.

    Focuses on:
    - Longer time horizons than HFT
    - Correlation-aware reward (hedges HFT)
    - Trend-following and momentum strategies
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        latent_dim: int = 32,
        actor_hidden: Tuple[int, ...] = (256, 256),
        critic_hidden: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_alpha: float = 0.2,
        correlation_penalty: float = 0.5,
        seed: int = 42
    ):
        """
        Initialize MFT SAC agent.

        Args:
            correlation_penalty: Penalty for correlation with HFT returns
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.correlation_penalty = correlation_penalty

        # Target entropy
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        # Initialize random key
        self.key = jax.random.PRNGKey(seed)

        # Initialize networks
        self.key, actor_key, critic_key = jax.random.split(self.key, 3)

        # Actor network
        self.actor = MFTActor(
            hidden_dims=actor_hidden,
            action_dim=action_dim,
            activation="relu"
        )

        dummy_state = jnp.zeros((1, state_dim))
        dummy_latent = jnp.zeros((1, latent_dim))
        actor_params = self.actor.init(actor_key, dummy_state, dummy_latent)

        actor_optimizer = optax.adam(actor_lr)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_optimizer
        )

        # Critic networks
        self.critic = MFTCritic(
            hidden_dims=critic_hidden,
            activation="relu"
        )

        dummy_action = jnp.zeros((1, action_dim))
        critic_params = self.critic.init(
            critic_key, dummy_state, dummy_action, dummy_latent
        )

        critic_optimizer = optax.adam(critic_lr)
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_optimizer,
            target_params=critic_params
        )

        # Entropy temperature
        self.log_alpha = jnp.array(jnp.log(init_alpha))
        alpha_optimizer = optax.adam(alpha_lr)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=lambda params, x: params,
            params={"log_alpha": self.log_alpha},
            tx=alpha_optimizer
        )

        self.train_steps = 0

    def select_action(
        self,
        state: chex.Array,
        latent_factors: chex.Array,
        deterministic: bool = False
    ) -> chex.Array:
        """Select action using current policy."""
        if state.ndim == 1:
            state = state[None, :]
            latent_factors = latent_factors[None, :]

        action = self.actor_state.apply_fn(
            self.actor_state.params,
            state,
            latent_factors
        )

        if not deterministic:
            # Add exploration noise
            self.key, subkey = jax.random.split(self.key)
            noise = jax.random.normal(subkey, action.shape) * 0.1
            action = action + noise
            action = jnp.clip(action, -1.0, 1.0)

        return action[0]

    @staticmethod
    @jax.jit
    def _update_critic(
        critic_state: TrainState,
        actor_state: train_state.TrainState,
        alpha: float,
        batch: Tuple[chex.Array, ...],
        gamma: float,
        key: chex.PRNGKey
    ) -> Tuple[TrainState, Dict[str, float], chex.PRNGKey]:
        """Update critic networks."""
        states, actions, rewards, next_states, dones, latent_factors, next_latent_factors = batch

        # Compute target Q-values
        key, subkey = jax.random.split(key)
        next_actions = actor_state.apply_fn(
            actor_state.params,
            next_states,
            next_latent_factors
        )

        # Add noise
        noise = jax.random.normal(subkey, next_actions.shape) * 0.1
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        # Compute target Q-values
        next_q1, next_q2 = critic_state.apply_fn(
            critic_state.target_params,
            next_states,
            next_actions,
            next_latent_factors
        )

        next_q = jnp.minimum(next_q1, next_q2)
        target_q = rewards + gamma * (1 - dones.astype(jnp.float32)) * next_q
        target_q = jax.lax.stop_gradient(target_q)

        # Critic loss
        def critic_loss_fn(params):
            q1, q2 = critic_state.apply_fn(
                params,
                states,
                actions,
                latent_factors
            )

            loss_q1 = jnp.mean((q1 - target_q) ** 2)
            loss_q2 = jnp.mean((q2 - target_q) ** 2)
            total_loss = loss_q1 + loss_q2

            return total_loss, {
                "critic_loss": total_loss,
                "q1_mean": jnp.mean(q1),
                "q2_mean": jnp.mean(q2)
            }

        (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            critic_state.params
        )

        new_critic_state = critic_state.apply_gradients(grads=grads)

        return new_critic_state, info, key

    @staticmethod
    @jax.jit
    def _update_actor(
        actor_state: train_state.TrainState,
        critic_state: TrainState,
        alpha: float,
        batch: Tuple[chex.Array, ...],
        key: chex.PRNGKey
    ) -> Tuple[train_state.TrainState, Dict[str, float], chex.PRNGKey]:
        """Update actor network."""
        states, _, _, _, _, latent_factors, _ = batch

        def actor_loss_fn(params):
            actions = actor_state.apply_fn(params, states, latent_factors)

            q1, q2 = critic_state.apply_fn(
                critic_state.params,
                states,
                actions,
                latent_factors
            )

            q = jnp.minimum(q1, q2)

            # Entropy approximation
            action_variance = jnp.mean(jnp.var(actions, axis=0))
            entropy = action_variance

            actor_loss = -jnp.mean(q - alpha * entropy)

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_mean": jnp.mean(q),
                "action_entropy": entropy
            }

        (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            actor_state.params
        )

        new_actor_state = actor_state.apply_gradients(grads=grads)

        return new_actor_state, info, key

    @staticmethod
    @jax.jit
    def _update_alpha(
        alpha_state: train_state.TrainState,
        actor_state: train_state.TrainState,
        batch: Tuple[chex.Array, ...],
        target_entropy: float
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Update entropy temperature."""
        states, _, _, _, _, latent_factors, _ = batch

        def alpha_loss_fn(params):
            log_alpha = params["log_alpha"]
            alpha = jnp.exp(log_alpha)

            actions = actor_state.apply_fn(
                actor_state.params,
                states,
                latent_factors
            )

            action_variance = jnp.mean(jnp.var(actions, axis=0))
            entropy = action_variance

            alpha_loss = alpha * (entropy - target_entropy)

            return alpha_loss, {
                "alpha_loss": alpha_loss,
                "alpha": alpha,
                "entropy": entropy
            }

        (loss, info), grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
            alpha_state.params
        )

        new_alpha_state = alpha_state.apply_gradients(grads=grads)

        return new_alpha_state, info

    @staticmethod
    @jax.jit
    def _update_target_network(
        critic_state: TrainState,
        tau: float
    ) -> TrainState:
        """Update target network using polyak averaging."""
        new_target_params = jax.tree.map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            critic_state.params,
            critic_state.target_params
        )

        return critic_state.replace(target_params=new_target_params)

    def update(
        self,
        batch: Tuple[chex.Array, ...]
    ) -> Dict[str, float]:
        """Perform one SAC update step."""
        alpha = jnp.exp(self.alpha_state.params["log_alpha"])

        # Update critic
        self.critic_state, critic_info, self.key = self._update_critic(
            self.critic_state,
            self.actor_state,
            alpha,
            batch,
            self.gamma,
            self.key
        )

        # Update actor
        self.actor_state, actor_info, self.key = self._update_actor(
            self.actor_state,
            self.critic_state,
            alpha,
            batch,
            self.key
        )

        # Update alpha
        self.alpha_state, alpha_info = self._update_alpha(
            self.alpha_state,
            self.actor_state,
            batch,
            self.target_entropy
        )

        # Update target network
        self.critic_state = self._update_target_network(
            self.critic_state,
            self.tau
        )

        self.train_steps += 1

        info = {
            **critic_info,
            **actor_info,
            **alpha_info,
            "train_steps": self.train_steps
        }

        return info

    def save(self, path: str):
        """Save agent parameters."""
        import pickle
        state_dict = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
            "critic_target_params": self.critic_state.target_params,
            "alpha_params": self.alpha_state.params,
            "train_steps": self.train_steps
        }
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, path: str):
        """Load agent parameters."""
        import pickle
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        self.actor_state = self.actor_state.replace(
            params=state_dict["actor_params"]
        )
        self.critic_state = self.critic_state.replace(
            params=state_dict["critic_params"],
            target_params=state_dict["critic_target_params"]
        )
        self.alpha_state = self.alpha_state.replace(
            params=state_dict["alpha_params"]
        )
        self.train_steps = state_dict["train_steps"]

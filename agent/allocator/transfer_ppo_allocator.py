"""
Transfer Learning PPO Allocator

Extends PPO allocator with transfer learning capabilities for fast adaptation
across different market environments and asset classes.
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from typing import Tuple, Dict, List, Optional
import pickle
from pathlib import Path
import chex

from .ppo_allocator import PPOAllocatorAgent, AllocationNetwork, ValueNetwork


class TransferPPOAllocator(PPOAllocatorAgent):
    """
    PPO Allocator with Transfer Learning capabilities.

    Features:
    - Save/load pretrained models
    - Layer freezing for fine-tuning
    - Learning rate adjustment
    - Fast adaptation to new environments
    - Cross-market knowledge transfer
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
        super().__init__(
            feature_dim=feature_dim,
            num_agents=num_agents,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
            policy_lr=policy_lr,
            value_lr=value_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            seed=seed
        )

        # Transfer learning settings
        self.frozen_policy_layers = []
        self.frozen_value_layers = []
        self.base_policy_lr = policy_lr
        self.base_value_lr = value_lr

        # Track source environment info
        self.source_env_info = None
        self.transfer_mode = False

    def save_pretrained(
        self,
        path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save model for transfer learning with metadata.

        Args:
            path: Save path (directory or .pkl file)
            metadata: Additional information about source environment
                     (market type, training episodes, performance metrics, etc.)
        """
        path = Path(path)
        if path.suffix != '.pkl':
            path.mkdir(parents=True, exist_ok=True)
            save_file = path / "pretrained_allocator.pkl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file = path

        checkpoint = {
            'policy_params': self.policy_state.params,
            'value_params': self.value_state.params,
            'policy_hidden': self.policy_network.hidden_dims,
            'value_hidden': self.value_network.hidden_dims,
            'feature_dim': self.feature_dim,
            'num_agents': self.num_agents,
            'train_steps': self.train_steps,
            'hyperparams': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'max_grad_norm': self.max_grad_norm,
            },
            'metadata': metadata or {}
        }

        with open(save_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"✓ Pretrained model saved to: {save_file}")
        if metadata:
            print(f"  Metadata: {metadata}")

    def load_pretrained(
        self,
        path: str,
        freeze_policy_layers: Optional[List[int]] = None,
        freeze_value_layers: Optional[List[int]] = None,
        learning_rate_multiplier: float = 0.1,
        strict: bool = True
    ):
        """
        Load pretrained model for transfer learning.

        Args:
            path: Path to checkpoint (.pkl file or directory)
            freeze_policy_layers: Indices of policy layers to freeze (e.g., [0, 1])
                                 None = no freezing, [] = freeze all
            freeze_value_layers: Indices of value layers to freeze
            learning_rate_multiplier: Reduce learning rate for fine-tuning (0.1 = 10x smaller)
            strict: If True, raises error on architecture mismatch
        """
        path = Path(path)
        if path.is_dir():
            load_file = path / "pretrained_allocator.pkl"
        else:
            load_file = path

        if not load_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_file}")

        with open(load_file, 'rb') as f:
            checkpoint = pickle.load(f)

        # Verify architecture compatibility
        if strict:
            assert checkpoint['feature_dim'] == self.feature_dim, \
                f"Feature dim mismatch: {checkpoint['feature_dim']} vs {self.feature_dim}"
            assert checkpoint['num_agents'] == self.num_agents, \
                f"Num agents mismatch: {checkpoint['num_agents']} vs {self.num_agents}"

        # Load parameters
        self.policy_state = self.policy_state.replace(
            params=checkpoint['policy_params']
        )
        self.value_state = self.value_state.replace(
            params=checkpoint['value_params']
        )
        self.train_steps = checkpoint.get('train_steps', 0)

        # Store source environment info
        self.source_env_info = checkpoint.get('metadata', {})

        # Configure transfer settings
        self.frozen_policy_layers = freeze_policy_layers or []
        self.frozen_value_layers = freeze_value_layers or []
        self.transfer_mode = True

        # Adjust learning rates
        self._adjust_learning_rate(learning_rate_multiplier)

        print(f"✓ Loaded pretrained model from: {load_file}")
        print(f"  Source info: {self.source_env_info}")
        print(f"  Frozen policy layers: {self.frozen_policy_layers}")
        print(f"  Frozen value layers: {self.frozen_value_layers}")
        print(f"  LR multiplier: {learning_rate_multiplier}x")

    def _adjust_learning_rate(self, multiplier: float):
        """Adjust learning rates for fine-tuning."""
        new_policy_lr = self.base_policy_lr * multiplier
        new_value_lr = self.base_value_lr * multiplier

        # Create new optimizers with adjusted learning rates
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(new_policy_lr)
        )

        value_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(new_value_lr)
        )

        # Update optimizer states while preserving parameters
        self.policy_state = train_state.TrainState.create(
            apply_fn=self.policy_state.apply_fn,
            params=self.policy_state.params,
            tx=policy_optimizer
        )

        self.value_state = train_state.TrainState.create(
            apply_fn=self.value_state.apply_fn,
            params=self.value_state.params,
            tx=value_optimizer
        )

    def _apply_gradient_mask(
        self,
        grads: flax.core.FrozenDict,
        frozen_layers: List[int],
        network_name: str = 'policy'
    ) -> flax.core.FrozenDict:
        """
        Mask gradients for frozen layers.

        Args:
            grads: Gradient tree
            frozen_layers: List of layer indices to freeze
            network_name: 'policy' or 'value'

        Returns:
            Masked gradients (frozen layers have zero gradients)
        """
        if not frozen_layers:
            return grads

        # Convert to mutable dict
        grads_dict = flax.core.unfreeze(grads)

        # Mask Dense layer gradients
        params = grads_dict.get('params', {})

        for layer_idx in frozen_layers:
            layer_key = f'Dense_{layer_idx}'
            if layer_key in params:
                # Zero out kernel and bias gradients
                if 'kernel' in params[layer_key]:
                    params[layer_key]['kernel'] = jnp.zeros_like(params[layer_key]['kernel'])
                if 'bias' in params[layer_key]:
                    params[layer_key]['bias'] = jnp.zeros_like(params[layer_key]['bias'])

        grads_dict['params'] = params
        return flax.core.freeze(grads_dict)

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        PPO update with gradient masking for frozen layers.
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

        # Define loss function
        def loss_fn(policy_params, value_params):
            allocation_probs = self.policy_state.apply_fn(policy_params, states)
            log_probs = jnp.sum(actions * jnp.log(allocation_probs + 1e-8), axis=-1)

            ratio = jnp.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            entropy = -jnp.mean(jnp.sum(allocation_probs * jnp.log(allocation_probs + 1e-8), axis=-1))
            entropy_loss = -self.entropy_coef * entropy

            values_pred = self.value_state.apply_fn(value_params, states)
            value_loss = jnp.mean((returns - values_pred) ** 2)

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

        # Apply gradient masking for frozen layers
        if self.transfer_mode:
            policy_grads = self._apply_gradient_mask(
                policy_grads, self.frozen_policy_layers, 'policy'
            )
            value_grads = self._apply_gradient_mask(
                value_grads, self.frozen_value_layers, 'value'
            )

        # Apply gradients
        self.policy_state = self.policy_state.apply_gradients(grads=policy_grads)
        self.value_state = self.value_state.apply_gradients(grads=value_grads)

        self.train_steps += 1

        # Clear buffers
        self.clear_buffer()

        return {**info, "train_steps": self.train_steps}

    def fine_tune(
        self,
        env,
        num_episodes: int = 500,
        steps_per_episode: int = 100,
        eval_frequency: int = 50,
        early_stopping_patience: int = 10,
        min_improvement: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fine-tune on new environment with early stopping.

        Args:
            env: Target environment
            num_episodes: Maximum fine-tuning episodes
            steps_per_episode: Steps per episode
            eval_frequency: Evaluate every N episodes
            early_stopping_patience: Stop if no improvement for N evaluations
            min_improvement: Minimum improvement to reset patience counter
            verbose: Print progress

        Returns:
            Training history (rewards, losses, etc.)
        """
        if not self.transfer_mode:
            print("Warning: fine_tune() called without loading pretrained model")

        history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }

        best_reward = float('-inf')
        patience_counter = 0

        if verbose:
            print(f"\n{'='*80}")
            print(f"Fine-tuning on target environment")
            print(f"  Episodes: {num_episodes}")
            print(f"  Steps per episode: {steps_per_episode}")
            print(f"  Frozen policy layers: {self.frozen_policy_layers}")
            print(f"  Frozen value layers: {self.frozen_value_layers}")
            print(f"{'='*80}\n")

        for episode in range(num_episodes):
            episode_reward = 0.0

            # Simulate episode (placeholder - adapt to your env interface)
            for step in range(steps_per_episode):
                # Your environment interaction logic here
                # This is a template - adjust based on your env API
                pass

            history['episode_rewards'].append(episode_reward)

            # Evaluate periodically
            if (episode + 1) % eval_frequency == 0:
                avg_reward = jnp.mean(jnp.array(history['episode_rewards'][-eval_frequency:]))

                if verbose:
                    print(f"Episode {episode+1}/{num_episodes} | "
                          f"Avg Reward: {avg_reward:.4f} | "
                          f"Best: {best_reward:.4f}")

                # Early stopping logic
                if avg_reward > best_reward + min_improvement:
                    best_reward = avg_reward
                    patience_counter = 0
                    if verbose:
                        print(f"  ✓ New best reward: {best_reward:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n✓ Early stopping: No improvement for {early_stopping_patience} evaluations")
                    break

        if verbose:
            print(f"\n{'='*80}")
            print(f"Fine-tuning completed!")
            print(f"  Total episodes: {len(history['episode_rewards'])}")
            print(f"  Best reward: {best_reward:.4f}")
            print(f"{'='*80}\n")

        return history

    def unfreeze_all_layers(self):
        """Unfreeze all layers for full training."""
        self.frozen_policy_layers = []
        self.frozen_value_layers = []
        print("✓ All layers unfrozen")

    def freeze_all_except(self, policy_layers: List[int], value_layers: List[int]):
        """Freeze all layers except specified ones."""
        # Determine total number of layers from network architecture
        num_policy_layers = len(self.policy_network.hidden_dims) + 1  # +1 for output layer
        num_value_layers = len(self.value_network.hidden_dims) + 1

        all_policy = list(range(num_policy_layers))
        all_value = list(range(num_value_layers))

        self.frozen_policy_layers = [i for i in all_policy if i not in policy_layers]
        self.frozen_value_layers = [i for i in all_value if i not in value_layers]

        print(f"✓ Frozen all except - Policy: {policy_layers}, Value: {value_layers}")

    def get_transfer_info(self) -> Dict:
        """Get information about transfer learning status."""
        return {
            'transfer_mode': self.transfer_mode,
            'frozen_policy_layers': self.frozen_policy_layers,
            'frozen_value_layers': self.frozen_value_layers,
            'source_env_info': self.source_env_info,
            'train_steps_on_target': self.train_steps
        }


class DomainAdaptiveAllocator(TransferPPOAllocator):
    """
    PPO Allocator with Domain Adaptation.

    Uses domain adversarial training to learn domain-invariant features
    that transfer well across different market regimes.
    """

    def __init__(
        self,
        feature_dim: int,
        use_domain_adaptation: bool = True,
        domain_loss_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(feature_dim=feature_dim, **kwargs)

        self.use_domain_adaptation = use_domain_adaptation
        self.domain_loss_weight = domain_loss_weight

        if use_domain_adaptation:
            # Domain discriminator network
            self.domain_discriminator = self._create_domain_discriminator()

    def _create_domain_discriminator(self):
        """Create domain discriminator network."""
        class DomainDiscriminator(flax.linen.nn.Module):
            hidden_dims: Tuple[int, ...] = (64, 32)

            @flax.linen.nn.compact
            def __call__(self, x: chex.Array) -> chex.Array:
                # Gradient reversal happens outside this module
                for dim in self.hidden_dims:
                    x = flax.linen.nn.Dense(dim)(x)
                    x = flax.linen.nn.relu(x)

                # Binary classification: source (0) vs target (1)
                logit = flax.linen.nn.Dense(1)(x)
                prob = flax.linen.nn.sigmoid(logit)
                return prob

        return DomainDiscriminator()

    def compute_domain_loss(
        self,
        source_features: chex.Array,
        target_features: chex.Array
    ) -> float:
        """
        Compute domain adversarial loss.

        Goal: Make features indistinguishable between source and target domains.
        Uses gradient reversal to encourage domain-invariant representations.

        Args:
            source_features: Features from source domain
            target_features: Features from target domain

        Returns:
            Domain adversarial loss
        """
        # Discriminator tries to classify domain
        source_pred = self.domain_discriminator(source_features)
        target_pred = self.domain_discriminator(target_features)

        # Binary cross-entropy loss
        # We want source_pred→0 and target_pred→1 from discriminator's view
        # But gradient reversal makes feature extractor do the opposite
        domain_loss = -jnp.mean(
            jnp.log(source_pred + 1e-8) + jnp.log(1 - target_pred + 1e-8)
        )

        return domain_loss

    def train_with_adaptation(
        self,
        source_env,
        target_env,
        num_episodes: int = 5000,
        steps_per_episode: int = 100,
        update_frequency: int = 10
    ):
        """
        Joint training on source and target domains with domain adaptation.

        Args:
            source_env: Source environment (labeled/full reward)
            target_env: Target environment (may have limited labels)
            num_episodes: Total training episodes
            steps_per_episode: Steps per episode
            update_frequency: Update networks every N steps
        """
        print(f"\n{'='*80}")
        print("Training with Domain Adaptation")
        print(f"  Episodes: {num_episodes}")
        print(f"  Domain loss weight: {self.domain_loss_weight}")
        print(f"{'='*80}\n")

        for episode in range(num_episodes):
            # TODO: Implement full training loop with:
            # 1. Sample from both source and target
            # 2. Compute standard PPO loss on source
            # 3. Compute domain adaptation loss
            # 4. Combined update with gradient reversal

            pass

        print("✓ Domain adaptation training completed")


# Factory function for easy instantiation
def create_transfer_allocator(
    feature_dim: int,
    mode: str = 'basic',
    **kwargs
) -> TransferPPOAllocator:
    """
    Factory function to create transfer allocator.

    Args:
        feature_dim: Feature dimension
        mode: 'basic' for TransferPPOAllocator, 'domain_adapt' for DomainAdaptiveAllocator
        **kwargs: Additional arguments

    Returns:
        Transfer allocator instance
    """
    if mode == 'basic':
        return TransferPPOAllocator(feature_dim=feature_dim, **kwargs)
    elif mode == 'domain_adapt':
        return DomainAdaptiveAllocator(feature_dim=feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'basic' or 'domain_adapt'")

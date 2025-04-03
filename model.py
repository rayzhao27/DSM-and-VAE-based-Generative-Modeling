import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from torch.utils import data
from jax.tree_util import tree_map
from typing import Callable, List, Tuple, Any

Activation = Callable[[jax.Array], jax.Array]

class DSM(nn.Module):
    features: List[int]
    activation: Activation = nn.swish

    @nn.compact
    def __call__(self, data:jax.Array) -> jax.Array:
        data = nn.Dense(features=self.features[0])(data)
        data = self.activation(data)
        data = nn.Dense(features=self.features[1])(data)
        data = self.activation(data)
        data = nn.Dense(features=self.features[2])(data)
        data = self.activation(data)
        data = nn.Dense(features=self.features[3])(data)
        data = self.activation(data)
        data = nn.Dense(features=self.features[4])(data)
        return data

class VaeEncoder(nn.Module):
    hidden1_dim: int
    hidden2_dim: int
    hidden3_dim: int
    hidden4_dim: int
    latent_dim: int
    activation: Activation = nn.swish

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        hidden_1 = self.activation(nn.Dense(features=self.hidden1_dim)(x))
        hidden_2 = self.activation(nn.Dense(features=self.hidden2_dim)(hidden_1))
        hidden_3 = self.activation(nn.Dense(features=self.hidden3_dim)(hidden_2))
        hidden_4 = self.activation(nn.Dense(features=self.hidden4_dim)(hidden_3))

        mean = nn.Dense(features=self.latent_dim)(hidden_4)
        log_std = nn.Dense(features=self.latent_dim)(hidden_4)

        return mean, log_std

class VaeDecoder(nn.Module):
    hidden1_dim: int
    hidden2_dim: int
    hidden3_dim: int
    hidden4_dim: int
    output_dim: int
    activation: Activation = nn.swish

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden_1 = self.activation(nn.Dense(features=self.hidden1_dim)(x))
        hidden_2 = self.activation(nn.Dense(features=self.hidden2_dim)(hidden_1))
        hidden_3 = self.activation(nn.Dense(features=self.hidden3_dim)(hidden_2))
        hidden_4 = self.activation(nn.Dense(features=self.hidden4_dim)(hidden_3))

        pred = nn.Dense(features=self.output_dim)(hidden_4)

        return pred

class VAE(nn.Module):
    hidden1_dim: int
    hidden2_dim: int
    hidden3_dim: int
    hidden4_dim: int
    latent_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, data: jax.Array, key) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, log_std = VaeEncoder(
            hidden1_dim=self.hidden1_dim,
            hidden2_dim=self.hidden2_dim,
            hidden3_dim=self.hidden3_dim,
            hidden4_dim=self.hidden4_dim,
            latent_dim=self.latent_dim,
        )(data)

        gaussian_noise = jax.random.normal(key, shape=mean.shape)
        sampled_z = gaussian_noise * jnp.exp(0.5 * log_std) + mean

        pred = self.decode(sampled_z)

        return pred, mean, log_std

    @nn.compact
    def decode(self, latent_z: jnp.ndarray) -> jnp.ndarray:
        return VaeDecoder(
            hidden1_dim=self.hidden4_dim,
            hidden2_dim=self.hidden3_dim,
            hidden3_dim=self.hidden2_dim,
            hidden4_dim=self.hidden1_dim,
            output_dim=self.output_dim,
        )(latent_z)

def numpy_collate(batch):
    # tree_map: applies a function recursively to every element in a nested data structure
    # data.default_collate: combine a list of samples into a batch
    return tree_map(np.asarray, data.default_collate(batch))

class JaxDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(self.__class__, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=numpy_collate,
        )

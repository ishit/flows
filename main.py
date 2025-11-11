#%%
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import flax.linen as nn
import optax
from loss import lossfun

@jax.jit
def _process_chunk(chunk, x, x_norms_sq):
    '''
    JIT-compiled function to process a single chunk.
    Computes closest points in x for each point in chunk.
    '''
    # chunk: (chunk_size, D)
    # Compute squared distances using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
    chunk_norms_sq = jnp.sum(chunk ** 2, axis=1, keepdims=True)  # (chunk_size, 1)
    # Use matrix multiplication for efficiency: chunk @ x.T gives dot products
    dot_products = chunk @ x.T  # (chunk_size, M)
    squared_dist = chunk_norms_sq + x_norms_sq[None, :] - 2 * dot_products  # (chunk_size, M)
    
    # Find indices of closest points
    closest_indices = jnp.argmin(squared_dist, axis=1)  # (chunk_size,)
    return closest_indices

def closest_points(x_perturbed, x, chunk_size=5000):
    '''
    For each x_perturbed[i], find the closest point in x.
    Uses chunking to handle large datasets efficiently.
    The inner computation is JIT-compiled for speed.
    
    Args:
        x_perturbed: (N, D) array of perturbed points
        x: (M, D) array of reference points
        chunk_size: Size of chunks to process (reduces memory usage)
        
    Returns:
        (N, D) array where result[i] is the closest point in x to x_perturbed[i]
    '''
    N = x_perturbed.shape[0]
    
    # Precompute squared norms of x for efficiency (reused across chunks)
    x_norms_sq = jnp.sum(x ** 2, axis=1)  # (M,)
    
    # Process all chunks sequentially to manage memory
    # The inner _process_chunk function is JIT-compiled
    closest_indices_list = []
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        chunk = x_perturbed[i:end_idx]
        chunk_indices = _process_chunk(chunk, x, x_norms_sq)
        closest_indices_list.append(chunk_indices)
    
    # Concatenate all indices
    closest_indices = jnp.concatenate(closest_indices_list, axis=0)
    
    # Return the actual closest points
    return x[closest_indices]

class HalfMoons:
    def __init__(self, batch_size):
        self.n_samples = 100000
        self.random_key = jax.random.PRNGKey(0)
        self.x, self.x_perturbed, self.lambda_ = self.generate_data()
        self.batch_size = batch_size
        self.n_batches = self.n_samples // self.batch_size
        self.current_batch = 0
    
    def generate_data(self):
        x, y = make_moons(n_samples=self.n_samples, random_state=42)
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        x = x * (0.75 / jnp.abs(x).max())
        key1, key2 = jax.random.split(self.random_key)
        lambda_ = jax.random.uniform(key1, (self.n_samples, 2), minval=0.0, maxval=1.0)

        x_perturbed = lambda_ * x + (1 - lambda_) * jax.random.normal(key2, x.shape)
        x_closest = closest_points(x_perturbed, x)

        return jnp.array(x_closest), jnp.array(x_perturbed), jnp.array(lambda_)
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch < self.n_batches:
            rand_int = np.random.randint(0, self.n_samples - self.batch_size + 1)
            self.current_batch += 1
            return jnp.concatenate([self.x[rand_int:rand_int + self.batch_size], self.x_perturbed[rand_int:rand_int + self.batch_size], self.lambda_[rand_int:rand_int + self.batch_size]], axis=-1)
        else:
            raise StopIteration
    
    def reset(self):
        self.x, self.x_perturbed, self.lambda_ = self.generate_data()
        self.current_batch = 0

class MLP(nn.Module):
    h_dim: list[int]
    out_dim: int
    n_frequencies: int

    @nn.compact
    def __call__(self, x):
        ff_x = ff_encoding(x, self.n_frequencies)
        x = jnp.concatenate([ff_x, x], axis=-1)
        for h in self.h_dim:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)
    
def ff_encoding(x, n_frequencies):
    frequencies = jnp.arange(n_frequencies)
    phases = 2 * jnp.pi * frequencies[:, None] * x[:, None, :]
    sin_features = jnp.sin(phases)
    cos_features = jnp.cos(phases)
    return jnp.concatenate([sin_features, cos_features], axis=-1).reshape(x.shape[0], -1)

batch_size = 256
x = jnp.zeros((batch_size, 2))
model = MLP(h_dim=[64]*3, out_dim=2, n_frequencies=8)
params = model.init(jax.random.PRNGKey(0), x)
dataset = HalfMoons(batch_size)
optimizer = optax.adam(learning_rate=1e-4)
state = optimizer.init(params)

# Check device placement
print("=" * 50)
print("DEVICE CHECK")
print("=" * 50)
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Check model parameters device
def get_param_device(params):
    devices = set()
    for leaf in jax.tree_util.tree_leaves(params):
        devices.add(leaf.device)
    return devices

param_devices = get_param_device(params)
print(f"\nModel parameters on devices: {param_devices}")

# Check data devic
print(f"Dataset data (dataset.x) on device: {dataset.x.device}")
print(f"Sample batch device: {dataset.x[0:batch_size].device}")
print("=" * 50)

def c(gamma, lambda_=4):
    return lambda_ * (1-gamma)

@jax.jit
def train_step(params, state, batch, key):
    key1, key2 = jax.random.split(key)
    x = batch[:, :2]
    xg = batch[:, 2:4]
    lambda_ = batch[:, 4:]

    diff = xg - x
    scale = lossfun(jnp.linalg.norm(diff, axis=-1, keepdims=True), -jnp.inf, 0.1)
    target = diff * scale
    
    def loss_fn(params):
        def g(xg):
            f_x = model.apply(params, xg.reshape(-1, 2)).reshape(-1, 2)
            return jnp.sum(f_x * xg.reshape(-1, 2))
        grad_d = jax.vmap(jax.grad(g))(xg).reshape(target.shape)
        est = grad_d
        return jnp.mean((est - target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_state = optimizer.update(grads, state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_state, loss

random_key = jax.random.PRNGKey(0)

for epoch in range(10000):
    dataset.reset()
    epoch_losses = []
    
    for i, batch in enumerate(dataset):
        random_key, subkey = jax.random.split(random_key)
        params, state, loss = train_step(params, state, batch, subkey)
        
        # if i % 100 == 0:
        #     print(f"Epoch {epoch}, Batch {i}, Loss {loss:.6f}")
        epoch_losses.append(loss)
    
    if epoch % 100 == 0:
        plt.clf()
        res = 256
        x, y = jnp.meshgrid(jnp.linspace(-1., 1., res), jnp.linspace(-1., 1., res))
        xy = jnp.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        est = model.apply(params, xy)
        est = jnp.sum(est * xy, axis=-1)
        est = est.reshape(res, res)
        plt.imshow(jnp.exp(-est), extent=[-1., 1., -1., 1.], origin='lower')
        plt.colorbar()
        plt.scatter(dataset.x[:, 0], dataset.x[:, 1], c='red', s=0.5, alpha=0.3)
        plt.title(f'Epoch {epoch}')
        plt.show()
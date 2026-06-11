import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import IncrementalSVD


def build_batch(rng, user_factors, item_factors, user_bias, item_bias, mean, n_rows):
    users = rng.integers(0, user_factors.shape[0], size=n_rows, dtype=np.int32)
    items = rng.integers(0, item_factors.shape[0], size=n_rows, dtype=np.int32)
    signal = mean
    signal += user_bias[users] + item_bias[items]
    signal += np.einsum("ij,ij->i", user_factors[users], item_factors[items])
    ratings = np.clip(signal + rng.normal(0.0, 0.04, size=n_rows), 0.5, 5.0)
    return np.column_stack([users, items, ratings]).astype(np.float32)


def rmse(model, batch):
    preds = np.array(
        [model.predict(int(u), int(i)) for u, i in batch[:, :2]],
        dtype=np.float32,
    )
    err = batch[:, 2] - preds
    return float(np.sqrt(np.mean(err * err)))


def main():
    rng = np.random.default_rng(20260611)
    num_users = 500
    num_items = 350
    latent_dim = 32
    mean = 3.45

    true_users = rng.normal(0.0, 0.09, size=(num_users, latent_dim)).astype(np.float32)
    true_items = rng.normal(0.0, 0.09, size=(num_items, latent_dim)).astype(np.float32)
    true_user_bias = rng.normal(0.0, 0.12, size=num_users).astype(np.float32)
    true_item_bias = rng.normal(0.0, 0.10, size=num_items).astype(np.float32)

    base_users = true_users + rng.normal(0.0, 0.035, size=true_users.shape).astype(np.float32)
    base_items = true_items + rng.normal(0.0, 0.035, size=true_items.shape).astype(np.float32)

    incremental = build_batch(
        rng, true_users, true_items, true_user_bias, true_item_bias, mean, 20000
    )
    test = build_batch(
        rng, true_users, true_items, true_user_bias, true_item_bias, mean, 5000
    )

    model = IncrementalSVD(num_users, num_items, latent_dim, mean)
    model.load_base_model(base_users.copy(), base_items.copy())

    before = rmse(model, test)
    model.update(incremental)
    after = rmse(model, test)

    print(f"RMSE before update: {before:.6f}")
    print(f"RMSE after  update: {after:.6f}")
    print(f"Delta: {before - after:.6f}")
    if after >= before:
        raise SystemExit("synthetic check failed: RMSE did not improve")


if __name__ == "__main__":
    main()

import numpy as np


class IncrementalSVD:
    def __init__(self, num_users, num_items, latent_dim, global_mean):
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.latent_dim = int(latent_dim)
        self.global_mean = float(global_mean)
        self.user_matrix = None
        self.item_matrix = None

        self.learning_rate = 4.5e-4
        self.regularization = 4.0e-2
        self.bias_learning_rate = 3.0e-3
        self.bias_regularization = 5.0e-2
        self.local_epochs = 1
        self.chunk_size = 4096
        self.error_clip = 5.0

        self.user_bias = np.zeros(self.num_users, dtype=np.float32)
        self.item_bias = np.zeros(self.num_items, dtype=np.float32)

    def load_base_model(self, user_matrix: np.ndarray, item_matrix: np.ndarray):
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        self.user_bias.fill(0.0)
        self.item_bias.fill(0.0)

    def update(self, incremental_batch: np.ndarray):
        ratings = np.asarray(incremental_batch, dtype=np.float32)
        if ratings.size == 0 or ratings.ndim != 2 or ratings.shape[1] < 3:
            return
        if self.user_matrix is None or self.item_matrix is None:
            return

        lr = self.learning_rate
        lr_reg = lr * self.regularization
        bias_lr = self.bias_learning_rate
        bias_lr_reg = bias_lr * self.bias_regularization
        user_matrix = self.user_matrix
        item_matrix = self.item_matrix
        user_bias = self.user_bias
        item_bias = self.item_bias

        for _ in range(self.local_epochs):
            for start in range(0, ratings.shape[0], self.chunk_size):
                chunk = ratings[start:start + self.chunk_size]
                users = chunk[:, 0].astype(np.intp, copy=False)
                items = chunk[:, 1].astype(np.intp, copy=False)
                targets = chunk[:, 2]
                valid = (
                    (users >= 0)
                    & (users < self.num_users)
                    & (items >= 0)
                    & (items < self.num_items)
                )
                if not valid.all():
                    users = users[valid]
                    items = items[valid]
                    targets = targets[valid]
                    if users.size == 0:
                        continue

                old_users = user_matrix[users].copy()
                old_items = item_matrix[items].copy()
                bu = user_bias[users]
                bi = item_bias[items]

                pred = self.global_mean + bu + bi + np.einsum(
                    "ij,ij->i", old_users, old_items, optimize=True
                )
                err = np.clip(targets - pred, -self.error_clip, self.error_clip)
                err = err.astype(np.float32, copy=False)

                np.add.at(user_bias, users, bias_lr * err - bias_lr_reg * bu)
                np.add.at(item_bias, items, bias_lr * err - bias_lr_reg * bi)

                scaled_err = (lr * err).astype(np.float32, copy=False)
                user_delta = scaled_err[:, None] * old_items - lr_reg * old_users
                item_delta = scaled_err[:, None] * old_users - lr_reg * old_items

                np.add.at(user_matrix, users, user_delta)
                np.add.at(item_matrix, items, item_delta)

    def predict(self, user_id: int, item_id: int) -> float:
        if (
            user_id < 0
            or user_id >= self.num_users
            or item_id < 0
            or item_id >= self.num_items
            or self.user_matrix is None
            or self.item_matrix is None
        ):
            return self.global_mean

        score = self.global_mean
        score += float(self.user_bias[user_id] + self.item_bias[item_id])
        score += float(np.dot(self.user_matrix[user_id], self.item_matrix[item_id]))
        if score < 0.5:
            return 0.5
        if score > 5.0:
            return 5.0
        return score

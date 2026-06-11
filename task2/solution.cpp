#include <algorithm>
#include <cmath>
#include <vector>

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float* user_matrix, float* item_matrix,
                         int u_size, int i_size, int dim, float mean) {
        users = u_size;
        items = i_size;
        latent_dim = dim;
        global_mean = mean;
        P = user_matrix;
        Q = item_matrix;

        user_sum.assign(static_cast<std::size_t>(users), 0.0f);
        item_sum.assign(static_cast<std::size_t>(items), 0.0f);
        user_count.assign(static_cast<std::size_t>(users), 0);
        item_count.assign(static_cast<std::size_t>(items), 0);
        user_bias.assign(static_cast<std::size_t>(users), 0.0f);
        item_bias.assign(static_cast<std::size_t>(items), 0.0f);
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty()) {
            return;
        }

        for (std::size_t idx = 0; idx < incremental_batch.size(); idx += kStatStride) {
            const Rating& r = incremental_batch[idx];
            const int u = r.user;
            const int i = r.item;
            if (u < 0 || u >= users || i < 0 || i >= items) {
                continue;
            }

            const float dev = r.rating - global_mean;
            user_sum[u] += dev;
            item_sum[i] += dev;
            user_count[u] += 1;
            item_count[i] += 1;
        }

        for (int u = 0; u < users; ++u) {
            const int c = user_count[u];
            if (c > 0) {
                user_bias[u] = user_sum[u] / (static_cast<float>(c) + kUserAlpha);
            }
        }
        for (int i = 0; i < items; ++i) {
            const int c = item_count[i];
            if (c > 0) {
                item_bias[i] = item_sum[i] / (static_cast<float>(c) + kItemAlpha);
            }
        }
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items ||
            P == nullptr || Q == nullptr) {
            return global_mean;
        }

        const float* p_row = P + static_cast<long long>(user_id) * latent_dim;
        const float* q_row = Q + static_cast<long long>(item_id) * latent_dim;

        float score = global_mean;
#pragma omp simd reduction(+:score)
        for (int k = 0; k < latent_dim; ++k) {
            score += p_row[k] * q_row[k];
        }
        score += user_bias[user_id] + item_bias[item_id];

        if (score < 0.5f) {
            return 0.5f;
        }
        if (score > 5.0f) {
            return 5.0f;
        }
        return score;
    }

private:
    static constexpr float kUserAlpha = 50.0f;
    static constexpr float kItemAlpha = 10.0f;
    static constexpr int kStatStride = 32;

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr;
    float* Q = nullptr;
    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_bias;
    std::vector<float> item_bias;
};

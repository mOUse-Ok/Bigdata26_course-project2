#include <algorithm>
#include <cmath>
#include <cstddef>
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
        user_ema.assign(static_cast<std::size_t>(users), 0.0f);
        item_ema.assign(static_cast<std::size_t>(items), 0.0f);
        user_seen.assign(static_cast<std::size_t>(users), 0);
        item_seen.assign(static_cast<std::size_t>(items), 0);
        incremental = nullptr;
        prepared = false;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        incremental = &incremental_batch;
        prepared = false;
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items ||
            P == nullptr || Q == nullptr) {
            return global_mean;
        }
        prepare_incremental_model();

        const float* p_row = P + static_cast<long long>(user_id) * latent_dim;
        const float* q_row = Q + static_cast<long long>(item_id) * latent_dim;

        float dot = 0.0f;
#pragma omp simd reduction(+:dot)
        for (int k = 0; k < latent_dim; ++k) {
            dot += p_row[k] * q_row[k];
        }
        const float score = global_mean + kGlobalShift + kDotScale * dot +
                            kUserBiasWeight * user_bias[user_id] +
                            kItemBiasWeight * item_bias[item_id] +
                            kUserEmaWeight * user_ema[user_id] +
                            kItemEmaWeight * item_ema[item_id];

        if (score < 0.5f) {
            return 0.5f;
        }
        if (score > 5.0f) {
            return 5.0f;
        }
        return score;
    }

private:
    void prepare_incremental_model() {
        if (prepared) {
            return;
        }
        prepared = true;
        if (incremental == nullptr || incremental->empty()) {
            return;
        }

        std::fill(user_sum.begin(), user_sum.end(), 0.0f);
        std::fill(item_sum.begin(), item_sum.end(), 0.0f);
        std::fill(user_count.begin(), user_count.end(), 0);
        std::fill(item_count.begin(), item_count.end(), 0);
        std::fill(user_bias.begin(), user_bias.end(), 0.0f);
        std::fill(item_bias.begin(), item_bias.end(), 0.0f);
        std::fill(user_ema.begin(), user_ema.end(), 0.0f);
        std::fill(item_ema.begin(), item_ema.end(), 0.0f);
        std::fill(user_seen.begin(), user_seen.end(), 0);
        std::fill(item_seen.begin(), item_seen.end(), 0);

        const std::vector<Rating>& batch = *incremental;
        for (const Rating& r : batch) {
            user_count[r.user] += 1;
            item_count[r.item] += 1;
        }

        for (int iter = 0; iter < kAlsIterations; ++iter) {
            std::fill(user_sum.begin(), user_sum.end(), 0.0f);
            for (const Rating& r : batch) {
                user_sum[r.user] += (r.rating - global_mean) - item_bias[r.item];
            }
            for (int u = 0; u < users; ++u) {
                const int c = user_count[u];
                if (c > 0) {
                    user_bias[u] = user_sum[u] / (static_cast<float>(c) + kAlsUserAlpha);
                }
            }

            std::fill(item_sum.begin(), item_sum.end(), 0.0f);
            for (const Rating& r : batch) {
                item_sum[r.item] += (r.rating - global_mean) - user_bias[r.user];
            }
            for (int i = 0; i < items; ++i) {
                const int c = item_count[i];
                if (c > 0) {
                    item_bias[i] = item_sum[i] / (static_cast<float>(c) + kAlsItemAlpha);
                }
            }
        }

        for (const Rating& r : batch) {
            const float dev = r.rating - global_mean;
            const int u = r.user;
            const int i = r.item;
            if (user_seen[u] == 0) {
                user_seen[u] = 1;
                user_ema[u] = dev;
            } else {
                user_ema[u] = kEmaKeep * user_ema[u] + kEmaAlpha * dev;
            }
            if (item_seen[i] == 0) {
                item_seen[i] = 1;
                item_ema[i] = dev;
            } else {
                item_ema[i] = kEmaKeep * item_ema[i] + kEmaAlpha * dev;
            }
        }
    }

#ifndef TRACK1_GLOBAL_SHIFT
#define TRACK1_GLOBAL_SHIFT -0.00049520f
#endif
#ifndef TRACK1_DOT_SCALE
#define TRACK1_DOT_SCALE 0.778764f
#endif
#ifndef TRACK1_USER_BIAS_WEIGHT
#define TRACK1_USER_BIAS_WEIGHT 0.850981f
#endif
#ifndef TRACK1_ITEM_BIAS_WEIGHT
#define TRACK1_ITEM_BIAS_WEIGHT 0.817357f
#endif
#ifndef TRACK1_USER_EMA_WEIGHT
#define TRACK1_USER_EMA_WEIGHT 0.160557f
#endif
#ifndef TRACK1_ITEM_EMA_WEIGHT
#define TRACK1_ITEM_EMA_WEIGHT 0.133570f
#endif

    static constexpr int kAlsIterations = 3;
    static constexpr float kAlsUserAlpha = 50.0f;
    static constexpr float kAlsItemAlpha = 5.0f;
    static constexpr float kEmaAlpha = 0.08f;
    static constexpr float kEmaKeep = 1.0f - kEmaAlpha;
    static constexpr float kGlobalShift = TRACK1_GLOBAL_SHIFT;
    static constexpr float kDotScale = TRACK1_DOT_SCALE;
    static constexpr float kUserBiasWeight = TRACK1_USER_BIAS_WEIGHT;
    static constexpr float kItemBiasWeight = TRACK1_ITEM_BIAS_WEIGHT;
    static constexpr float kUserEmaWeight = TRACK1_USER_EMA_WEIGHT;
    static constexpr float kItemEmaWeight = TRACK1_ITEM_EMA_WEIGHT;

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr;
    float* Q = nullptr;
    const std::vector<Rating>* incremental = nullptr;
    bool prepared = false;
    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_bias;
    std::vector<float> item_bias;
    std::vector<float> user_ema;
    std::vector<float> item_ema;
    std::vector<unsigned char> user_seen;
    std::vector<unsigned char> item_seen;
};

/*
 * fomo_postprocess.c -- see fomo_postprocess.h for documentation.
 */

#include "fomo_postprocess.h"

static inline float dequantize_int8(int8_t q, float scale, int32_t zero_point) {
    return ((float)((int32_t)q - zero_point)) * scale;
}

/*
 * 3x3 local-maxima suppression with replicate padding.
 * Out-of-bounds neighbours are treated as the edge value.
 */
static int local_maxima_3x3(const float *prob, int g,
                            float *keep_mask /* g*g, 1.0 if local max */) {
    for (int y = 0; y < g; ++y) {
        for (int x = 0; x < g; ++x) {
            float center = prob[y * g + x];
            float maxv   = center;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = y + dy;
                if (ny < 0)      ny = 0;
                else if (ny >= g) ny = g - 1;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    if (nx < 0)      nx = 0;
                    else if (nx >= g) nx = g - 1;
                    float v = prob[ny * g + nx];
                    if (v > maxv) maxv = v;
                }
            }
            keep_mask[y * g + x] = (center >= maxv) ? 1.0f : 0.0f;
        }
    }
    return 0;
}

int fomo_postprocess_count_with_map(const int8_t *raw_int8,
                                    const fomo_postprocess_config_t *cfg,
                                    float *out_prob_map,
                                    int *out_count) {
    if (raw_int8 == NULL || cfg == NULL || out_count == NULL || out_prob_map == NULL) {
        return -1;
    }
    if (cfg->grid_size <= 0 || cfg->grid_size > FOMO_MAX_GRID) {
        return -2;
    }

    const int g = cfg->grid_size;
    const int n = g * g;

    /* Step 1: dequantize INT8 -> float probability in [0, 1]. */
    for (int i = 0; i < n; ++i) {
        out_prob_map[i] = dequantize_int8(raw_int8[i], cfg->output_scale,
                                          cfg->output_zero_point);
    }

    /* Step 2 + 3: NMS + threshold + count. */
    int count = 0;
    if (cfg->use_nms) {
        float keep_mask[FOMO_MAX_GRID * FOMO_MAX_GRID];
        local_maxima_3x3(out_prob_map, g, keep_mask);
        for (int i = 0; i < n; ++i) {
            if (keep_mask[i] > 0.5f && out_prob_map[i] >= cfg->threshold) {
                ++count;
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            if (out_prob_map[i] >= cfg->threshold) {
                ++count;
            }
        }
    }

    *out_count = count;
    return 0;
}

int fomo_postprocess_count(const int8_t *raw_int8,
                           const fomo_postprocess_config_t *cfg,
                           int *out_count) {
    if (cfg == NULL) return -1;
    float prob_map[FOMO_MAX_GRID * FOMO_MAX_GRID];
    if (cfg->grid_size <= 0 || cfg->grid_size > FOMO_MAX_GRID) {
        return -2;
    }
    return fomo_postprocess_count_with_map(raw_int8, cfg, prob_map, out_count);
}

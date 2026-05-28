/*
 * fomo_postprocess.h
 *
 * Drop-in post-processing for the keras_fomo grid detector running on
 * the Nuvoton M55M1 (Cortex-M55 + Ethos-U55) via TensorFlow Lite Micro.
 *
 * Pipeline:
 *   model output  : INT8 tensor of shape (1, G, G, 1)  (default G = 6)
 *   step 1        : dequantize INT8 -> float32 probability in [0, 1]
 *   step 2        : 3x3 local-maxima suppression (replicate-pad)
 *   step 3        : threshold and count surviving cells
 *
 * Drop both fomo_postprocess.h and fomo_postprocess.c into the firmware
 * project (the Nuvoton ML_M55M1_SampleCode TFLM example is the recommended
 * starting point) and call fomo_postprocess_count() once per frame.
 *
 * No malloc, no FPU-only code paths, no recursion. Stack usage is bounded
 * by FOMO_MAX_GRID * FOMO_MAX_GRID floats (default 12 * 12 * 4 = 576 B).
 */

#ifndef FOMO_POSTPROCESS_H_
#define FOMO_POSTPROCESS_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time upper bound on supported grid size. Increase if you train
 * a model with a larger grid (e.g. 12 or 16). */
#ifndef FOMO_MAX_GRID
#define FOMO_MAX_GRID 16
#endif

typedef struct {
    /* Quantization parameters of the model output tensor. Read these from
     * the TfLiteTensor at runtime: tensor->params.scale / .zero_point. */
    float   output_scale;
    int32_t output_zero_point;

    /* Detection threshold on the dequantized probability. 0.30 worked
     * best on val for the keras_fomo baseline; tune per-deployment. */
    float   threshold;

    /* Square grid side, e.g. 6 for the default model. */
    int     grid_size;

    /* If non-zero, apply 3x3 local-maxima suppression before thresholding
     * (default behaviour to avoid double-counting adjacent cells). */
    uint8_t use_nms;
} fomo_postprocess_config_t;

/* Convenience initialiser matching the trained keras_fomo baseline. */
static inline fomo_postprocess_config_t fomo_postprocess_default_config(
        float output_scale, int32_t output_zero_point) {
    fomo_postprocess_config_t cfg = {
        .output_scale       = output_scale,
        .output_zero_point  = output_zero_point,
        .threshold          = 0.30f,
        .grid_size          = 6,
        .use_nms            = 1,
    };
    return cfg;
}

/*
 * Run post-processing.
 *
 * @param raw_int8  pointer to the model's INT8 output buffer (length
 *                  cfg->grid_size * cfg->grid_size).
 * @param cfg       configuration. grid_size must be <= FOMO_MAX_GRID.
 * @param out_count [out] estimated number of people (>= 0).
 *
 * @return 0 on success, negative on argument error.
 */
int fomo_postprocess_count(const int8_t *raw_int8,
                           const fomo_postprocess_config_t *cfg,
                           int *out_count);

/*
 * Variant that also returns the dequantized probability map (caller-owned
 * buffer of cfg->grid_size * cfg->grid_size floats). Useful for debugging
 * over UART / on a small display.
 */
int fomo_postprocess_count_with_map(const int8_t *raw_int8,
                                    const fomo_postprocess_config_t *cfg,
                                    float *out_prob_map,
                                    int *out_count);

#ifdef __cplusplus
}
#endif

#endif  /* FOMO_POSTPROCESS_H_ */

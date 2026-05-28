/*
 * test_fomo_postprocess.c -- host-side unit test for fomo_postprocess.
 *
 * Build & run on your dev machine (gcc, no MCU required):
 *
 *     gcc -std=c11 -O2 -Wall -Wextra \
 *         firmware/fomo_postprocess/fomo_postprocess.c \
 *         firmware/fomo_postprocess/test_fomo_postprocess.c \
 *         -o /tmp/test_fomo && /tmp/test_fomo
 *
 * Expected output:
 *     [test] empty grid                 PASS (count=0)
 *     [test] single peak                PASS (count=1)
 *     [test] two well-separated peaks   PASS (count=2)
 *     [test] adjacent peaks (NMS)       PASS (count=1)
 *     [test] threshold filtering        PASS (count=0)
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fomo_postprocess.h"

/* The test pretends scale=1/255, zero_point=-128. With those params,
 * an INT8 value q maps back to (q + 128) / 255, so q=127 -> ~1.0,
 * q=0 -> ~0.5, q=-128 -> 0. This is close to what Vela typically picks
 * for a sigmoid output but the post-processing must work for ANY scale
 * the converter chooses. */
static const float SCALE        = 1.0f / 255.0f;
static const int32_t ZERO_POINT = -128;

static int8_t prob_to_int8(float p) {
    int v = (int)lrintf(p / SCALE + ZERO_POINT);
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

static int run_case(const char *name, const float *probs, int grid, int expected) {
    int8_t q[FOMO_MAX_GRID * FOMO_MAX_GRID];
    for (int i = 0; i < grid * grid; ++i) q[i] = prob_to_int8(probs[i]);

    fomo_postprocess_config_t cfg = fomo_postprocess_default_config(SCALE, ZERO_POINT);
    cfg.grid_size = grid;

    int count = -1;
    int rc = fomo_postprocess_count(q, &cfg, &count);
    int ok = (rc == 0) && (count == expected);
    printf("[test] %-32s %s (count=%d, expected=%d)\n",
           name, ok ? "PASS" : "FAIL", count, expected);
    return ok ? 0 : 1;
}

int main(void) {
    int failures = 0;

    {
        float probs[6 * 6] = {0};
        failures += run_case("empty grid", probs, 6, 0);
    }
    {
        float probs[6 * 6] = {0};
        probs[2 * 6 + 3] = 0.9f;
        failures += run_case("single peak", probs, 6, 1);
    }
    {
        float probs[6 * 6] = {0};
        probs[1 * 6 + 1] = 0.8f;
        probs[4 * 6 + 4] = 0.7f;
        failures += run_case("two well-separated peaks", probs, 6, 2);
    }
    {
        /* Two adjacent cells both above threshold; NMS should keep only one. */
        float probs[6 * 6] = {0};
        probs[2 * 6 + 2] = 0.9f;
        probs[2 * 6 + 3] = 0.7f;  /* lower neighbour -> suppressed by NMS */
        failures += run_case("adjacent peaks (NMS)", probs, 6, 1);
    }
    {
        /* Below threshold -> not counted. */
        float probs[6 * 6] = {0};
        probs[3 * 6 + 3] = 0.20f;  /* below default 0.30 */
        failures += run_case("threshold filtering", probs, 6, 0);
    }

    if (failures) {
        printf("\n%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}

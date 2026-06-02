/**************************************************************************//**
 * @file     FOMOPostProcessing.cpp
 * @version  V1.00
 * @brief    FOMO grid-detector post-processing source file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "FOMOPostProcessing.hpp"
#include "log_macros.h"

namespace arm
{
namespace app
{
namespace fomo
{

FOMOPostProcessing::FOMOPostProcessing(arm::app::FOMOModel *model,
                                       float threshold,
                                       bool  useNms)
    : m_model(model)
    , m_threshold(threshold)
    , m_useNms(useNms)
{
}

static inline float DequantizeInt8(int8_t q, float scale, int32_t zeroPoint)
{
    return static_cast<float>(static_cast<int32_t>(q) - zeroPoint) * scale;
}

/* 3x3 local-maxima with replicate padding. Out-of-range neighbours are
 * clamped to the edge value. */
static void LocalMaxima3x3(const float *prob, int g, float *keepMask)
{
    for (int y = 0; y < g; ++y)
    {
        for (int x = 0; x < g; ++x)
        {
            float center = prob[y * g + x];
            float maxv   = center;

            for (int dy = -1; dy <= 1; ++dy)
            {
                int ny = y + dy;
                if (ny < 0)      ny = 0;
                else if (ny >= g) ny = g - 1;

                for (int dx = -1; dx <= 1; ++dx)
                {
                    int nx = x + dx;
                    if (nx < 0)      nx = 0;
                    else if (nx >= g) nx = g - 1;

                    float v = prob[ny * g + nx];
                    if (v > maxv) maxv = v;
                }
            }

            keepMask[y * g + x] = (center >= maxv) ? 1.0f : 0.0f;
        }
    }
}

void FOMOPostProcessing::RunPostProcessing(FOMOResult &resultOut)
{
    TfLiteTensor *out = m_model->GetOutputTensor(0);

    /* Output shape is (1, G, G, 1). G is a runtime value but capped by
     * FOMO_MAX_GRID so we don't need dynamic memory. */
    const int g = out->dims->data[1];
    if (g <= 0 || g > FOMO_MAX_GRID)
    {
        printf_err("FOMO: unsupported grid size %d (max %d)\n", g, FOMO_MAX_GRID);
        resultOut.count    = 0;
        resultOut.gridSize = 0;
        return;
    }
    resultOut.gridSize = g;

    const float   scale     = out->params.scale;
    const int32_t zeroPoint = out->params.zero_point;
    const int8_t *raw       = static_cast<const int8_t *>(out->data.data);
    const int     n         = g * g;

    /* Step 1: dequantize int8 -> float probabilities. */
    for (int i = 0; i < n; ++i)
    {
        resultOut.probMap[i] = DequantizeInt8(raw[i], scale, zeroPoint);
    }

    /* Step 2 + 3: optional NMS, then threshold and count. */
    int count = 0;
    if (m_useNms)
    {
        float keepMask[FOMO_MAX_GRID * FOMO_MAX_GRID];
        LocalMaxima3x3(resultOut.probMap, g, keepMask);
        for (int i = 0; i < n; ++i)
        {
            if (keepMask[i] > 0.5f && resultOut.probMap[i] >= m_threshold)
            {
                ++count;
            }
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            if (resultOut.probMap[i] >= m_threshold) ++count;
        }
    }

    resultOut.count = count;
}

} /* namespace fomo */
} /* namespace app */
} /* namespace arm */

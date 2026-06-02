/**************************************************************************//**
 * @file     FOMOPostProcessing.hpp
 * @version  V1.00
 * @brief    FOMO grid-detector post-processing header file
 *
 * Reads the int8 output tensor from the FOMO model, dequantizes it to
 * float probabilities, applies 3x3 local-maxima suppression, and counts
 * how many cells survive the threshold. The count is the estimated number
 * of people in the frame.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef FOMO_POST_PROCESSING_HPP
#define FOMO_POST_PROCESSING_HPP

#include "FOMOModel.hpp"

namespace arm
{
namespace app
{
namespace fomo
{

/* Compile-time upper bound on supported grid size. Bump if you train a
 * model with a larger grid. Stack usage is O(MAX_GRID^2 * 4 bytes). */
constexpr int FOMO_MAX_GRID = 16;

struct FOMOResult
{
    int   count;                              /* number of detected people */
    float probMap[FOMO_MAX_GRID * FOMO_MAX_GRID];
    int   gridSize;                           /* size used for probMap */
};

/**
 * @brief   Helper class to manage tensor post-processing for the FOMO model.
 */
class FOMOPostProcessing
{
public:
    /**
     * @brief       Constructor.
     * @param[in]   model       Pointer to the FOMOModel instance.
     * @param[in]   threshold   Per-cell probability threshold (default 0.30
     *                          which matched the val-best for our baseline).
     * @param[in]   useNms      Apply 3x3 local-maxima suppression (default true).
     **/
    explicit FOMOPostProcessing(arm::app::FOMOModel *model,
                                 float threshold = 0.30f,
                                 bool  useNms    = true);

    /**
     * @brief       Run post-processing on the model's last inference.
     * @param[out]  resultOut   Filled with count + dequantized prob map.
     **/
    void RunPostProcessing(FOMOResult &resultOut);

    /* Runtime tunables (no need to recompile to retune). */
    void SetThreshold(float t)        { m_threshold = t; }
    void SetUseNms(bool b)            { m_useNms    = b; }
    float GetThreshold() const        { return m_threshold; }

private:
    arm::app::FOMOModel *m_model;
    float                m_threshold;
    bool                 m_useNms;
};

} /* namespace fomo */
} /* namespace app */
} /* namespace arm */

#endif /* FOMO_POST_PROCESSING_HPP */

/**************************************************************************//**
 * @file     FOMOModel.hpp
 * @version  V1.00
 * @brief    FOMO grid-detector model header file
 *
 * Adapted from the Nuvoton YOLOv8n object-detection sample for the
 * keras_fomo person-counting model:
 *   * Input  : (1, 192, 192, 3) int8
 *   * Output : (1, 6, 6, 1)     int8 (sigmoid heatmap)
 *
 * After Vela compilation 100% of the ops fold into the ``ethos-u`` custom
 * operator, so the only op the resolver needs to register is EthosU.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef _FOMO_MODEL_HPP_
#define _FOMO_MODEL_HPP_

#include "Model.hpp"

#define FOMO_INPUT_TENSOR        (192)
#define FOMO_GRID_SIZE           (6)
#define FOMO_NUM_CLASSES         (1)

namespace arm
{
namespace app
{

class FOMOModel : public Model
{
public:
    /* Indices for the expected model -- based on input tensor shape */
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    /** @brief   Gets the reference to op resolver interface class. */
    const tflite::MicroOpResolver &GetOpResolver() override;

    /** @brief   Adds operations to the op resolver instance. */
    bool EnlistOperations() override;

private:
    /* Maximum number of individual operations that can be enlisted. */
    static constexpr int ms_maxOpCnt = 1;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* _FOMO_MODEL_HPP_ */

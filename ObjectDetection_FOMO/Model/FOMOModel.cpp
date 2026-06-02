/**************************************************************************//**
 * @file     FOMOModel.cpp
 * @version  V1.00
 * @brief    FOMO grid-detector model source file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "FOMOModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::FOMOModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::FOMOModel::EnlistOperations()
{
#if defined(ARM_NPU)

    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n",
             tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }

#endif /* ARM_NPU */
    return true;
}

# ObjectDetection_FOMO — keras_fomo person counter on M55M1

This sample is adapted from `ObjectDetection_YOLOv8n` to run the
keras_fomo grid detector instead of YOLOv8n.

* Input  : (1, 192, 192, 3) int8, scale 0.0078431, zero_point -1
* Output : (1, 6, 6, 1) int8 sigmoid heatmap
* Vela   : 100% NPU placement, ~19.6 ms / inference on Ethos-U55-256

## Files

| File                                | Purpose                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| `Model/FOMOModel.cpp` / `.hpp`      | Model wrapper, op resolver registers only `EthosU`       |
| `FOMOPostProcessing.cpp` / `.hpp`   | Dequantize + 3x3 NMS + threshold + count                 |
| `main.cpp`                          | Capture loop, inference, count overlay rendering         |
| `Model/model_int8_vela.tflite`      | Vela-compiled FOMO model (load from SD card at boot)     |
| `KEIL/ObjectDetection.uvprojx`      | uVision project — open this in Keil to build             |

The infrastructure files (`BoardInit*`, `Device/`, `NPU/`, `Pattern/`,
`ModelFileReader.*`, `mpu_config_M55M1.h`, `ffconf_M55M1.h`,
`board_config.h`) are unchanged from the YOLOv8n sample.

## Setup

1. Install `M55M1BSP-3.01.003` (the parent `install.py` does this for you).
2. Copy `Model/model_int8_vela.tflite` to the **root of the SD card**.
   At boot the firmware loads it from `0:\\model_int8_vela.tflite` into
   HyperRAM at `0x82400000`.
3. Open `KEIL/ObjectDetection.uvprojx` in Keil uVision5, build, and
   flash via Nu-Link.
4. Open a 115200-baud serial terminal — you will see lines like:

       Added ETHOS-U support to op resolver
       Model file size 2496160
       [fomo] people = 2
       [fomo] people = 0

5. The LCD shows the camera frame with the detected count in red and a
   green box around every active grid cell.

## Tuning

Edit one constant in `main.cpp` to retune without retraining:

```c
#define FOMO_THRESHOLD (0.30f)   // 0.30 was best-on-val for our baseline
```

If the model double-counts (1 person seen as 2 in adjacent cells),
either raise the threshold or keep `useNms=true` (default).

## Going back to YOLOv8

The original `ObjectDetection_YOLOv8n` sample is still present alongside
this one. They share no source files — switching back is just a matter
of opening the other folder's KEIL project.

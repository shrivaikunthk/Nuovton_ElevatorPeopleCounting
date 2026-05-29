# Firmware deployment (Windows + Keil + Nu-Link)

End-to-end guide for building, flashing, and running the FOMO
person-counting firmware on the Nuvoton M55M1 board.

This document is the **deployment** half of the repo. For the model
training/eval side see [`README.md`](README.md) and
[`model_work/MODEL_WORK.md`](model_work/MODEL_WORK.md).

## What you need from this repo

```
.
├── FIRMWARE_DEPLOYMENT.md        <-- you are here
├── ObjectDetection_FOMO/         <-- the firmware project (drop into BSP)
│   ├── KEIL/ObjectDetection.uvprojx     <-- open this in Keil uVision5
│   ├── main.cpp
│   ├── FOMOPostProcessing.{hpp,cpp}
│   ├── Model/FOMOModel.{hpp,cpp}
│   ├── Model/model_int8_vela.tflite     <-- the trained NPU-ready model
│   └── ... (all supporting infra files)
├── sd_card/
│   └── model_int8_vela.tflite    <-- copy this file to the SD card root
├── compile_vela.bat              <-- only needed if you retrain
└── model_work/                   <-- training results, metrics, plots
    ├── MODEL_WORK.md             <-- full write-up of both models
    ├── fomo/                     <-- FOMO config, sweep, test eval, int8 tflite
    └── yolo/                     <-- YOLOv8n training curves, weights, plots
```

For the training results, accuracy metrics, and inference outputs of
both models, read **`model_work/MODEL_WORK.md`**.

## What you need to install on Windows

| Tool                              | Purpose                              | Free? | Required? |
| --------------------------------- | ------------------------------------ | ----- | --------- |
| **Keil uVision5 MDK-Arm**         | Builds the firmware                  | Free Community edition or Nuvoton-keyed full version | yes |
| **NuMicro ICP Programming Tool**  | Flashes via Nu-Link                  | Yes (Nuvoton website) | yes |
| **Python 3.10+**                  | Only if you retrain models           | Yes | no |
| **VS Code**                       | Comfy editor for .cpp/.hpp files     | Yes | optional |

You do **not** need:
- Vivado / Vitis (those are Xilinx FPGA tools, irrelevant to ARM MCUs)
- WSL / git-bash (everything is native Windows)
- An ARM GCC toolchain (Keil's ARMClang is built-in)

### Download links

- **Keil MDK Community**: <https://www.keil.arm.com/mdk-community/>
  When installing, make sure you add support for **Cortex-M55** (it's in
  the device family pack).
- **NuMicro ICP Programming Tool**:
  <https://www.nuvoton.com/tool-and-software/software-development-tool/programmer/>
  Look for "NuMicro ICP Programming Tool".
- **Nu-Link Driver** (USB driver for the on-board debugger): same
  Nuvoton page as above.

## Step-by-step: from zero to live inference

### Step 1 -- Copy this bundle to Windows

Zip the **whole repository** (not just this folder) and copy it over.
You also need the file `ML_M55M1_SampleCode-master/` in the repo root
because it contains the M55M1 BSP that the Keil project links against.

If you only want the minimum:

1. Copy `windows_deployment/` (this folder) to Windows.
2. On Windows, download `ML_M55M1_SampleCode-master.zip` directly from
   <https://github.com/OpenNuvoton/ML_M55M1_SampleCode> and unzip it.
3. Place the unzipped folder next to `windows_deployment/`, so both
   live at the same level.

### Step 2 -- Expand the M55M1 BSP

```cmd
cd ML_M55M1_SampleCode-master
python install.py
```

`install.py` downloads the BSP zip from GitHub and unpacks it inside
`M55M1BSP-3.01.003/`. After this, the folder
`M55M1BSP-3.01.003/Library/` exists (it does not before `install.py`
runs).

### Step 3 -- Drop our sample into the BSP tree

Copy `windows_deployment/ObjectDetection_FOMO/` into:

```text
ML_M55M1_SampleCode-master/M55M1BSP-3.01.003/SampleCode/NuEdgeWise/
```

So you end up with:

```text
ML_M55M1_SampleCode-master/
  M55M1BSP-3.01.003/
    SampleCode/
      NuEdgeWise/
        ObjectDetection_FOMO/      <-- you just copied this
        ObjectDetection_YOLOv8n/   <-- already there
        ...
```

The Keil project uses **relative paths** (`..\..\..\..\Library\...`) so
it MUST live at this exact depth.

### Step 4 -- Open Keil and build

1. Launch Keil uVision5.
2. `File -> Open Project` -> navigate to
   `...\NuEdgeWise\ObjectDetection_FOMO\KEIL\ObjectDetection.uvprojx`
3. Wait for Keil to index. Confirm the **target device is Nuvoton M55M1**
   in the toolbar.
4. Press **F7** (Build). First build takes 3-5 minutes because it
   compiles the TFLM library too.

Expected at the end of the build:

```text
"ObjectDetection\Objects\ObjectDetection.axf" - 0 Error(s), N Warning(s)
```

A handful of warnings from third-party code (TFLM, openmv) is normal.

### Step 5 -- Prepare the SD card

1. Format a microSD card as **FAT32** (any size, the model is 2.5 MB).
2. Copy `windows_deployment/sd_card/model_int8_vela.tflite` to the
   **root** of the SD card (not inside a folder).
3. Eject the card and insert it into the M55M1 board's SD slot.

### Step 6 -- Flash the firmware

1. Connect the M55M1 board to your PC via the Nu-Link USB port. The
   first time you do this, Windows will install the Nu-Link driver
   automatically (if you installed it in step 0).
2. In Keil, press **F8** (or `Flash -> Download`). The Nu-Link green
   LED blinks while flashing.
3. After flashing, press the **reset** button on the board.

### Step 7 -- Watch it run

Open a serial terminal (PuTTY, TeraTerm, or the one built into VS Code):

- Port: Nu-Link Virtual COM (check Device Manager for the COM number)
- Baud: **115200**
- Data/Parity/Stop: 8-N-1, no flow control

You should see:

```text
Set tensor arena cache policy to WTRA
Added ETHOS-U support to op resolver
Model file size 2496160
Initialised model
[fomo] people = 0
[fomo] people = 0
[fomo] people = 2
Total inference rate: 50
```

On the LCD (if your board has one connected): the live camera frame,
the count `people=N` in red at the top-left, and a green box on every
active grid cell.

## Tuning without re-flashing

Three knobs you can change in `main.cpp` and re-flash (1-minute cycle):

```c
#define FOMO_THRESHOLD (0.30f)    // 0.30 = best-on-val; raise = stricter
```

```cpp
// in main():
arm::app::fomo::FOMOPostProcessing postProcess(&model, FOMO_THRESHOLD, true);
//                                                                     ^^^^
//                                                          false = disable NMS
```

To swap the model entirely (e.g. after retraining), just copy a new
`model_int8_vela.tflite` onto the SD card. The firmware reads it at
boot -- no re-flash needed.

## Troubleshooting

| Symptom                                              | Likely cause                                           | Fix                                                                                              |
| ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Keil error: cannot open `Library\...` files          | `install.py` was not run, or folder is at wrong depth  | Re-check Step 2 + Step 3                                                                         |
| Keil error: `FOMOModel.hpp` not found                 | Include path broken                                    | Project Properties -> C/C++ -> Include Paths must contain `..\Model\include`                     |
| Build OK, but boot prints `Failed to prepare model`  | SD card not detected or model file misnamed            | File on SD root must be exactly `model_int8_vela.tflite` (case-sensitive on the M55M1 FAT layer) |
| `Failed to initialise model`                         | Tensor arena too small                                 | KEIL project -> C/C++ Defines -> change `ACTIVATION_BUF_SZ=0x00100000` to `0x00200000`, rebuild  |
| Serial prints `people = 36` constantly               | Input pre-processing wrong (no `-128` shift)           | Verify `main.cpp` still has `signed_req_data[i] = (int8_t)req_data[i] - 128;`                    |
| Serial prints `people = 0` always                    | Camera not RGB, or threshold too high                  | Try `FOMO_THRESHOLD = 0.10f`. If still 0, camera config is wrong                                  |
| Build succeeds but Keil can't flash                  | Nu-Link driver not installed                           | Install Nu-Link driver from Nuvoton site; reconnect USB                                          |

## Re-training and re-deploying

If you retrain the model on Windows (or anywhere), the redeploy flow is:

```cmd
REM 1. Re-run Vela on the new int8 model
compile_vela.bat keras_fomo\runs\fomo\model_int8.tflite

REM 2. Copy the result to the SD card root
copy keras_fomo\runs\fomo\vela\model_int8_vela.tflite E:\

REM 3. Eject the SD card, plug it into the board, reset.
REM    NO RE-FLASH NEEDED -- the firmware loads the new model at boot.
```

That's it -- the firmware itself does not need to be rebuilt unless you
change post-processing logic.

## What if I want to use VS Code instead of Keil?

You can edit the source files in VS Code, but **building requires an
ARM compiler**. The options are:

1. **Recommended**: edit in VS Code, build in Keil (just have both open).
2. **Advanced**: set up a `gcc-arm-none-eabi` toolchain + CMake. The
   Nuvoton BSP ships with Keil-only project files, so you would have to
   author your own CMakeLists.txt. Not recommended unless you really
   dislike Keil.

## Confirmed working

The host-side parts of this pipeline have been verified end-to-end on
the dev machine:

- Vela compile: 100% NPU placement, 19.6 ms / inference
- C post-processing: 5/5 host unit tests PASS
- Model size on disk: 2.5 MB

The on-device side (Keil build + flash + camera capture) cannot be
tested without the physical board, so report any issues with the serial
log and we'll fix them together.
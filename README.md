# Nuvoton M55M1 person counter — models, training, and deployment

Documentation repository for the two models trained for the **Nuvoton
M55M1 + Ethos-U55-256 NPU** overhead person-counting application.

* **FOMO** (production) — a 6 x 6 grid heatmap detector. 100% NPU
  placement after Vela. 19.6 ms / inference. Test count MAE = 0.42
  people. **This is the model that runs on the board.**
* **YOLOv8n-ReLU6** (reference) — full bounding-box detector. Higher
  fidelity (val mAP50 = 0.978) but ~6x slower on the NPU. Kept as a
  proof of dataset quality and as a fallback if the use case ever
  changes to per-person localisation.


## Repo layout

```
.
├── README.md                       <-- you are here
├── docs/                           <-- long-form documentation
│   ├── SETUP.md                    Environment + tooling install
│   ├── DATASETS.md                 Dataset prep, splits, labels
│   ├── TRAINING.md                 Training recipes (FOMO + YOLO)
│   ├── EVALUATION.md               How to score a trained model
│   ├── BestTraining.md             FOMO best-run analysis
│   ├── Nuvoton_YOLOv8_BestRun.md   YOLO best-run analysis
│   ├── DEPLOYMENT_M55M1.md         On-device deployment guide
│   └── TROUBLESHOOTING.md          Common failure modes
│
├── keras_fomo/                     <-- FOMO Python package
│   ├── data.py                     Dataset loader + augmentation
│   ├── model.py                    MobileNetV2 -> 6x6 sigmoid head
│   ├── train.py                    Training loop entrypoint
│   ├── evaluate.py                 Eval (count MAE, threshold sweep)
│   ├── requirements.txt            Python dependencies
│   └── README.md                   Package usage
│
├── scripts/                        <-- runnable scripts (Linux/macOS)
│   ├── train_fomo.py               Train FOMO end-to-end
│   ├── evaluate_fomo.py            Score a trained FOMO model
│   ├── sweep_threshold_fomo.py     Pick deployment threshold
│   ├── compare_fomo_runs.py        Diff metrics across runs
│   ├── run_fomo_experiments.sh     Batch sweep (multiple configs)
│   ├── train_nuvoton_yolo.sh       Train YOLOv8n-ReLU6
│   ├── prepare_nuvoton_yolo_dataset.py   Build YOLO-format dataset
│   ├── evaluate_nuvoton_yolo.py    Score the YOLO model
│   ├── build_splits.py             Reproduce train/val/test splits
│   ├── inspect_dataset.py          Sanity-check loaded data
│   ├── export_label_previews.py    Render ground-truth previews
│   ├── compile_vela.sh             Vela compile for Ethos-U55-256
│   ├── gen_model_header.py         Wrap a .tflite as C array
│   ├── train_baseline.py           Naive baseline (no NPU specifics)
│   └── evaluate_baseline.py        Baseline scoring
│
├── firmware/                       <-- on-device C code
│   └── fomo_postprocess/           Post-processing (dequant + NMS + count)
│       ├── fomo_postprocess.h
│       ├── fomo_postprocess.c
│       └── test_fomo_postprocess.c Host-side unit tests (5 PASS expected)
│
├── model_work/                     <-- training outputs (committed for reference)
│   ├── MODEL_WORK.md               Full write-up of both runs
│   ├── fomo/                       FOMO config, sweep, eval, int8 .tflite
│   └── yolo/                       YOLO args, results, plots, best.pt
│
├── ObjectDetection_FOMO/           <-- Keil firmware project (the actual binary)
│   └── ... (drop into Nuvoton BSP NuEdgeWise/, see FIRMWARE_DEPLOYMENT.md)
├── sd_card/model_int8_vela.tflite  <-- copy to SD card for the board
├── compile_vela.bat                <-- Windows version of compile_vela.sh
└── FIRMWARE_DEPLOYMENT.md          <-- step-by-step Windows + Keil setup
```

## Reproducing the models

### 0. Get the data

The training data is **not in this repo** (it is fetched from source).
You need:

* Roboflow project **overhead-person-detection** (single class: person).
* See [`docs/DATASETS.md`](docs/DATASETS.md) for download URL + the
  exact prep steps (resize to 192 x 192, train/val/test = 80/10/10).
* Place the prepared dataset at `./overhead-person-detection/` at the
  repo root (or point `--dataset_root` at wherever you put it).

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r keras_fomo/requirements.txt
```

Plus `ethos-u-vela==3.10.0` if you want to compile for the NPU. Full
tooling matrix lives in [`docs/SETUP.md`](docs/SETUP.md).

### 2. Train FOMO

```bash
python scripts/train_fomo.py \
    --dataset_root overhead-person-detection \
    --output_dir keras_fomo/runs/fomo \
    --epochs 30 --batch_size 32 --lr 1e-3 \
    --image_size 192 --grid_size 6 --seed 42
```

Outputs land in `keras_fomo/runs/fomo/` (gitignored):
`best.keras`, `final.keras`, `model_float32.tflite`, `model_int8.tflite`,
`training_log.csv`, `config.json`.

### 3. Pick the deployment threshold

```bash
python scripts/sweep_threshold_fomo.py \
    --model keras_fomo/runs/fomo/best.keras \
    --thresholds 0.30 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80
```

This is what produced `model_work/fomo/sweep_best.json`. The winner
on our val set was **0.30** (count MAE 0.408).

### 4. Score on test

```bash
python scripts/evaluate_fomo.py \
    --model keras_fomo/runs/fomo/best.keras \
    --split test --threshold 0.30 \
    --output_dir keras_fomo/runs/fomo/eval_test_best
```

Matches `model_work/fomo/eval_test_best/summary.json`: test count MAE
**0.418**, RMSE 1.04, empty-image FP rate **3.0%**.

### 5. Vela-compile for the NPU

```bash
bash scripts/compile_vela.sh keras_fomo/runs/fomo/model_int8.tflite
# or on Windows:
compile_vela.bat keras_fomo\runs\fomo\model_int8.tflite
```

Output: `keras_fomo/runs/fomo/vela/model_int8_vela.tflite`. This is
the file you drop on the SD card. Vela report shows **100% NPU
placement, ~19.6 ms / inference**.

### 6. Train the reference YOLO (optional)

```bash
bash scripts/train_nuvoton_yolo.sh
# Uses Nuvoton's ReLU6 YOLOv8n fork:
#   https://github.com/OpenNuvoton/ML_YOLO
```

Hyperparameters that produced `model_work/yolo/`:
300 epochs, batch=64, imgsz=192, SGD lr0=0.01, patience=30.

Final scores in `model_work/yolo/results.csv`:
**precision 0.949, recall 0.941, mAP50 0.978, mAP50-95 0.686**.

### 7. Verify the post-processing on the host

```bash
gcc -std=c11 -Wall -Wextra \
    firmware/fomo_postprocess/test_fomo_postprocess.c \
    firmware/fomo_postprocess/fomo_postprocess.c \
    -o /tmp/test_fomo -lm
/tmp/test_fomo
# Expect 5 PASS lines.
```

## Deployment

```text
1. Drop ObjectDetection_FOMO/ into the Nuvoton BSP at
   M55M1BSP-3.01.003/SampleCode/NuEdgeWise/
2. Open KEIL/ObjectDetection.uvprojx, build (F7), flash (F8)
3. Copy sd_card/model_int8_vela.tflite to SD card root
4. Reset board, watch [fomo] people=N on the serial terminal.
```

## Provenance

| Artefact                                     | Produced by                                       | Documented in                          |
| -------------------------------------------- | ------------------------------------------------- | -------------------------------------- |
| `model_work/fomo/training_log.csv`           | `scripts/train_fomo.py`                           | `model_work/MODEL_WORK.md` §1          |
| `model_work/fomo/sweep_best.json`            | `scripts/sweep_threshold_fomo.py`                 | `model_work/MODEL_WORK.md` §1          |
| `model_work/fomo/eval_test_best/`            | `scripts/evaluate_fomo.py` (or keras_fomo/evaluate.py) | `model_work/MODEL_WORK.md` §1     |
| `model_work/fomo/model_int8.tflite`          | `scripts/train_fomo.py` (post-train quantization) | `docs/TRAINING.md`                     |
| `sd_card/model_int8_vela.tflite`             | `scripts/compile_vela.sh`                         | `docs/DEPLOYMENT_M55M1.md`             |
| `model_work/yolo/*`                          | `scripts/train_nuvoton_yolo.sh`                   | `docs/Nuvoton_YOLOv8_BestRun.md`       |
| `ObjectDetection_FOMO/`                      | Hand-adapted from `ObjectDetection_YOLOv8n` sample | `FIRMWARE_DEPLOYMENT.md`               |

## License

Code in this repo is dual-licensed:

* Files **adapted from the Nuvoton ML_M55M1_SampleCode** (everything in
  `ObjectDetection_FOMO/` except `FOMOModel.*` and `FOMOPostProcessing.*`)
  remain under the original Nuvoton license — see file headers.
* Files **authored here** (`keras_fomo/`, `scripts/`, `firmware/`,
  `FOMOModel.*`, `FOMOPostProcessing.*`, all `docs/` and `*.md` files):
  Apache-2.0.



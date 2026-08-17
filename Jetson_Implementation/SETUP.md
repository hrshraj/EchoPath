# EchoPath — Jetson Implementation

`EchoPath_Jetson.ipynb` runs on **both** machines. Cell 1 detects which, sets `BACKEND`,
and everything downstream follows.

|                | Workstation        | Jetson Nano (Eagle 101)        |
|----------------|--------------------|--------------------------------|
| `BACKEND`      | `torch`            | `tensorrt`                     |
| Python         | 3.11               | **3.6** (JetPack 4.6 ceiling)  |
| Detector       | Ultralytics `best.pt` | TensorRT `.plan` (FP16)     |
| Depth          | MiDaS via `torch.hub` | TensorRT `.plan` (FP16)     |
| Input          | 13 video files     | CSI camera                     |
| Output         | annotated + voiced `.mp4` | spoken alerts only      |

Detection, depth and fusion logic are identical on both. Only the backend differs —
which is what makes a workstation-vs-device comparison meaningful later.

---

## Cell map

| Cells | Purpose | Runs on |
|---|---|---|
| 1–4 | Probe, paths, models, fusion + `AlertPolicy` | **both** |
| 5 | Depth calibration plots | workstation |
| 6 | Rotation diagnostic | workstation |
| 7 | Batch-process the 13 videos | workstation |
| 8 | Alert summary + **policy audit** | workstation |
| 9 | `TTSSpeaker` | **both** |
| 10 | Narrate a processed video | workstation |
| 11 | Bake voice into `.mp4` | workstation |
| **12** | **Live CSI camera runtime** | **device** |

On the Jetson the path is **1 → 4, 9, 12**. Cells 5–11 need matplotlib, video files or
ffmpeg, none of which belong on the board.

---

## Workstation setup

```bash
# conda manages the environment; pip installs into it.
#
# --override-channels is required on this machine: .condarc lists Anaconda's
# `defaults` channels, and `conda env create -f environment.yml` refuses to run until
# their Terms of Service are accepted. Using conda-forge alone sidesteps that, and we
# only need python + pip + ipykernel from conda anyway.
conda create -p .\env -y --override-channels -c conda-forge python=3.11 pip ipykernel jupyter

# CUDA 11.8 torch is not on PyPI - take it from the PyTorch index first
.\env\python.exe -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118

.\env\python.exe -m pip install -r requirements-workstation.txt
```

Activate by **path**, not by name — it is a prefix environment living inside this folder:

```bash
conda activate E:\EchoPath\Jetson_Implementation\env
```

Call `env\python.exe` directly rather than going through `conda run -p`, which threw an
internal error on this machine.

Then run the notebook top to bottom. Outputs land in `output/`.

> Do not add `--index-url` to the top of a requirements file. It applies to *every*
> package in it, and `ultralytics` / `pyttsx3` / `seaborn` are not on the PyTorch index.
> That is why the repository's original top-level `requirements.txt` fails.

---

## Jetson setup

### 1 · Flash

The production module has **16 GB eMMC**, and a full JetPack 4.6 install is ~14 GB —
before swap and engines. **Put the root filesystem on the 128 GB microSD.**

You cannot flash this from Windows. The eMMC module has no SD-card image; it flashes
over **USB recovery mode** through the Type-B port from an **x86_64 host running Ubuntu
18.04 or 20.04**. Use bare metal or a dual boot, not a VM — the board re-enumerates on
the USB bus mid-flash and VM passthrough tends to drop it.

Use **Tanna TechBiz's BSP**, not stock L4T. A third-party carrier differs in pin mux,
CSI lane routing and fan PWM; stock L4T boots to a console and then gives you no
cameras and no fan. The QR code on the retail box links to the flashing guide.

### 2 · Prepare the OS

```bash
sudo systemctl set-default multi-user.target      # headless: frees ~1 GB of 4 GB
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile    # add to /etc/fstab
sudo nvpmodel -m 0 && sudo jetson_clocks          # MAXN + pinned clocks
sudo apt install -y espeak libespeak1 alsa-utils
```

### 3 · Python environment

```bash
python3 -m venv --system-site-packages ~/echopath-env
source ~/echopath-env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_ROOT=/usr/local/cuda
pip install -r requirements-jetson.txt

python3 -c "import tensorrt, pycuda.autoinit, cv2, numpy; print(tensorrt.__version__)"
python3 -c "import cv2; print('GStreamer' in cv2.getBuildInformation())"   # must be True
```

`--system-site-packages` is the whole trick: the venv inherits JetPack's `tensorrt` and
`cv2` while still giving you somewhere to put `pycuda`.

### 4 · Build the engines

Copy `onnx/*.onnx` from the workstation, then build **on the board** — a `.plan` is not
portable across TensorRT version or GPU architecture.

```bash
mkdir -p engines
/usr/src/tensorrt/bin/trtexec --onnx=onnx/yolov8n_640.onnx --fp16 --verbose \
    --workspace=512 --saveEngine=engines/yolo_640_fp16.plan
/usr/src/tensorrt/bin/trtexec --onnx=onnx/midas_small_256.onnx --fp16 \
    --workspace=512 --saveEngine=engines/midas_256_fp16.plan
```

FP16, not INT8 — Maxwell is compute capability 5.3 and has no DP4A. `--int8` warns and
silently falls back, so the engine *looks* quantised and performs like FP16. Check the
`--verbose` log for layers that fell back to FP32.

Build them one at a time. Concurrent builds will OOM in 4 GB even with swap.

### 5 · Verify hardware independently, before running anything

Debugging a full pipeline on an unproven board conflates software bugs with device-tree
problems. Prove each piece on its own first.

```bash
# camera
gst-launch-1.0 nvarguscamerasrc num-buffers=300 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvvidconv ! fakesink -v

# audio — the Eagle 101 has NO analog codec, so a USB dongle is mandatory
aplay -l
aplay -D default /usr/share/sounds/alsa/Front_Center.wav
```

### 6 · Run

Open the notebook, run cells **1–4 and 9**, then set `RUN_LIVE = True` in cell 12.

---

## Traps

| | Why |
|---|---|
| **Never `pip install opencv-python`** | Shadows JetPack's build, which has no GStreamer → the CSI pipeline silently stops opening, and nothing points back at the pip install |
| **Engines are not portable** | Built on the workstation, they will not deserialise here. ONNX is the portable artefact |
| **No INT8** | Maxwell SM 5.3 lacks DP4A. Input resolution is the only latency dial left |
| **Keep `YOLO_IMGSZ = 640`** | At 416 the letterbox loses up to 52 % of detections on 1080×1920 portrait input. mAP on the validation *images* says otherwise — it does not have that geometry |
| **`CLASS_NAMES` must have 19 entries** | The `.plan` carries weights but not names. Using the paper's 17 shifts every index above 10, and `wet_floor_sign` would report as `monitor` — which is silenced, so a wet-floor hazard produces no alert. Cell 3 asserts this at load |
| **No audio hardware** | USB dongle required. Avoid Bluetooth: A2DP adds 100–200 ms straight onto the alert path |

---

## Status

**Working and verified:** the workstation path end to end — 13 videos, 13 annotated +
13 voiced outputs, and the alert-policy audit passing (min gap 3.00 s, cooldown 3.07 s).

**Written but never executed:** the TensorRT backend and the live camera loop. There is
no board yet. Expect first-contact debugging around binding names and whether the YOLO
output tensor arrives as `(1, 23, anchors)` or transposed.

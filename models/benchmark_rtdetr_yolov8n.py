"""Controlled YOLOv8n vs RT-DETR-L benchmark (reviewer comment 3.5).

Same machine, software, input size and frames for every model:
  * speed: 390 frames (30 per video) from the 13 Academic Block videos, decoded and rotated beforehand so that
    decoding and drawing are not timed; 640 px input, batch 1, confidence 0.40, FP32 and FP16; 50 warm-up
    predictions; GPU synchronised around every timed call; two rounds with the models alternating, the second
    reported. Ultralytics' own preprocess / inference / postprocess split is recorded as well.
  * accuracy: the v4 validation split (2,922 images), restricted to the five classes that COCO shares with the
    EchoPath scheme (person, chair, potted plant, table = dining table, monitor = tv). The stock COCO models are
    scored on labels remapped to COCO ids; the fine-tuned EchoPath model on its own ids. Neither stock model has
    seen these images.
Models: stock YOLOv8n and RT-DETR-L (both COCO-pretrained, the weights used in the original comparison) and the
fine-tuned EchoPath YOLOv8n (v4). RT-DETR is loaded with the RTDETR class so that its own post-processing is used.

Run from E:\\EchoPath:  .venv\\Scripts\\python models\\benchmark_rtdetr_yolov8n.py
"""
import json, os, time, glob
import numpy as np, cv2, torch, yaml
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops

ROOT = r"E:\EchoPath"
VIDEO_DIR = os.path.join(ROOT, "Video_Data")
VAL_DIR = os.path.join(ROOT, "dataset_merged_v4", "valid")
OUT_DIR = os.path.join(ROOT, "models", "benchmark_results")
MODELS = {  # name: (loader, weights)
    "YOLOv8n (EchoPath v4, fine-tuned)": (YOLO, os.path.join(ROOT, r"runs\detect\runs\echopath\v4\weights\best.pt")),
    "YOLOv8n (COCO)": (YOLO, os.path.join(ROOT, "yolov8n.pt")),
    "RT-DETR-L (COCO)": (RTDETR, os.path.join(ROOT, "rtdetr-l.pt")),
}
FRAMES_PER_VIDEO, WARMUP, IMGSZ, CONF = 30, 50, 640, 0.40
OUR_TO_COCO = {0: 0, 1: 56, 11: 58, 15: 60, 16: 62}     # person, chair, potted_plant, table, monitor


def load_frames():
    frames = []
    for v in sorted(os.listdir(VIDEO_DIR)):
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, v))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(np.linspace(0, n - 1, FRAMES_PER_VIDEO).astype(int).tolist())
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(f)
            i += 1
        cap.release()
    return frames


def stats(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), mean=float(x.mean()))


def time_model(model, frames, half):
    kw = dict(imgsz=IMGSZ, conf=CONF, half=half, device=0, verbose=False)
    for f in frames[:WARMUP]:
        model.predict(f, **kw)
    total, pre, inf, post = [], [], [], []
    for f in frames:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(f, **kw)[0]
        torch.cuda.synchronize()
        total.append((time.perf_counter() - t0) * 1e3)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    return dict(total_ms=stats(total), preprocess_ms=stats(pre), inference_ms=stats(inf), postprocess_ms=stats(post),
                fps_median=1000 / float(np.median(total)))


def build_coco5_eval(coco_names):
    """Hard-linked copy of the v4 validation images with labels remapped to COCO ids (5 shared classes)."""
    d = os.path.join(OUT_DIR, "eval_coco5")
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    for img in glob.glob(os.path.join(VAL_DIR, "images", "*")):
        dst = os.path.join(d, "images", os.path.basename(img))
        if not os.path.exists(dst):
            os.link(img, dst)
        stem = os.path.splitext(os.path.basename(img))[0]
        src_lbl = os.path.join(VAL_DIR, "labels", stem + ".txt")
        lines = []
        if os.path.exists(src_lbl):
            for ln in open(src_lbl):
                p = ln.split()
                if p and int(p[0]) in OUR_TO_COCO:
                    lines.append(" ".join([str(OUR_TO_COCO[int(p[0])])] + p[1:]))
        open(os.path.join(d, "labels", stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
    y = os.path.join(d, "data.yaml")
    yaml.safe_dump(dict(path=d, train="images", val="images", names=dict(coco_names)), open(y, "w"))
    return y


def accuracy(model, data_yaml, classes):
    m = model.val(data=data_yaml, classes=classes, imgsz=IMGSZ, batch=16, device=0, plots=False, verbose=False,
                  project=OUT_DIR, name="val_tmp", exist_ok=True)
    per = {}
    for k, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(k)
        per[m.names[int(c)]] = dict(P=float(p), R=float(r), AP50=float(ap50), AP50_95=float(ap))
    return dict(mAP50=float(np.mean([v["AP50"] for v in per.values()])),
                mAP50_95=float(np.mean([v["AP50_95"] for v in per.values()])), per_class=per)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = load_frames()
    print(f"{len(frames)} frames in memory ({frames[0].shape[1]}x{frames[0].shape[0]}); "
          f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    coco_yaml = build_coco5_eval(YOLO(MODELS["YOLOv8n (COCO)"][1]).names)
    ours_yaml = os.path.join(ROOT, "dataset_merged_v4", "data.yaml")
    results = {}
    loaded = {name: loader(w) for name, (loader, w) in MODELS.items()}
    # Timing: two rounds with the models alternating, so GPU clock ramp-up does not favour the later models;
    # the second round is reported (both are saved).
    timing = {name: {} for name in MODELS}
    for rnd in (1, 2):
        for name, model in loaded.items():
            for half in (False, True):
                timing[name][f"round{rnd}_{'fp16' if half else 'fp32'}"] = time_model(model, frames, half)
    for name, (loader, w) in MODELS.items():
        model = loaded[name]
        params = sum(p.numel() for p in model.model.parameters())
        gflops = get_flops(model.model, IMGSZ)
        res = dict(weights=os.path.basename(w), file_mb=os.path.getsize(w) / 1e6, params_m=params / 1e6, gflops=gflops,
                   fp32=timing[name]["round2_fp32"], fp16=timing[name]["round2_fp16"], round1=dict(
                       fp32=timing[name]["round1_fp32"], fp16=timing[name]["round1_fp16"]))
        fine_tuned = "fine-tuned" in name
        res["accuracy"] = accuracy(model, ours_yaml if fine_tuned else coco_yaml,
                                   sorted(OUR_TO_COCO) if fine_tuned else sorted(OUR_TO_COCO.values()))
        results[name] = res
        print(f"{name}: {res['file_mb']:.1f} MB, {res['params_m']:.1f} M params, {res['gflops']:.1f} GFLOPs | "
              f"FP32 {res['fp32']['total_ms']['median']:.1f} ms ({res['fp32']['inference_ms']['median']:.1f} inference) "
              f"-> {res['fp32']['fps_median']:.0f} FPS | FP16 {res['fp16']['total_ms']['median']:.1f} ms "
              f"-> {res['fp16']['fps_median']:.0f} FPS | mAP50 (5 shared classes) {res['accuracy']['mAP50']:.3f}", flush=True)
    meta = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                ultralytics=__import__("ultralytics").__version__, frames=len(frames), imgsz=IMGSZ, conf=CONF,
                batch=1, warmup=WARMUP)
    json.dump(dict(meta=meta, results=results), open(os.path.join(OUT_DIR, "benchmark.json"), "w"), indent=1)
    print("saved", os.path.join(OUT_DIR, "benchmark.json"))

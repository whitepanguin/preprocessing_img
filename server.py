# main.py (CPU-only)
import io, base64, os
from typing import List, Any, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import easyocr
import torch
from threading import Lock

# ---- CPU 강제 세팅 (CUDA 완전 비활성화) ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 어떤 GPU도 보이지 않게
try:
    torch.backends.cudnn.enabled = False  # cuDNN 비활성화
except Exception:
    pass
# (선택) 스레드 수 제한 – 서버 환경에 맞춰 조절
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
cv2.setNumThreads(int(os.getenv("OPENCV_NUM_THREADS", "2")))

app = FastAPI(title="EasyOCR CPU Server (pt 지원, CPU-only)", version="1.2.0")
_lock = Lock()

# -------------------- 전처리 (손글씨/저대비 강화) --------------------
def preprocess_handwriting(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), 1)
    h, w = thr.shape[:2]
    scale = 2 if max(h, w) < 80 else 1.5
    thr = cv2.resize(thr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# -------------------- 프리셋 --------------------
INFER_PRESETS: Dict[str, Dict[str, Any]] = {
    "field_money_precise": dict(
        text_threshold=0.5, low_text=0.5, link_threshold=0.5,
        canvas_size=2560, mag_ratio=1.5,
        paragraph=False, detail=1,
        contrast_ths=0.25, adjust_contrast=0.6,
        ycenter_ths=0.5, height_ths=0.6, width_ths=0.5, slope_ths=0.1, add_margin=0.18,
        allowlist="0123456789,.-₩원",
    ),
    "field_date_id_phone": dict(
        text_threshold=0.5, low_text=0.5, link_threshold=0.5,
        canvas_size=2560, mag_ratio=1.4,
        paragraph=False, detail=1,
        contrast_ths=0.25, adjust_contrast=0.6,
        ycenter_ths=0.5, height_ths=0.6, width_ths=0.55, slope_ths=0.1, add_margin=0.16,
        allowlist="0123456789-./:",
    ),
    "field_account": dict(
        text_threshold=0.55, low_text=0.55, link_threshold=0.55,
        canvas_size=2560, mag_ratio=1.3,
        paragraph=False, detail=1,
        contrast_ths=0.2, adjust_contrast=0.55,
        ycenter_ths=0.45, height_ths=0.55, width_ths=0.55, slope_ths=0.08, add_margin=0.14,
        allowlist="0123456789-",
        blocklist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ),
    "body_beam": dict(
        text_threshold=0.45, low_text=0.45, link_threshold=0.4,
        canvas_size=3000, mag_ratio=1.2,
        paragraph=True, detail=1,
        contrast_ths=0.25, adjust_contrast=0.6,
        ycenter_ths=0.6, height_ths=0.6, width_ths=0.6, slope_ths=0.15, add_margin=0.18,
    ),
    "body_tinyfont": dict(
        text_threshold=0.4, low_text=0.4, link_threshold=0.35,
        canvas_size=3400, mag_ratio=1.5,
        paragraph=True, detail=1,
        contrast_ths=0.3, adjust_contrast=0.65,
        ycenter_ths=0.65, height_ths=0.65, width_ths=0.65, slope_ths=0.16, add_margin=0.2,
    ),
    "body_lowprint": dict(
        text_threshold=0.35, low_text=0.35, link_threshold=0.3,
        canvas_size=3000, mag_ratio=1.5,
        paragraph=True, detail=1,
        contrast_ths=0.3, adjust_contrast=0.7,
        ycenter_ths=0.7, height_ths=0.7, width_ths=0.65, slope_ths=0.18, add_margin=0.22,
    ),
    "table_merge_light": dict(
        text_threshold=0.45, low_text=0.45, link_threshold=0.45,
        canvas_size=3000, mag_ratio=1.2,
        paragraph=False, detail=1,
        contrast_ths=0.25, adjust_contrast=0.6,
        ycenter_ths=0.7, height_ths=0.7, width_ths=0.55, slope_ths=0.12, add_margin=0.2,
    ),
    "table_complex": dict(
        text_threshold=0.5, low_text=0.5, link_threshold=0.5,
        canvas_size=3000, mag_ratio=1.2,
        paragraph=False, detail=1,
        contrast_ths=0.2, adjust_contrast=0.55,
        ycenter_ths=0.55, height_ths=0.55, width_ths=0.5, slope_ths=0.1, add_margin=0.16,
    ),
    "camera_lowcontrast": dict(
        text_threshold=0.4, low_text=0.4, link_threshold=0.35,
        canvas_size=3000, mag_ratio=1.3,
        paragraph=False, detail=1,
        contrast_ths=0.3, adjust_contrast=0.7,
        ycenter_ths=0.65, height_ths=0.65, width_ths=0.6, slope_ths=0.15, add_margin=0.2,
        rotation_info=[0, 90, 180, 270],
    ),
    "camera_skew": dict(
        text_threshold=0.45, low_text=0.45, link_threshold=0.4,
        canvas_size=3000, mag_ratio=1.3,
        paragraph=False, detail=1,
        contrast_ths=0.25, adjust_contrast=0.6,
        ycenter_ths=0.7, height_ths=0.7, width_ths=0.65, slope_ths=0.2, add_margin=0.22,
    ),
    "handwritten": dict(
        text_threshold=0.35, low_text=0.35, link_threshold=0.3,
        canvas_size=3000, mag_ratio=1.6,
        paragraph=False, detail=1,
        contrast_ths=0.3, adjust_contrast=0.7,
        ycenter_ths=0.75, height_ths=0.75, width_ths=0.7, slope_ths=0.2, add_margin=0.24,
    ),
    "checkbox_nearby": dict(
        text_threshold=0.52, low_text=0.52, link_threshold=0.48,
        canvas_size=2800, mag_ratio=1.2,
        paragraph=False, detail=1,
        contrast_ths=0.2, adjust_contrast=0.55,
        ycenter_ths=0.55, height_ths=0.55, width_ths=0.5, slope_ths=0.1, add_margin=0.14,
        blocklist="□■☑✔✓✗✘●○•◦◻◼",
    ),
}

# -------------------- 가중치 로드 도우미 (CPU 고정) --------------------
def load_easyocr_pt(reader: easyocr.Reader, path: str, strict: bool = False) -> Dict[str, Any]:
    """
    학습한 .pt를 Reader에 로드 (recognizer/detector 지원). 항상 CPU로 로드.
    """
    if not path or not os.path.exists(path):
        return {}
    ckpt = torch.load(path, map_location="cpu")
    rec = ckpt.get("recognizer")
    det = ckpt.get("detector")
    if rec is not None and hasattr(reader, "recognizer"):
        reader.recognizer.load_state_dict(rec, strict=strict)
    if det is not None and hasattr(reader, "detector"):
        try:
            reader.detector.load_state_dict(det, strict=strict)
        except Exception:
            pass
    return ckpt

# -------------------- 모델 로더 (CPU 전용 Reader) --------------------
def create_reader():
    # 언어/모델 디렉토리만 환경변수로 받고, gpu/download 옵션은 무시(항상 CPU/오프라인 기본)
    langs = os.getenv("EASYOCR_LANGS", "ko,en").split(",")
    models_dir = os.getenv("EASYOCR_MODELS_DIR", "./models")
    os.makedirs(models_dir, exist_ok=True)
    recog_network = os.getenv("EASYOCR_RECOG", "korean_g2")

    reader = easyocr.Reader(
        langs,
        gpu=False,  # <<< CPU 고정
        model_storage_directory=models_dir,
        user_network_directory=models_dir,
        download_enabled=False,  # 서버에서는 기본 오프라인 권장
        recog_network=recog_network
    )

    # 학습 가중치 로드 (선택)
    ckpt_path = os.getenv("EASYOCR_CKPT", "easyocr_kog2_finetune.pt")
    meta = load_easyocr_pt(reader, ckpt_path, strict=False)
    if meta:
        print("[CKPT] loaded:", {
            "recog_network": meta.get("recog_network"),
            "version": meta.get("version"),
            "extra": meta.get("extra")
        })
    else:
        print("[CKPT] no checkpoint loaded (using base weights)")
    return reader

@app.on_event("startup")
def _startup():
    app.state.reader = create_reader()

# -------------------- 응답 스키마 --------------------
class OCRItem(BaseModel):
    bbox: List[List[float]]
    text: str
    conf: float

class OCRResponse(BaseModel):
    message: str
    text: str
    mean_conf: float
    items: List[OCRItem]
    image: str  # base64 JPEG

# -------------------- OCR 실행 --------------------
def _run_ocr(
    bgr: np.ndarray,
    use_preprocess: bool,
    preset_name: str
) -> Tuple[str, float, List[Dict[str, Any]], np.ndarray]:
    if bgr is None or bgr.size == 0:
        raise HTTPException(400, "Invalid image")

    vis = bgr.copy()
    if use_preprocess:
        bgr = preprocess_handwriting(bgr)

    kwargs = INFER_PRESETS.get(preset_name, INFER_PRESETS["body_beam"])

    with _lock:  # Reader 내부 상태 보호
        results = app.state.reader.readtext(bgr, **kwargs)

    items, confs, texts = [], [], []
    for r in results:
        try:
            bbox, text, conf = r[0], str(r[1]), float(r[2])
        except Exception:
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                bbox, text, conf = r[0], str(r[1]), float(r[2])
            else:
                continue
        items.append({"bbox": bbox, "text": text, "conf": conf})
        texts.append(text); confs.append(conf)

        pts = np.array(bbox, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(vis, [pts], True, (255,0,0), 2)
        x, y = int(pts[0,0,0]), int(pts[0,0,1])
        cv2.putText(vis, f"{text} {conf:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

    agg_text = " ".join(texts).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return agg_text, mean_conf, items, vis

# -------------------- 라우트 --------------------
@app.get("/health")
def health():
    md = os.getenv("EASYOCR_MODELS_DIR", "./models")
    files = sorted(os.listdir(md)) if os.path.isdir(md) else []
    ckpt = os.getenv("EASYOCR_CKPT", "easyocr_kog2_finetune.pt")
    exists = os.path.exists(ckpt)
    return {
        "ok": True,
        "models_dir": md,
        "files": files,
        "ckpt": ckpt,
        "ckpt_exists": exists,
        "cpu_only": True
    }

@app.get("/presets")
def list_presets():
    return {"presets": list(INFER_PRESETS.keys())}

@app.post("/detect", response_model=OCRResponse)
async def detect_service(
    message: str = Form(...),
    file: UploadFile = File(...),
    preprocess: bool = Form(False),
    preset: str = Form("body_beam"),
):
    img = Image.open(io.BytesIO(await file.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    text, mean_conf, items, vis = _run_ocr(
        bgr, use_preprocess=preprocess, preset_name=preset
    )

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis_rgb).save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return OCRResponse(
        message=message,
        text=text,
        mean_conf=mean_conf,
        items=items,
        image=img_b64,
    )

if __name__ == "__main__":
    import uvicorn
    # CPU-only 환경에서는 worker 1개 권장(모델 1회 로드, 락으로 안전)
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)

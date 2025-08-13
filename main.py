import io, base64, os
from typing import List, Optional, Any, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import easyocr

app = FastAPI(title="EasyOCR CPU Server", version="1.0.0")

# -------- 전처리(손글씨 강화) --------
def preprocess_handwriting(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0: return bgr
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),1)
    h, w = thr.shape[:2]
    scale = 2 if max(h,w) < 80 else 1.5
    thr = cv2.resize(thr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# -------- 모델 로더 --------
def create_reader():
    langs = os.getenv("EASYOCR_LANGS", "ko,en").split(",")
    models_dir = os.getenv("EASYOCR_MODELS_DIR", "./models")
    os.makedirs(models_dir, exist_ok=True)
    gpu = os.getenv("EASYOCR_GPU", "0") == "1"            # 서버는 CPU면 0
    download = os.getenv("EASYOCR_DOWNLOAD", "0") == "1"  # 오프라인이면 0
    return easyocr.Reader(
        langs, gpu=gpu,
        model_storage_directory=models_dir,
        user_network_directory=models_dir,
        download_enabled=download,
    )

@app.on_event("startup")
def _startup():
    app.state.reader = create_reader()

# -------- 응답 스키마 (예측만) --------
class OCRItem(BaseModel):
    bbox: List[List[float]]  # 4점 좌표
    text: str
    conf: float

class OCRResponse(BaseModel):
    message: str
    text: str             # 전체 합쳐진 텍스트
    mean_conf: float
    items: List[OCRItem]  # 각 단어/라인의 bbox/text/conf
    image: str            # 박스/텍스트 오버레이된 base64(JPEG)

# -------- OCR 실행 --------
def _run_ocr(bgr: np.ndarray, use_preprocess: bool) -> Tuple[str, float, List[Dict[str, Any]], np.ndarray]:
    if bgr is None or bgr.size == 0:
        raise HTTPException(400, "Invalid image")
    vis = bgr.copy()
    if use_preprocess:
        bgr = preprocess_handwriting(bgr)

    results = app.state.reader.readtext(bgr, detail=1, paragraph=False)
    items, confs, texts = [], [], []
    for r in results:
        try:
            bbox, text, conf = r[0], str(r[1]), float(r[2])
        except Exception:
            if isinstance(r, (list,tuple)) and len(r) >= 3:
                bbox, text, conf = r[0], str(r[1]), float(r[2])
            else:
                continue
        items.append({"bbox": bbox, "text": text, "conf": conf})
        texts.append(text); confs.append(conf)

        pts = np.array(bbox, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(vis, [pts], True, (255,0,0), 2)
        x, y = pts[0,0,0], pts[0,0,1]
        cv2.putText(vis, f"{text} {conf:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 2)

    agg_text = " ".join(texts).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return agg_text, mean_conf, items, vis

@app.get("/health")
def health():
    md = os.getenv("EASYOCR_MODELS_DIR", "./models")
    files = sorted(os.listdir(md)) if os.path.isdir(md) else []
    return {"ok": True, "models_dir": md, "files": files}

@app.post("/detect", response_model=OCRResponse)
async def detect_service(
    message: str = Form(...),
    file: UploadFile = File(...),
    preprocess: bool = Form(False),
):
    img = Image.open(io.BytesIO(await file.read()))
    if img.mode != "RGB": img = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    text, mean_conf, items, vis = _run_ocr(bgr, use_preprocess=preprocess)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO(); Image.fromarray(vis_rgb).save(buf, format='JPEG')
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
    uvicorn.run(app, host="0.0.0.0", port=8001)

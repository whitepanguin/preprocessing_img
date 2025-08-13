# server.py
import os, io, base64
from typing import List, Any, Dict, Tuple
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageOps,ImageFont, ImageDraw
import torch
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import asyncio, json, uuid
from fastapi import Form
from fastapi.responses import StreamingResponse

# 진행상태 저장소
_progress_lock = Lock()
progress_states: Dict[str, Dict[str, Any]] = {}  # {pid: {"pct":int, "stage":str, "done":bool}}


def progress_update(pid: str, pct: int, stage: str):
    if not pid:
        return
    with _progress_lock:
        st = progress_states.setdefault(pid, {"pct": 0, "stage": "", "done": False})
        st["pct"] = int(max(0, min(100, pct)))
        st["stage"] = str(stage)
    print(f"[PROG] {pid} -> {pct}% | {stage}") 

def progress_finish(pid: str):
    if not pid:
        return
    with _progress_lock:
        st = progress_states.setdefault(pid, {"pct": 0, "stage": "", "done": False})
        st["pct"] = 100
        st["done"] = True
    print(f"[PROG] {pid} -> 100% | done")   

def _load_korean_font(font_size: int) -> ImageFont.FreeTypeFont:
    # 우선순위 후보들(Windows → 프로젝트 → Noto)
    candidates = [
        os.getenv("KOR_FONT_PATH", ""),                 # 환경변수로 직접 지정 가능
        r"C:\Windows\Fonts\malgun.ttf",                 # 맑은 고딕 (Windows)
        r"C:\Windows\Fonts\malgunsl.ttf",               # 맑은 고딕 Semilight
        "./fonts/NanumGothic.ttf",                      # 프로젝트 내
        "./fonts/NotoSansCJK-Regular.ttc",              # Noto CJK
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", # Linux
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                f = ImageFont.truetype(p, font_size)
                print(f"[FONT] using: {p}")
                return f
            except Exception as e:
                print(f"[FONT] fail: {p} -> {e}")
                continue
    print("[FONT] fallback to default (Korean may not render)")
    return ImageFont.load_default()

# -------------------- 환경 설정 --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU 강제
try:
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
except Exception:
    pass
cv2.setNumThreads(int(os.getenv("OPENCV_NUM_THREADS", "2")))

# -------------------- 경로/설정 --------------------
FONT_PATH = os.getenv("KOR_FONT_PATH", "./fonts/NanumGothic.ttf")
CKPT_PATH     = os.getenv("EASYOCR_CKPT", "./models/easyocr_kog2_finetune.pt")
MODELS_DIR    = os.getenv("EASYOCR_MODELS_DIR", "./models")
RECOG_NETWORK = os.getenv("EASYOCR_RECOG", "korean_g2").strip()  # ★ 공백 제거
LANGS         = os.getenv("EASYOCR_LANGS", "ko,en").split(",")

# (선택) 처음 한 번은 자동 다운로드 허용
EASYOCR_DOWNLOAD = os.getenv("EASYOCR_DOWNLOAD", "1") == "1"     # ★ 기본 1


# -------------------- FastAPI --------------------
app = FastAPI(title="EasyOCR CPU Server (pt)", version="1.3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_lock = Lock()

# -------------------- 전처리 (기본 미사용) --------------------
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
    scale = 2 if max(h,w) < 80 else 1.5
    thr = cv2.resize(thr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# -------------------- 스키마 --------------------
class OCRItem(BaseModel):
    bbox: List[List[float]]
    text: str
    conf: float

class OCRResponse(BaseModel):
    text: str
    mean_conf: float
    items: List[OCRItem]
    image: str
    progress_id: str | None = None

# -------------------- 체크포인트 로더 (부분 로드) --------------------
def _pick_recognizer_sd(ckpt: dict):
    # 다양한 저장 케이스 대응
    for k in ["recognizer", "state_dict", "model", "net", "weights"]:
        v = ckpt.get(k)
        if isinstance(v, dict):
            return v
    # 혹시 최상위가 바로 state_dict 형태인 경우
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    return None

def _normalize_keys(sd: dict):
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):            # DataParallel 제거
            k2 = k2[len("module."):]
        if k2.startswith("model."):             # 일부 저장 스크립트
            k2 = k2[len("model."):]
        # (필요시 추가 규칙): 양자화/이름 변형 정리
        k2 = k2.replace("SequenceModeling.0.rnn.rnn.", "SequenceModeling.0.rnn.")
        out[k2] = v
    return out

def _fix_first_conv_channel_mismatch(src: dict, dst: dict):
    """
    recognizer 첫 conv가 1채널인데, ckpt는 3채널(혹은 반대)인 경우 보정
    - 3→1: 평균채널로 축소
    - 1→3: 복제해서 3채널로 확장
    """
    cand_keys = [k for k in dst.keys() if k.endswith("ConvNet.0.weight") or k.endswith("cnn.0.weight")]
    if not cand_keys:
        return src  # 못 찾으면 패스

    k = cand_keys[0]
    if k in src and isinstance(src[k], torch.Tensor) and isinstance(dst[k], torch.Tensor):
        w_src, w_dst = src[k], dst[k]
        if w_src.dim()==4 and w_dst.dim()==4 and w_src.shape[0]==w_dst.shape[0] and w_src.shape[2:]==w_dst.shape[2:]:
            c_in_src, c_in_dst = w_src.shape[1], w_dst.shape[1]
            if c_in_src==3 and c_in_dst==1:
                # 3→1 평균
                src[k] = w_src.mean(dim=1, keepdim=True)
            elif c_in_src==1 and c_in_dst==3:
                # 1→3 복제
                src[k] = w_src.repeat(1,3,1,1)
    return src

# load_easyocr_pt 교체
def load_easyocr_pt(reader: easyocr.Reader, path: str):
    import torch
    if not path or not os.path.exists(path):
        print(f"[CKPT] not found: {path}")
        return {}, {"rec_matched":0,"rec_missing":0,"rec_unexpected":0,"det_loaded":False}

    ckpt = torch.load(path, map_location="cpu")
    stats = {"rec_matched":0,"rec_missing":0,"rec_unexpected":0,"det_loaded":False}

    rec = ckpt.get("recognizer")
    if rec is not None and hasattr(reader, "recognizer"):
        tgt = reader.recognizer.state_dict()
        inter = {k:v for k,v in rec.items() if k in tgt and getattr(tgt[k],"shape",None)==getattr(v,"shape",None)}
        stats["rec_matched"] = len(inter)
        stats["rec_missing"] = sum(1 for k in tgt.keys() if k not in inter)
        stats["rec_unexpected"] = sum(1 for k in rec.keys() if k not in tgt)
        tgt.update(inter)
        reader.recognizer.load_state_dict(tgt, strict=False)
        print(f"[CKPT] recognizer load: matched={stats['rec_matched']} missing={stats['rec_missing']} unexpected={stats['rec_unexpected']}")

    det = ckpt.get("detector")
    if det is not None and hasattr(reader, "detector"):
        try:
            tgt = reader.detector.state_dict()
            inter = {k:v for k,v in det.items() if k in tgt and getattr(tgt[k],"shape",None)==getattr(v,"shape",None)}
            tgt.update(inter)
            reader.detector.load_state_dict(tgt, strict=False)
            stats["det_loaded"] = True
            print("[CKPT] detector partial loaded")
        except Exception as e:
            print(f"[CKPT] detector load skipped: {e}")

    return ckpt, stats



# -------------------- Reader 준비 --------------------
def create_reader() -> easyocr.Reader:
    os.makedirs(MODELS_DIR, exist_ok=True)
    reader = easyocr.Reader(
        LANGS,
        gpu=False,
        model_storage_directory=MODELS_DIR,
        user_network_directory=MODELS_DIR,
        download_enabled=EASYOCR_DOWNLOAD,   # ★ 없으면 자동 다운로드
        recog_network=RECOG_NETWORK,         # 'korean_g2' (공백 제거됨)
        quantize=False
    )
    meta, stats = load_easyocr_pt(reader, CKPT_PATH)
    app.state.ckpt_stats = stats
    print("[CKPT] path:", CKPT_PATH, "| exists:", os.path.exists(CKPT_PATH))
    if meta:
        print("[CKPT] version:", meta.get("version"), "| recog_network:", meta.get("recog_network"))
    return reader


@app.on_event("startup")
def _startup():
    app.state.reader = create_reader()

# -------------------- OCR 실행 (checkbox_nearby 파라미터 직접 지정) --------------------
def _parse_item(r):
    """EasyOCR 결과 항목을 안전하게 (bbox, text, conf)로 변환. 실패 시 None 반환."""
    try:
        if isinstance(r, (list, tuple)):
            # detail=1이면 보통 (bbox, text, conf)
            if len(r) >= 3:
                bbox = r[0]
                text = "" if r[1] is None else str(r[1])
                conf = 0.0 if r[2] is None else float(r[2])
                return bbox, text, conf
            # 혹시 (bbox, text)만 오는 경우
            elif len(r) == 2:
                bbox = r[0]
                text = "" if r[1] is None else str(r[1])
                conf = 0.0
                return bbox, text, conf
    except Exception:
        pass
    return None

def _score_results(results):
    """안전 스코어: 내용 있는 텍스트 개수, 평균 conf"""
    non_empty = 0
    confs = []
    for r in results or []:
        parsed = _parse_item(r)
        if not parsed:
            continue
        _, text, conf = parsed
        if text.strip() and conf > 0:
            non_empty += 1
        confs.append(conf)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return non_empty, mean_conf


def _run_once(bgr: np.ndarray, kwargs: dict):
    kw = dict(kwargs)
    kw.pop("name", None)  # readtext에 name 넘기지 않기
    try:
        results = app.state.reader.readtext(bgr, **kw)
    except Exception as e:
        print(f"[WARN] readtext failed for {kwargs.get('name','?')}: {e}")
        results = []
    return results, _score_results(results)


def _maybe_preprocess(bgr: np.ndarray) -> np.ndarray:
    # 자동 전처리: 너무 어둡거나 대비 낮으면 한 번 보정
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean = float(np.mean(gray))
    if var < 50 or mean < 80:  # 임계는 데이터에 맞춰 조정 가능
        return preprocess_handwriting(bgr)
    return bgr

def _run_ocr(bgr: np.ndarray, progress_id: str = "", use_preprocess: bool = False) -> Tuple[str, float, List[Dict[str, Any]], np.ndarray]:
    if bgr is None or bgr.size == 0:
        raise HTTPException(400, "Invalid image")

    progress_update(progress_id, 5, "이미지 수신")

    if use_preprocess:
        progress_update(progress_id, 10, "전처리")
        bgr = preprocess_handwriting(bgr)
    else:
        progress_update(progress_id, 10, "전처리 생략")

    # 여러 후보 파라미터 자동 시도(이전 자동 폴백 로직을 쓰는 경우)
    CANDIDATES = [
        dict(name="checkbox_nearby", text_threshold=0.52, low_text=0.52, link_threshold=0.48,
             canvas_size=2800, mag_ratio=1.2, paragraph=False, detail=1, contrast_ths=0.2, adjust_contrast=0.55,
             ycenter_ths=0.55, height_ths=0.55, width_ths=0.5, slope_ths=0.1, add_margin=0.14,
             blocklist="□■☑✔✓✗✘●○•◦◻◼", rotation_info=[0,90,180,270]),
        dict(name="body_beam", text_threshold=0.45, low_text=0.45, link_threshold=0.40,
             canvas_size=3000, mag_ratio=1.2, paragraph=True, detail=1, contrast_ths=0.25, adjust_contrast=0.6,
             ycenter_ths=0.6, height_ths=0.6, width_ths=0.6, slope_ths=0.15, add_margin=0.18, rotation_info=[0,90,180,270]),
        dict(name="table_merge_light", text_threshold=0.45, low_text=0.45, link_threshold=0.45,
             canvas_size=3000, mag_ratio=1.2, paragraph=False, detail=1, contrast_ths=0.25, adjust_contrast=0.6,
             ycenter_ths=0.7, height_ths=0.7, width_ths=0.55, slope_ths=0.12, add_margin=0.2, rotation_info=[0,90,180,270]),
        dict(name="body_tinyfont", text_threshold=0.40, low_text=0.40, link_threshold=0.35,
             canvas_size=3400, mag_ratio=1.5, paragraph=True, detail=1, contrast_ths=0.30, adjust_contrast=0.65,
             ycenter_ths=0.65, height_ths=0.65, width_ths=0.65, slope_ths=0.16, add_margin=0.2, rotation_info=[0,90,180,270]),
    ]

    total = len(CANDIDATES)
    best = None

    with _lock:
        for i, kw in enumerate(CANDIDATES, start=1):
            # 이름 제거하고 readtext 호출
            params = {k: v for k, v in kw.items() if k != "name"}
            progress_update(progress_id, 10 + int(70 * i / total), f"OCR 추론({kw['name']})")
            results = app.state.reader.readtext(bgr, **params)

            score = _score_results(results)
            if best is None or score > best[0]:
                best = (score, kw["name"], results)

    progress_update(progress_id, 85, "후처리/시각화")

    # 최종 결과 구성
    _, best_name, results = best if best else ((0, 0.0), "none", [])
    vis = bgr.copy()
    items, confs, texts = [], [], []
    for r in results or []:
        parsed = _parse_item(r)
        if not parsed:
            continue
        bbox, text, conf = parsed
        items.append({"bbox": bbox, "text": text, "conf": conf})
        texts.append(text)
        confs.append(conf)
        # 시각화 (필요 시 PIL 폰트로)
        pts = np.array(bbox, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (255, 0, 0), 2)

    agg_text = " ".join([t for t in texts if t.strip()]).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0

    progress_update(progress_id, 95, "이미지 인코딩")
    # detect_service 쪽에서 base64 변환 후 100% 마무리

    return agg_text, mean_conf, items, vis




# -------------------- 라우트 --------------------
@app.get("/health")
def health():
    files = sorted(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else []
    return {
        "ok": True,
        "models_dir": MODELS_DIR,
        "files": files,
        "ckpt": CKPT_PATH,
        "ckpt_exists": os.path.exists(CKPT_PATH),
        "lang_list": LANGS,
        "recog_network": RECOG_NETWORK,
        "ckpt_stats": getattr(app.state, "ckpt_stats", None),
    }

@app.get("/progress_json/{pid}")
def progress_json(pid: str):
    with _progress_lock:
        st = progress_states.get(pid)
        if not st:
            # 최초 접근 시 상태 생성
            st = progress_states.setdefault(pid, {"pct": 0, "stage": "waiting", "done": False})
        # 얕은 복사로 반환
        return {"pct": st["pct"], "stage": st["stage"], "done": st.get("done", False)}




@app.post("/detect", response_model=OCRResponse)
async def detect_service(
    file: UploadFile = File(...),
    progress_id: str = Form(None),   # ★ 추가
    preprocess: int = Form(0),
):
    # 진행ID 없으면 생성
    pid = progress_id or str(uuid.uuid4())
    print(f"[DETECT] pid={pid}")
    progress_update(pid, 1, "업로드 수신")

    img = Image.open(io.BytesIO(await file.read()))
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # (선택) 리사이즈로 속도 향상
    max_side = 2200
    h, w = bgr.shape[:2]
    print(f"[DETECT] size={w}x{h}")
    s = max(h, w) / float(max_side)
    if s > 1.0:
        bgr = cv2.resize(bgr, (int(w/s), int(h/s)), interpolation=cv2.INTER_AREA)

    text, mean_conf, items, vis = _run_ocr(
        bgr, progress_id=pid, use_preprocess=bool(preprocess)
    )

    # base64로 인코딩
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis_rgb).save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    progress_finish(pid)  # ★ 100% 완료

    return OCRResponse(text=text, mean_conf=mean_conf, items=items, image=img_b64)

if __name__ == "__main__":
    import uvicorn
    # 포트는 프런트/Java와 맞춰주세요 (예: 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

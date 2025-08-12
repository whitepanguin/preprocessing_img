import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# --- 함수 정의 부분 ---

def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x+w , y+h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상
    rect[2] = pts[np.argmax(s)]  # 우하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상
    rect[3] = pts[np.argmax(diff)]  # 좌하
    return rect

def transform_points(points, M):
    num_points = points.shape[0]
    pts_hom = np.hstack([points, np.ones((num_points, 1))])
    transformed = M @ pts_hom.T
    transformed /= transformed[2, :]
    return transformed[:2, :].T

def preprocess_image_and_annotation(image_path, json_path, font_path, save_img_path, save_json_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] 이미지 로드 실패: {image_path}")
        return False
    
    # JSON 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 이미지 복사 및 PIL 변환 준비
    image_with_bbox = image.copy()
    image_rgb_bbox = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
    pil_img_bbox = Image.fromarray(image_rgb_bbox)
    draw_bbox = ImageDraw.Draw(pil_img_bbox)
    font = ImageFont.truetype(font_path, 30)

    for ann in data["annotations"]:
        for poly in ann["polygons"]:
            points = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(image_with_bbox, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            text = poly.get("text", "")
            if text:
                x, y = points[0]
                draw_bbox.text((x, y - 20), text, font=font, fill=(255, 0, 0))

    image_with_bbox = cv2.cvtColor(np.array(pil_img_bbox), cv2.COLOR_RGB2BGR)

    # 그레이스케일 및 노이즈 제거
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.medianBlur(img_gray, 3)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    dst2 = np.zeros(img_gray.shape, np.uint8)
    bw = img_gray.shape[1] // 4
    bh = img_gray.shape[0] // 4
    for y in range(4):
        for x in range(4):
            img_ = img_gray[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
            dst_ = dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
            cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)

    gse = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img_gray = cv2.dilate(dst2, gse)
    img_gray = cv2.erode(img_gray, gse)

    _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    rect_contours = []
    rect_points = []
    for pts in contours:
        area = cv2.contourArea(pts)
        if area < 1800 or area > 3000:
            continue
        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)* 0.02, True)
        if len(approx) == 4:
            rect_contours.append(pts)
            rect_points.append(approx)

    if len(rect_contours) < 4:
        print(f"[WARN] 사각형 4개 미만: {image_path}")
        print(len(rect_contours))
        return False

    all_points = np.vstack([approx.reshape(4, 2) for approx in rect_points])

    hull = cv2.convexHull(all_points)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype("float32")
    ordered_box = order_points(box)

    (tl, tr, br, bl) = ordered_box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
    warped = cv2.warpPerspective(img_gray, M, (maxWidth, maxHeight))

    # 이미지 저장 (반전하여 저장)
    cv2.imwrite(save_img_path, ~warped)

    # JSON 좌표 변환
    transformed_data = data.copy()
    for ann in transformed_data["annotations"]:
        for poly in ann["polygons"]:
            points = np.array(poly["points"], dtype=np.float32)
            new_points = transform_points(points, M)
            new_points = np.round(new_points).astype(int)
            poly["points"] = new_points.tolist()

    # JSON 저장
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)

    print(f"[OK] 처리 완료: {os.path.basename(image_path)}")
    return True

# --- 메인 처리 루프 ---

def find_image_path(base_dir, base_name):
    jpeg_path = os.path.join(base_dir, base_name + ".jpeg")
    jpg_path = os.path.join(base_dir, base_name + ".jpg")

    if os.path.isfile(jpeg_path):
        return jpeg_path
    elif os.path.isfile(jpg_path):
        return jpg_path
    else:
        return None

def main():
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 환경에 맞게 수정 필요

    image_dir = "./1000/images"
    json_dir = "./1000/annotations"
    save_image_dir = "./1000/preprocessed_images"
    save_json_dir = "./1000/preprocessed_annotations"
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_json_dir, exist_ok=True)

    less_than_4_rect_list = []

    for idx in range(1, 1001):
        base_name = f"insure_{idx:05d}"
        image_path = find_image_path(image_dir, base_name)
        json_path = os.path.join(json_dir, base_name + ".json")

        if image_path is None:
            print(f"[WARN] 이미지 파일 없음: {base_name}")
            continue

        if not os.path.isfile(json_path):
            print(f"[WARN] JSON 파일 없음: {base_name}")
            continue

        save_img_path = os.path.join(save_image_dir, os.path.basename(image_path))
        save_json_path = os.path.join(save_json_dir, base_name + ".json")

        success = preprocess_image_and_annotation(image_path, json_path, font_path, save_img_path, save_json_path)
        if not success:
            # 사각형 4개 미만으로 실패시 리스트에 추가
            less_than_4_rect_list.append(base_name)

    # 사각형 4개 미만인 파일명 리스트를 list.txt로 저장
    with open("list.txt", "w", encoding="utf-8") as f:
        for name in less_than_4_rect_list:
            f.write(name + "\n")

    print(f"사각형 4개 미만인 파일 목록 list.txt에 저장 완료, 총 {len(less_than_4_rect_list)}개")


if __name__ == "__main__":
    main()

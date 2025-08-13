import json
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x+w , y+h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상: x+y 최소
    rect[2] = pts[np.argmax(s)]  # 우하: x+y 최대

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상: x-y 최소
    rect[3] = pts[np.argmax(diff)]  # 좌하: x-y 최대

    return rect
rect_contours = []
rect_points = []
all_points = []


# matplotlib 한글 폰트 설정 (플롯용)
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
print(font_name)  # 'Malgun Gothic' 등 폰트 이름 출력 확인
plt.rc('font', family=font_name)

# 파일 경로
image_path = "insure_00001.jpeg"
json_path = "insure_00001.json"

# 이미지 읽기 (OpenCV, BGR)
image = cv2.imread(image_path)

# JSON 읽기
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

image_with_bbox = image.copy()

# OpenCV 이미지(BGR) -> PIL 이미지(RGB) 변환
image_rgb_bbox = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
pil_img_bbox = Image.fromarray(image_rgb_bbox)
draw_bbox = ImageDraw.Draw(pil_img_bbox)
font = ImageFont.truetype(font_path, 30)  # 폰트 크기 조절 가능



for ann in data["annotations"]:
    for poly in ann["polygons"]:
        points = np.array(poly["points"], dtype=np.int32)
        # OpenCV로 다각형 그리기 (파란색)
        cv2.polylines(image_with_bbox, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        # 텍스트 그리기
        text = poly.get("text", "")
        if text:
            x, y = points[0]
            draw_bbox.text((x, y - 20), text, font=font, fill=(255, 0, 0))  # 빨간색 텍스트

# PIL 이미지 -> OpenCV 이미지(BGR) 변환
image_with_bbox = cv2.cvtColor(np.array(pil_img_bbox), cv2.COLOR_RGB2BGR)

# BGR → RGB 변환 (matplotlib 표시용)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 원본 이미지를 그레이스케일로 읽고 리사이즈
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 노이즈 제거
img_gray = cv2.medianBlur(img_gray, 3)
img_noi = cv2.GaussianBlur(img_gray, (3, 3), 0)

# 지역 이진
dst2 = np.zeros(img_noi.shape, np.uint8)
bw = img_gray.shape[1] // 4
bh = img_gray.shape[0] // 4
for y in range(4):
    for x in range(4):
        img_ = img_gray[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_ = dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)

gse = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
img_gray = cv2.dilate(~dst2, gse)
img_die = cv2.erode(img_gray, gse)

# 이미지 자르기
# 회색으로 받은 이미지를 반전해서 사각형 찾기
_, img_gray = cv2.threshold(~img_die, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(img_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
dst = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
color = (0, 255, 0)

scale_percent = 25  # 25% 크기
width = int(img_gray.shape[1] * scale_percent / 100)
height = int(img_gray.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)

for pts in contours:
    if cv2.contourArea(pts) < 2100 or cv2.contourArea(pts) > 3000:
        continue
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)* 0.02, True)
    print(approx)
    print(cv2.contourArea(pts))
    vtc = len(approx)
    print(vtc) 
    if vtc ==4:
        rect_contours.append(pts)
        rect_points.append(approx)
        setLabel(img_gray, pts, 'RECT')
        cv2.drawContours(dst, rect_contours, -1, color, 3)

if len(rect_contours) < 4:
    print(f"사각형이 4개 미만입니다! 현재 찾은 사각형 개수: {len(rect_contours)}")
    for i, approx in enumerate(rect_points):
        print(f"사각형 {i+1} 좌표: {approx.reshape(-1, 2)}")
    # 사각형이 적으니 투시변환은 하지 않고, 찾은 사각형만 이미지에 표시
    for pts in rect_contours:
        cv2.drawContours(img_gray, [pts], -1, (0, 255, 0), 3)
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
else:
    print(f"사각형 4개 이상 찾음: {len(rect_contours)}개")
    for approx in rect_points:  # rect_points = [approx1, approx2, approx3, approx4]
        pts = approx.reshape(4, 2)
        for p in pts:
            all_points.append(p)

    all_points = np.array(all_points)

    # 1) 컨벡스 헐(Convex Hull) 구하기
    hull = cv2.convexHull(all_points)

    # 2) 최소외접 사각형 구하기 (RotatedRect)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    # 3) 점 순서 정렬
    ordered_box = order_points(box)

    # 4) 투시 변환 크기 계산
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

    # 5) 투시 변환 매트릭스 계산 및 적용
    M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
    warped = cv2.warpPerspective(img_noi, M, (maxWidth, maxHeight))
    warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("warped transform", warped)



# 이미지 표시
plt.figure(figsize=(12, 16))
plt.imshow(image_with_bbox)
plt.axis("off")
#----
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('1 img_gray', image)
img_noi = cv2.resize(img_noi, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('2 img_noi', img_noi)
dst2 = cv2.resize(dst2, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('3 dst2', dst2)
dst = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('4 dst', dst)
img_die = cv2.resize(img_die, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('4 img_die', img_die)
# cv2.imshow('5 warped', warped)

#----
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

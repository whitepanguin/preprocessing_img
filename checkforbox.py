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
all_contours = []


# matplotlib 한글 폰트 설정 (플롯용)
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
print(font_name)  # 'Malgun Gothic' 등 폰트 이름 출력 확인
plt.rc('font', family=font_name)

# 파일 경로
image_path = "insure_00112.jpg"
json_path = "insure_00112.json"

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

# 지역 이진
dst2 = np.zeros(img_gray.shape, np.uint8)
bw = img_gray.shape[1] // 4
bh = img_gray.shape[0] // 4
for y in range(4):
    for x in range(4):
        img_ = img_gray[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_ = dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)


# 이미지 자르기
# 회색으로 받은 이미지를 반전해서 사각형 찾기
_, img_gray = cv2.threshold(dst2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(img_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
dst = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
color = (0, 255, 0)

for pts in contours:
    area = cv2.contourArea(pts)
    if area < 2100 or area > 3000:
        continue
    print(cv2.contourArea(pts))
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)* 0.02, True)
    vtc = len(approx)
    all_contours.append(pts)
    print(f"면적: {area}, 꼭짓점 수: {vtc}")
    # 라벨 표시 (img_contour_display에)
    setLabel(dst, pts, f'Area:{int(area)} Vtc:{vtc}')
    
    # contour 녹색으로 그림
    cv2.drawContours(dst, [pts], -1, (0, 255, 0), 2)

# 이미지 축소해서 보여주기 (원하면)
scale_percent = 25
width = int(dst.shape[1] * scale_percent / 100)
height = int(dst.shape[0] * scale_percent / 100)
dim = (width, height)
resized_img = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Contours with Labels', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
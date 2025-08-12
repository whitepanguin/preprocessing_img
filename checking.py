import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 한글 폰트 경로 (필요에 따라 바꾸세요)
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_size = 30
font = ImageFont.truetype(font_path, font_size)

# 이미지와 JSON 경로
image_path = "warped_insure_00102.jpeg"
json_path = "warped_insure_00102.json"

# 이미지 읽기 (OpenCV BGR)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

# JSON 읽기
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# OpenCV 이미지 -> PIL 이미지 변환 (텍스트 그리기 편리)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(pil_img)

for ann in data["annotations"]:
    for poly in ann["polygons"]:
        points = np.array(poly["points"], dtype=np.int32)
        # 바운딩 박스 좌표 계산
        x, y, w, h = cv2.boundingRect(points)
        # 사각형 그리기 (빨간색)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        # 텍스트 가져오기
        text = poly.get("text", "")
        if text:
            # 텍스트 위치 (바운딩박스 왼쪽 위보다 조금 위쪽)
            text_pos = (x, max(y - font_size, 0))
            draw.text(text_pos, text, font=font, fill="red")

# PIL 이미지 -> OpenCV 이미지로 다시 변환하여 보여주기
result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

cv2.imshow("Warped Image with Boxes and Text", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

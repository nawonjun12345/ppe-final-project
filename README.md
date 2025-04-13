
# 🦺 PPE Detection with YOLOv8

## 📌 프로젝트 개요
본 프로젝트는 산업현장에서 작업자의 **PPE (개인 보호 장비)** 착용 여부를 자동으로 감지하는 딥러닝 시스템을 구현한 것입니다.  
YOLOv8 객체 탐지 모델을 활용하여, 헬멧, 조끼, 안전화 등의 착용 여부를 이미지 기반으로 판별하고  
미착용 시 경고 메시지를 출력하는 기능을 포함합니다.

---

## 📂 데이터셋 정보

- 출처: [Roboflow Universe - PPE Detection v2i](https://universe.roboflow.com/)
- 클래스: Helmet, Vest, Boots, Gloves, Mask, Person, NO-Helmet, NO-Vest
- 구성: `train/`, `valid/`, `test/` 이미지와 YOLO 형식 라벨(.txt)
- ⚠ Gloves와 Mask 클래스는 학습에 충분하지 않아 실제 학습에선 제외됨

---

## 🧠 모델 정보

- 사용 모델: `YOLOv8n` (Ultralytics YOLO)
- 장점: 경량화, 실시간 탐지에 적합
- 프레임워크: Python + Google Colab

---

## 📜 코드 구성 및 셀 설명

1️⃣ **YOLO 설치**
```python
!pip install ultralytics
```
> Ultralytics YOLOv8 설치

2️⃣ **데이터셋 업로드**
```python
from google.colab import files
uploaded = files.upload()
```
> Roboflow에서 받은 `PPE.v2i.yolov8.zip` 파일 업로드

3️⃣ **압축 해제**
```python
import zipfile
with zipfile.ZipFile("PPE.v2i.yolov8.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/PPE.v2i.yolov8")
```

4️⃣ **모델 학습**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data="/content/PPE.v2i.yolov8/data.yaml", epochs=30, imgsz=640, batch=16)
```
> YOLOv8n으로 모델 학습 진행

5️⃣ **PPE 예측 + 결과 분석 및 시각화**
```python
# 이미지 업로드 및 감지
uploaded = files.upload()
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(...)

# 착용/미착용 여부 출력 + 결과 이미지 시각화
```

6️⃣ **결과 저장 및 다운로드**
```python
files.download('runs/detect/train/weights/best.pt')
files.download('runs/detect/train/results.png')
```

---

## ✅ 결과 예시

- 🎯 감지된 객체: ['Helmet', 'Vest', 'Boots', 'Person']
- ✅ 모든 보호구 착용 완료!
- 🚨 미착용 시: “미착용 항목: 헬멧, 조끼”

---

## 📦 포함 파일 안내

| 파일명 | 설명 |
|--------|------|
| `PPE_Detection_Result.ipynb` | 실행 후 전체 코드 노트북 |
| `PPE_Detection_Template.ipynb` | 실행 전 전체 코드 노트북 |
| `best.pt` | 학습된 YOLOv8n 모델 가중치 |
| `results.png` | 학습 성능 그래프 |
| `data.yaml` | 학습 클래스 설정 정보 |
---

## 🔗 GitHub 주소

https://github.com/nawonjun12345/ppe-final-project

---

## 🙏 마무리

- YOLOv8 모델을 통해 실제 PPE 착용 여부 감지가 가능함을 확인함
- 실시간 적용 및 웹 인터페이스 연동 등으로 확장 가능성 확인

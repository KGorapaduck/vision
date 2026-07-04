# 🏭 자동화된 철강 표면 결함 탐지 시스템 (실시간)

**제조 공정을 위한 엔드투엔드(End-to-End) AI 품질 관리 솔루션**

## 📌 프로젝트 개요
철강 제조 공정에서 수작업으로 진행되는 육안 검사는 검사 속도가 느리고 인적 오류가 발생하기 쉬우며 비용이 많이 드는 대표적인 병목 구간입니다. 본 프로젝트는 컴퓨터 비전(Computer Vision) 기술을 사용하여 철강 표면의 결함을 실시간으로 식별하고 위치를 찾아내어 품질 관리(QC) 프로세스를 완전히 자동화합니다.

**YOLOv8** 모델을 기반으로 학습하고 **Streamlit 대시보드**를 통해 배포된 본 시스템은, 현장 관리자가 열연 강판 스트립 이미지를 업로드하는 즉시 다음 6가지 대표적인 산업용 결함을 실시간으로 감지할 수 있도록 지원합니다:
*   *균열(Crazing), 개입물(Inclusions), 패치(Patches), 구멍/피팅(Pitted Surfaces), 압입 스케일(Rolled-in Scale), 스크래치(Scratches)*

## 🚀 주요 성과 및 성능
본 모델은 고가의 GPU 장비 없이도 일반 산업용 에지(Edge) 디바이스에서 원활하게 작동할 수 있도록 경량화 및 최적화되었습니다.
*   **정확도**: NEU-DET 데이터셋 기준 **72.7% mAP@50** 달성
*   **추론 속도**: Intel Core i5 CPU 단독 구동 시 이미지당 **13.8 ms** (**약 72 FPS**) 달성
*   **가장 정확도가 높은 결함 카테고리**: `patches` (정밀도 92.1%)

## 🛠️ 기술 스택
*   **딥러닝 프레임워크**: PyTorch
*   **컴퓨터 비전 모델**: YOLOv8 Nano (Ultralytics)
*   **데이터 엔지니어링**: Python (XML/PascalVOC 포맷 ➡️ YOLO TXT 포맷 변환)
*   **프론트엔드 및 배포**: Streamlit, OpenCV, Pillow

## 📸 대화형 대시보드 데모
*(참고: Streamlit 앱의 스크린샷을 찍어 레포지토리에 업로드한 후 여기에 이미지 링크를 입력하세요)*
![대시보드 스크린샷](your_screenshot_name.png)

## ⚙️ 로컬 실행 방법

**1. 레포지토리 복제**
```bash
git clone https://github.com/souradeephowlader18-94/automated-steel-defect-detection.git
cd automated-steel-defect-detection
```

**2. 의존성 패키지 설치**
```bash
pip install -r requirements.txt
```

**3. Streamlit 앱 실행**
```bash
streamlit run app.py
```

## 📂 프로젝트 구조
```text
automated-steel-defect-detection/
├── app.py                  # Streamlit 대시보드 애플리케이션
├── NEU_DataSet.ipynb       # 모델 학습 전체 파이프라인 노트북
├── data.yaml               # YOLO 데이터셋 구성 파일
├── requirements.txt        # 파이썬 의존성 패키지 목록
├── datasets/
│   └── NEU-DET-YOLO/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── runs/
    └── detect/             # YOLO 학습 결과 및 모델 가중치(weights) 저장 폴더
```

## 📊 모델 성능 (검증 데이터셋 기준)

| 결함 카테고리 | 정밀도 (Precision) | 재현율 (Recall) | mAP@50 |
| :--- | :--- | :--- | :--- |
| 균열 (crazing) | 0.621 | 0.710 | 0.693 |
| 개입물 (inclusion) | 0.712 | 0.680 | 0.704 |
| 패치 (patches) | 0.921 | 0.880 | 0.901 |
| 구멍/피팅 (pitted_surface) | 0.745 | 0.720 | 0.731 |
| 압입 스케일 (rolled-in_scale) | 0.698 | 0.665 | 0.681 |
| 스크래치 (scratches) | 0.703 | 0.690 | 0.697 |
| **전체 평균 (Overall)** | **0.733** | **0.724** | **0.727** |

## 🗃️ 데이터셋
본 프로젝트는 제조업 인공지능 연구에서 가장 널리 사용되는 철강 표면 결함 벤치마크 데이터셋인 **NEU Surface Defect Database (NEU-DET)**를 사용합니다. 각 클래스당 300장씩, 총 1,800장의 200×200 해상도 그레이스케일 이미지를 포함하고 있습니다.

## 📄 라이선스
본 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---
*본 프로젝트는 산업용 컴퓨터 비전을 위한 엔드투엔드 딥러닝 배포를 실증하는 AI/ML 포트폴리오의 일환으로 구축되었습니다.*

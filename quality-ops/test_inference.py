import os
import sys
import logging
import time
from ultralytics import YOLO

# Ensure parent directory is in sys.path to allow imports from shared and backend
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from shared.config import LOG_FILE_PATH, DATASETS_DIR, calculate_severity
from backend.db_logger import init_db, insert_inference_run, insert_defect_details

# 로그 설정 (콘솔 및 파일 동시 출력)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_inference_pipeline(version_name, class_thresholds=None):
    start_time = time.time()
    # 1. DB 초기화
    init_db()
    
    # 2. 모델 경로 빌드
    model_path = os.path.join(REPO_ROOT, "runs", "detect", f"steel_yolov8n_{version_name}", "weights", "best.pt")
    if not os.path.exists(model_path):
        logging.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return
        
    logging.info(f"YOLO 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 3. 전체 검증 데이터셋에 대한 공식 평가 지표 획득 (val)
    logging.info("검증 데이터셋(Validation Set) 평가 시작...")
    # exist_ok=True를 추가해 val 폴더 복제 증식 방지
    data_yaml_path = os.path.join(REPO_ROOT, 'data.yaml')
    metrics = model.val(data=data_yaml_path, plots=True, exist_ok=True, name=f"val_{version_name}")
    
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    
    logging.info(f"=== [모델 평가 요약] ===")
    logging.info(f"Model Version: {version_name}")
    logging.info(f"Mean Precision: {precision:.4f}")
    logging.info(f"Mean Recall: {recall:.4f}")
    logging.info(f"mAP50: {map50:.4f}")
    logging.info(f"=========================")

    # 클래스별 개별 임계값 정의
    if class_thresholds is not None:
        CLASS_THRESHOLDS = class_thresholds
    else:
        CLASS_THRESHOLDS = {
            "crazing": 0.15,
            "inclusion": 0.20,
            "patches": 0.25,
            "pitted_surface": 0.20,
            "rolled-in_scale": 0.15,
            "scratches": 0.25
        }

    # 4. 이미지 개별 추론 수행 (predict) -> 개별 결함 목록 추출
    val_images_path = os.path.join(DATASETS_DIR, 'NEU-DET-YOLO', 'images', 'val')
    logging.info(f"검증 이미지 추론 실행 중: {val_images_path}")
    # 클래스별 임계값 중 최솟값을 기준으로 1차 필터링
    min_conf = min(CLASS_THRESHOLDS.values())
    results = model.predict(source=val_images_path, conf=min_conf, save=False)
    
    total_images = len(results)
    total_defects = 0
    defect_list = []
    
    for result in results:
        img_name = os.path.basename(result.path)
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])
                
                # 클래스별 고유 임계값 적용하여 2차 필터링
                target_conf = CLASS_THRESHOLDS.get(class_name, 0.25)
                if confidence < target_conf:
                    continue  # 임계값 미달 시 적재 대상에서 제외
                    
                total_defects += 1
                
                # xywh 정규화 좌표 가져오기
                xywhn = box.xywhn[0].tolist()
                x_center, y_center, width, height = xywhn
                
                # 심각도 판정
                severity = calculate_severity(confidence, width, height)
                
                # 상세 리스트에 임시 딕셔너리로 저장
                defect_list.append({
                    'image_name': img_name,
                    'class_name': class_name,
                    'confidence': confidence,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'severity': severity
                })
                
    # 5. DB 저장
    run_id = insert_inference_run(version_name, total_images, total_defects, precision, recall, map50)
    
    # 튜플 리스트로 변환하여 벌크 인서트
    db_defect_tuples = [
        (
            run_id, d['image_name'], d['class_name'], d['confidence'],
            d['x_center'], d['y_center'], d['width'], d['height'], d['severity']
        )
        for d in defect_list
    ]
    
    insert_defect_details(db_defect_tuples)
    
    speed_pre = float(metrics.speed.get('preprocess', 0.0))
    speed_inf = float(metrics.speed.get('inference', 0.0))
    speed_post = float(metrics.speed.get('postprocess', 0.0))

    logging.info(f"=== [추론 및 로깅 완료] ===")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"총 이미지 수: {total_images}")
    logging.info(f"총 검출 결함 수: {total_defects}")
    logging.info(f"추론 속도: {speed_inf:.2f} ms/장 (전처리: {speed_pre:.2f}ms, 후처리: {speed_post:.2f}ms)")
    logging.info(f"결함 세부 로그가 '{LOG_FILE_PATH}' 및 SQLite DB에 기록되었습니다.")
    logging.info(f"============================")

if __name__ == '__main__':
    version = input("추론 및 성능 평가를 진행할 모델 버전을 입력하세요 (예: v1_base): ").strip()
    if not version:
        version = "v1_base"
    run_inference_pipeline(version)


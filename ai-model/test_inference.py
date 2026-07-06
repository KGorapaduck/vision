import os
import sys
import logging
from ultralytics import YOLO
from db_logger import init_db, insert_inference_run, insert_defect_details, calculate_severity

# 로그 설정 (콘솔 및 파일 동시 출력)
LOG_FILE = 'defect_detection.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_inference_pipeline(version_name):
    # 1. DB 초기화
    init_db()
    
    # 2. 모델 경로 빌드
    model_path = f"runs/detect/steel_yolov8n_{version_name}/weights/best.pt"
    if not os.path.exists(model_path):
        logging.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return
        
    logging.info(f"YOLO 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 3. 전체 검증 데이터셋에 대한 공식 평가 지표 획득 (val)
    logging.info("검증 데이터셋(Validation Set) 평가 시작...")
    metrics = model.val(data='data.yaml', plots=False)
    
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    
    logging.info(f"=== [모델 평가 요약] ===")
    logging.info(f"Model Version: {version_name}")
    logging.info(f"Mean Precision: {precision:.4f}")
    logging.info(f"Mean Recall: {recall:.4f}")
    logging.info(f"mAP50: {map50:.4f}")
    logging.info(f"=========================")

    # 4. 이미지 개별 추론 수행 (predict) -> 개별 결함 목록 추출
    val_images_path = 'datasets/NEU-DET-YOLO/images/val'
    logging.info(f"검증 이미지 추론 실행 중: {val_images_path}")
    results = model.predict(source=val_images_path, conf=0.25, save=False)
    
    total_images = len(results)
    total_defects = 0
    defect_list = []
    
    for result in results:
        img_name = os.path.basename(result.path)
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                total_defects += 1
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])
                
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
    
    logging.info(f"=== [추론 및 로깅 완료] ===")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"총 이미지 수: {total_images}")
    logging.info(f"총 검출 결함 수: {total_defects}")
    logging.info(f"결함 세부 로그가 '{LOG_FILE}' 및 SQLite DB에 기록되었습니다.")
    logging.info(f"============================")

if __name__ == '__main__':
    version = input("추론 및 성능 평가를 진행할 모델 버전을 입력하세요 (예: v1_base): ").strip()
    if not version:
        version = "v1_base"
    run_inference_pipeline(version)

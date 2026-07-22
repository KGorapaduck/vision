import os
import sys
import logging
import csv
import time
import glob

# Register NVIDIA CUDA 12 DLL directories BEFORE importing ONNX Runtime
site_packages = os.path.join(sys.prefix, "lib", "site-packages")
nvidia_base = os.path.join(site_packages, "nvidia")

dll_dirs = []
if os.path.exists(nvidia_base):
    for p in glob.glob(os.path.join(nvidia_base, "*", "bin")):
        if os.path.isdir(p):
            dll_dirs.append(p)

# Prepend to PATH so Windows Loader finds them first
for d in dll_dirs:
    if os.path.exists(d):
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

import torch
import onnxruntime as ort

from ultralytics import YOLO

# Ensure parent directory is in sys.path to allow imports from shared
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from shared.config import LOG_FILE_PATH, DATASETS_DIR, calculate_severity

# 로그 설정 (콘솔 및 파일 동시 출력)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_onnx_inference_pipeline(version_name, class_thresholds=None):
    # 1. PyTorch 모델 경로 확인 및 ONNX 변환 준비
    model_folder_name = f"steel_yolov8n_{version_name}"
    pt_model_path = os.path.join(REPO_ROOT, "runs", "detect", model_folder_name, "weights", "best.pt")
    
    if not os.path.exists(pt_model_path):
        logging.error(f"PyTorch 모델 파일을 찾을 수 없습니다: {pt_model_path}")
        return
        
    onnx_dir = os.path.join(REPO_ROOT, "runs", "detect", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    
    # ONNX 파일 경로 (모델 이름 사용)
    onnx_model_path = os.path.join(onnx_dir, f"{model_folder_name}.onnx")
    
    # ONNX 파일이 없으면 변환(export) 진행
    if not os.path.exists(onnx_model_path):
        logging.info(f"PyTorch 모델 ONNX 변환 진행 중: {pt_model_path} -> {onnx_model_path}")
        pt_model = YOLO(pt_model_path)
        exported_path = pt_model.export(format="onnx", dynamic=True, simplify=True)
        
        # export 기본 경로는 weights 폴더이므로 target 경로로 이동
        if os.path.exists(exported_path) and exported_path != onnx_model_path:
            os.replace(exported_path, onnx_model_path)
        logging.info(f"ONNX 변환 완료: {onnx_model_path}")
    else:
        logging.info(f"기존 ONNX 모델 사용: {onnx_model_path}")

    # 2. ONNX 모델 로드
    model = YOLO(onnx_model_path, task='detect')
    
    # 3. 검증 데이터셋(Validation Set) 시각화 및 지표 평가 (val)
    logging.info("ONNX 검증 데이터셋(Validation Set) 평가 시작...")
    data_yaml_path = os.path.join(REPO_ROOT, 'data.yaml')
    val_save_dir = os.path.join(onnx_dir, f"onnx_val_{version_name}")
    
    metrics = model.val(
        data=data_yaml_path,
        plots=True,
        exist_ok=True,
        project=onnx_dir,
        name=f"onnx_val_{version_name}"
    )
    
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    
    speed_pre = float(metrics.speed.get('preprocess', 0.0))
    speed_inf = float(metrics.speed.get('inference', 0.0))
    speed_post = float(metrics.speed.get('postprocess', 0.0))
    
    logging.info(f"=== [ONNX 모델 평가 요약] ===")
    logging.info(f"Model Version: {version_name}")
    logging.info(f"Mean Precision: {precision:.4f}")
    logging.info(f"Mean Recall: {recall:.4f}")
    logging.info(f"mAP50: {map50:.4f}")
    logging.info(f"Inference Speed: {speed_inf:.2f} ms/img (Pre: {speed_pre:.2f}ms, Post: {speed_post:.2f}ms)")
    logging.info(f"=============================")

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
    logging.info(f"ONNX 검증 이미지 추론 실행 중: {val_images_path}")
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
                
                # 클래스별 고유 임계값 적용 2차 필터링
                target_conf = CLASS_THRESHOLDS.get(class_name, 0.25)
                if confidence < target_conf:
                    continue
                    
                total_defects += 1
                
                # xywh 정규화 좌표
                xywhn = box.xywhn[0].tolist()
                x_center, y_center, width, height = xywhn
                
                # 심각도 판정
                severity = calculate_severity(confidence, width, height)
                
                defect_list.append({
                    'image_name': img_name,
                    'class_name': class_name,
                    'confidence': round(confidence, 4),
                    'x_center': round(x_center, 6),
                    'y_center': round(y_center, 6),
                    'width': round(width, 6),
                    'height': round(height, 6),
                    'severity': severity
                })

    # 5. CSV 파일로 저장 (DB 저장 대체)
    os.makedirs(val_save_dir, exist_ok=True)
    
    # 세부 결함 내역 CSV
    defect_csv_path = os.path.join(val_save_dir, 'defect_details.csv')
    csv_headers = ['image_name', 'class_name', 'confidence', 'x_center', 'y_center', 'width', 'height', 'severity']
    
    with open(defect_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(defect_list)

    # 요약 정보 CSV
    summary_csv_path = os.path.join(val_save_dir, 'inference_summary.csv')
    with open(summary_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['version_name', 'total_images', 'total_defects', 'precision', 'recall', 'map50', 'speed_preprocess_ms', 'speed_inference_ms', 'speed_postprocess_ms'])
        writer.writerow([version_name, total_images, total_defects, round(precision, 4), round(recall, 4), round(map50, 4), round(speed_pre, 2), round(speed_inf, 2), round(speed_post, 2)])
        
    logging.info(f"=== [ONNX 추론 및 CSV 저장 완료] ===")
    logging.info(f"총 이미지 수: {total_images}")
    logging.info(f"총 검출 결함 수: {total_defects}")
    logging.info(f"순수 추론 속도: {speed_inf:.2f} ms/장 (전처리: {speed_pre:.2f}ms, 후처리: {speed_post:.2f}ms)")
    logging.info(f"세부 결함 CSV: {defect_csv_path}")
    logging.info(f"평가 요약 CSV: {summary_csv_path}")
    logging.info(f"시각화 리포트 저장 폴더: {val_save_dir}")
    logging.info(f"=====================================")

if __name__ == '__main__':
    version = input("ONNX 변환 및 추론/평가를 진행할 모델 버전을 입력하세요 (예: v1_base): ").strip()
    if not version:
        version = "v1_base"
    run_onnx_inference_pipeline(version)

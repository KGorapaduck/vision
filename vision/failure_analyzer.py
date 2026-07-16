import os
import glob
import cv2
import sys
import argparse

# SCRIPT_DIR와 REPO_ROOT를 sys.path에 추가하여 shared 모듈을 불러올 수 있도록 보정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from shared.config import DEFECT_CLASSES, DATASETS_DIR, RUNS_DIR
from ultralytics import YOLO

def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    return intersection / union

def xywhn2xyxy(box):
    xc, yc, w, h = box
    xmin = xc - w / 2
    ymin = yc - h / 2
    xmax = xc + w / 2
    ymax = yc + h / 2
    return [xmin, ymin, xmax, ymax]

def run_failure_analysis(model_path, output_name, conf_threshold=0.25):
    """
    YOLO 검증 에러를 분석하고 시각화하여 runs/detect/failure_visuals_[output_name] 폴더에 저장합니다.
    """
    val_img_dir = os.path.join(DATASETS_DIR, "NEU-DET-YOLO", "images", "val")
    val_lbl_dir = os.path.join(DATASETS_DIR, "NEU-DET-YOLO", "labels", "val")
    
    # 지정된 output_name 그대로 결과 저장 폴더 경로 생성
    output_dir = os.path.join(RUNS_DIR, "detect", f"failure_visuals_{output_name}")
    
    # 디렉토리 생성
    subdirs = ["missed", "misclassified", "low_confidence"]
    for sd in subdirs:
        os.makedirs(os.path.join(output_dir, sd), exist_ok=True)
        
    print(f"\n[{output_name}] 모델 실패 분석 및 고해상도 시각화를 시작합니다 (conf=0.01)...")
    model = YOLO(model_path)
    
    results = model.predict(source=val_img_dir, conf=0.01, save=False, verbose=False)
    
    total_gt_boxes = 0
    stats = {
        "Correct": 0,
        "Missed": 0,
        "Misclassified": 0,
        "LowConfidence": 0
    }
    
    for r in results:
        img_path = r.path
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(val_lbl_dir, label_name)
        
        # 1. 정답 라벨 읽기
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.read().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    gt_boxes.append({
                        "class_id": cls_id,
                        "box": xywhn2xyxy([xc, yc, w, h])
                    })
                    
        # 2. 예측 결과 읽기
        pred_boxes = []
        if len(r.boxes) > 0:
            for box_coords, cls_id, conf in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                pred_boxes.append({
                    "class_id": int(cls_id),
                    "box": box_coords,
                    "conf": conf
                })
                
        # 시각화용 이미지 로드 및 3배 확대
        img = cv2.imread(img_path)
        img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC)
        h_img, w_img = img.shape[:2]
        
        has_error = False
        error_types = set()
        
        # 3. 매칭 및 그리기
        for gt in gt_boxes:
            total_gt_boxes += 1
            gt_cls = gt["class_id"]
            gt_cls_name = DEFECT_CLASSES[gt_cls]
            
            max_iou = 0
            best_pred = None
            
            for pred in pred_boxes:
                iou = bbox_iou(gt["box"], pred["box"])
                if iou > max_iou:
                    max_iou = iou
                    best_pred = pred
                    
            gt_xmin = int(gt["box"][0] * w_img)
            gt_ymin = int(gt["box"][1] * h_img)
            gt_xmax = int(gt["box"][2] * w_img)
            gt_ymax = int(gt["box"][3] * h_img)
            
            cv2.rectangle(img, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{gt_cls_name}", (gt_xmin, max(20, gt_ymin - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if max_iou < 0.25:
                has_error = True
                error_types.add("missed")
                stats["Missed"] += 1
            else:
                pred_cls_name = DEFECT_CLASSES[best_pred["class_id"]]
                pred_conf = best_pred["conf"]
                
                p_xmin = int(best_pred["box"][0] * w_img)
                p_ymin = int(best_pred["box"][1] * h_img)
                p_xmax = int(best_pred["box"][2] * w_img)
                p_ymax = int(best_pred["box"][3] * h_img)
                
                cv2.rectangle(img, (p_xmin, p_ymin), (p_xmax, p_ymax), (0, 0, 255), 2)
                cv2.putText(img, f"PR:{pred_cls_name}({pred_conf:.2f})", (p_xmin, min(h_img - 8, p_ymax + 18)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if isinstance(conf_threshold, dict):
                    target_conf = conf_threshold.get(pred_cls_name, 0.25)
                else:
                    target_conf = conf_threshold

                if best_pred["class_id"] != gt_cls:
                    has_error = True
                    error_types.add("misclassified")
                    stats["Misclassified"] += 1
                elif pred_conf < target_conf:
                    has_error = True
                    error_types.add("low_confidence")
                    stats["LowConfidence"] += 1
                else:
                    stats["Correct"] += 1
                    
        if has_error:
            for et in error_types:
                cv2.imwrite(os.path.join(output_dir, et, img_name), img)
                
    print("\n" + "="*50)
    print(f"       YOLOv8n FAILURE REPORT: {output_name}       ")
    print("="*50)
    print(f"Total Evaluated Boxes: {total_gt_boxes}")
    print(f"1. Correct: {stats['Correct']} ({stats['Correct']/total_gt_boxes*100:.1f}%)")
    print(f"2. Missed: {stats['Missed']} ({stats['Missed']/total_gt_boxes*100:.1f}%)")
    print(f"3. Misclassified: {stats['Misclassified']} ({stats['Misclassified']/total_gt_boxes*100:.1f}%)")
    print(f"4. Low Confidence: {stats['LowConfidence']} ({stats['LowConfidence']/total_gt_boxes*100:.1f}%)")
    print("="*50)
    print(f"🎉 시각화 결과 저장 완료: {output_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 실패 원인 정량/정성적 분석 및 시각화")
    parser.add_argument("--model-version", type=str, default="v1_base", help="검증할 모델의 버전 이름 (예: v1_base, v2_augmented)")
    parser.add_argument("--output-name", type=str, default=None, help="결과 폴더 이름 (미지정 시 model-version과 동일하게 설정)")
    parser.add_argument("--model-path", type=str, default=None, help="평가할 YOLO 가중치 파일(.pt) 경로 (수동 지정 시 --model-version 무시)")

    args = parser.parse_args()

    # 1. 모델 가중치 경로 자동 조립 또는 수동 로드
    model_path = args.model_path
    if not model_path:
        model_path = os.path.join(REPO_ROOT, "runs", "detect", f"steel_yolov8n_{args.model_version}", "weights", "best.pt")

    # 2. 결과 폴더 식별 이름 설정 (CLI 입력이 없으면 터미널에서 대화식으로 질문)
    output_name = args.output_name
    if not output_name:
        try:
            prompt_msg = f"결과 저장 폴더 이름 식별자({args.model_version}_[입력값])을 입력해 주세요 (Enter 입력 시 기본값 자동 설정): "
            user_input = input(prompt_msg).strip()
            if user_input:
                output_name = f"{args.model_version}_{user_input}"
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass

    if not output_name:
        output_name = args.model_version

    # 3. 클래스별 임계값 — 변경 시 이 딕셔너리만 직접 수정
    CLASS_THRESHOLDS = {
        "crazing":        0.15,
        "inclusion":      0.20,
        "patches":        0.25,
        "pitted_surface": 0.20,
        "rolled-in_scale":0.15,
        "scratches":      0.25
    }

    if os.path.exists(model_path):
        print(f"[INFO] 클래스별 개별 임계값 적용: {CLASS_THRESHOLDS}")
        run_failure_analysis(model_path, output_name, conf_threshold=CLASS_THRESHOLDS)
    else:
        print(f"[ERROR] 지정된 모델 파일이 없습니다: {model_path}")
        print("올바른 --model-version 명칭을 입력하셨거나 --model-path 전체 경로가 유효한지 다시 확인해 주세요.")
        sys.exit(1)


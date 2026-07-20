import os
import sys
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Ensure parent directory is in sys.path to allow imports from shared
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from shared.config import DEFECT_CLASSES, RAW_DATA_DIR, DATASETS_DIR

# ==========================================
# [설정 정의]
# ==========================================
BASEDIR = os.path.join(RAW_DATA_DIR, 'NEU-DET')  # 다운로드받은 원본 NEU-DET 데이터가 들어있는 폴더
OUTPUTDIR = os.path.join(DATASETS_DIR, 'NEU-DET-YOLO')  # YOLO 포맷으로 변환되어 저장될 경로
CLASSES = DEFECT_CLASSES


# ==========================================
# [1단계: XML (Pascal VOC) -> YOLO TXT 포맷 변환 함수]
# ==========================================
def convert_to_yolo_format(xml_file, img_width, img_height):
    """
    Pascal VOC XML 파일의 절대 좌표(xmin, ymin, xmax, ymax)를 
    YOLO의 상대 좌표(x_center, y_center, width, height)로 변환하는 함수입니다.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []

    # XML 내부의 모든 객체(Object) 탐색
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASSES: 
            continue
        
        class_id = CLASSES.index(name)  # 클래스 인덱스 부여 (0 ~ 5)
        
        # XML 바운딩 박스 절대좌표 추출
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # YOLO 포맷 계산 공식 (0~1 사이의 상대값)
        xcenter = (xmin + xmax) / 2 / img_width
        ycenter = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # 결과를 문자열로 저장
        yolo_lines.append(f'{class_id} {xcenter} {ycenter} {width} {height}')
        
    return yolo_lines


# ==========================================
# [2단계: 데이터 수집 및 3단계 Stratified Split (Train 70% / Val 15% / Test 15%)]
# ==========================================
def collect_all_data_items():
    """
    rawdata/NEU-DET/train 및 validation 내 모든 XML/이미지 쌍을 클래스별로 교차 수색하여 수집합니다.
    (XML과 JPG가 서로 다른 서브 폴더에 교차 유실 배치된 경우도 완전 탐색 매칭)
    """
    items_by_class = {c: [] for c in CLASSES}
    
    for source_split in ['train', 'validation']:
        xml_source = os.path.join(BASEDIR, source_split, 'annotations')
        if not os.path.exists(xml_source):
            continue
            
        xml_files = [f for f in os.listdir(xml_source) if f.endswith('.xml')]
        for xml_file in xml_files:
            class_name = xml_file.rsplit('_', 1)[0]
            if class_name not in items_by_class:
                continue
                
            img_filename_jpg = xml_file.replace('.xml', '.jpg')
            actual_xml_path = os.path.join(xml_source, xml_file)
            
            # train 및 validation 경로 양쪽을 교차 수색하여 JPG 실체 위치 탐색
            actual_img_path = None
            for check_split in ['train', 'validation']:
                candidate_img_path = os.path.join(BASEDIR, check_split, 'images', class_name, img_filename_jpg)
                if os.path.exists(candidate_img_path):
                    actual_img_path = candidate_img_path
                    break
                    
            if actual_img_path is not None:
                items_by_class[class_name].append({
                    'xml_path': actual_xml_path,
                    'img_path': actual_img_path,
                    'xml_file': xml_file,
                    'img_file': img_filename_jpg
                })
            else:
                print(f"Warning: {xml_file} 에 대응하는 이미지 파일을 찾을 수 없어 제외되었습니다.")
            
    # 각 클래스별 파일 정렬 (crazing_1 ~ crazing_300 순서 보장)
    for c in CLASSES:
        items_by_class[c].sort(key=lambda x: int(x['xml_file'].rsplit('_', 1)[1].replace('.xml', '')) if x['xml_file'].rsplit('_', 1)[1].replace('.xml', '').isdigit() else x['xml_file'])
        
    return items_by_class


def process_items_for_split(items, split_name, apply_clahe=True):
    """
    지정된 분할(train/val/test)의 아이템 목록을 YOLO 포맷으로 가공 및 복사합니다.
    apply_clahe=False일 경우 원본 이미지를 그대로 복사합니다.
    """
    print(f'Processing {split_name} data ({len(items)} items, CLAHE={apply_clahe})...')
    img_dest = os.path.join(OUTPUTDIR, 'images', split_name)
    label_dest = os.path.join(OUTPUTDIR, 'labels', split_name)
    
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)
    
    for item in tqdm(items):
        try:
            # XML에서 YOLO 포맷 텍스트 추출 (NEU-DET 200x200)
            yolo_data = convert_to_yolo_format(item['xml_path'], 200, 200)
            
            # 텍스트 라벨 파일 저장
            txt_filename = item['xml_file'].replace('.xml', '.txt')
            with open(os.path.join(label_dest, txt_filename), 'w') as f:
                f.write('\n'.join(yolo_data))
                
            # 이미지 전처리 및 복사
            if os.path.exists(item['img_path']):
                if apply_clahe:
                    img = cv2.imread(item['img_path'], cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced_img = clahe.apply(img)
                        cv2.imwrite(os.path.join(img_dest, item['img_file']), enhanced_img)
                    else:
                        shutil.copy(item['img_path'], os.path.join(img_dest, item['img_file']))
                else:
                    shutil.copy(item['img_path'], os.path.join(img_dest, item['img_file']))
                    
        except Exception as e:
            print(f"Error processing {item['xml_file']}: {e}")


# ==========================================
# [3단계: 가공 작업 수행 및 data.yaml 설정 생성]
# ==========================================
def prepare_dataset(apply_clahe=True):
    # 기존 가공 폴더 초기화
    if os.path.exists(OUTPUTDIR):
        shutil.rmtree(OUTPUTDIR)
        
    items_by_class = collect_all_data_items()
    
    train_items, val_items, test_items = [], [], []
    
    # 각 클래스(300장)별로 70%(210장) / 15%(45장) / 15%(45장) Stratified Split
    for c, items in items_by_class.items():
        total_count = len(items)
        train_end = int(total_count * 0.70)  # 210
        val_end = train_end + int(total_count * 0.15)  # 255
        
        train_items.extend(items[:train_end])
        val_items.extend(items[train_end:val_end])
        test_items.extend(items[val_end:])
        
    print(f"Stratified Split 완료: Train {len(train_items)}장, Val {len(val_items)}장, Test {len(test_items)}장 (총 {len(train_items)+len(val_items)+len(test_items)}장)")
    
    process_items_for_split(train_items, 'train', apply_clahe=apply_clahe)
    process_items_for_split(val_items, 'val', apply_clahe=apply_clahe)
    process_items_for_split(test_items, 'test', apply_clahe=apply_clahe)
    
    print(f'Data is ready at {OUTPUTDIR}')

    # YOLO 학습에 필요한 데이터셋 정보 YAML 파일 작성 (test 경로 포함)
    yaml_content = f"""path: {OUTPUTDIR}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, name in enumerate(CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    data_yaml_path = os.path.join(REPO_ROOT, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f'data.yaml has been created at {data_yaml_path}!')


# ==========================================
# [4단계: YOLO 모델 로드 및 학습]
# ==========================================
def train_model(version_name):
    # 사전 학습된 YOLOv8n(Nano) 모델 가중치 로드
    yolo_model_path = os.path.join(REPO_ROOT, 'yolov8n.pt')
    model = YOLO(yolo_model_path)
    
    data_yaml_path = os.path.join(REPO_ROOT, 'data.yaml')
    
    # 버전 이름(v1_base vs v2_augmented)에 따른 하이퍼파라미터 분기
    if 'v1' in version_name.lower():
        epochs = 30
        imgsz = 224
        degrees = 0.0
        flipud = 0.0
        print(f"[{version_name}] v1 Baseline 셋업 적용: epochs={epochs}, imgsz={imgsz}, Augmentation=Off")
    else:
        epochs = 50
        imgsz = 640
        degrees = 15.0
        flipud = 0.5
        print(f"[{version_name}] v2 Augmented 셋업 적용: epochs={epochs}, imgsz={imgsz}, degrees={degrees}, flipud={flipud}")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        degrees=degrees,
        flipud=flipud,
        exist_ok=True,
        name=f"steel_yolov8n_{version_name}"
    )
    return model, results.save_dir


# ==========================================
# [5단계: 학습 완료 모델 검증 및 추론 결과 시각화]
# ==========================================
def validate_and_visualize(model_path):
    """
    학습된 최적 가중치를 불러와 검증 데이터셋에 대해 성능을 평가하고 예측 이미지를 시각화합니다.
    """
    model = YOLO(model_path)
    
    # 5.1 검증 데이터셋 예측 수행 및 결과 이미지 자동 저장
    val_images_path = os.path.join(OUTPUTDIR, 'images', 'val')
    results = model.predict(source=val_images_path, conf=0.25, save=True)
    
    # 5.2 저장된 폴더에서 샘플 3장을 가져와 Matplotlib으로 화면에 출력
    save_dir = results[0].save_dir
    files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))][:3]
    
    if files:
        plt.figure(figsize=(15, 5))
        for i, file in enumerate(files):
            img = cv2.imread(os.path.join(save_dir, file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, len(files), i+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'Prediction {i+1}')
        plt.show()

    # 5.3 최종 검증 지표 출력 (mAP50, Precision, Recall)
    metrics = model.val()
    print(f'mAP50 Accuracy: {metrics.box.map50:.3f}')
    print(f'Precision: {metrics.box.mp:.3f}')
    print(f'Recall: {metrics.box.mr:.3f}')


if __name__ == '__main__':
    if os.path.exists(BASEDIR):
        if len(sys.argv) > 1:
            version_name = sys.argv[1].strip()
        else:
            version_name = input("학습할 모델의 버전 이름을 입력해 주세요 (예: v1_base, v2_augmented): ").strip()
        if not version_name:
            version_name = "v1_base"
            
        # v1 버전일 경우 CLAHE 미적용, v2 버전일 경우 CLAHE 적용
        apply_clahe = False if 'v1' in version_name.lower() else True
        
        prepare_dataset(apply_clahe=apply_clahe)
        model, save_dir = train_model(version_name)
        
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        validate_and_visualize(best_model_path)
    else:
        print(f"'{BASEDIR}' 폴더가 존재하지 않습니다. 먼저 원본 데이터셋을 다운로드해 주세요.")



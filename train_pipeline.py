import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# ==========================================
# [설정 정의]
# ==========================================
BASEDIR = 'rawdata'  # 다운로드받은 원본 NEU-DET 데이터가 들어있는 폴더
OUTPUTDIR = 'datasets/NEU-DET-YOLO'  # YOLO 포맷으로 변환되어 저장될 경로

# NEU-DET 데이터셋이 가진 6가지 결함 종류 정의
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


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
# [2단계: 폴더별 데이터 가공 및 정렬]
# ==========================================
def process_folder(split_name, source_split_name):
    """
    원본 이미지와 어노테이션 XML을 읽어, 지정된 YOLO 디렉터리 구조로 분류 복사합니다.
    """
    print(f'Processing {split_name} data...')
    img_source = os.path.join(BASEDIR, source_split_name, 'images')
    xml_source = os.path.join(BASEDIR, source_split_name, 'annotations')
    
    img_dest = os.path.join(OUTPUTDIR, 'images', split_name)
    label_dest = os.path.join(OUTPUTDIR, 'labels', split_name)
    
    # 저장될 폴더 자동 생성
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)
    
    xml_files = [f for f in os.listdir(xml_source) if f.endswith('.xml')]
    
    # 진행바(tqdm)와 함께 변환 및 이미지 복사 작업 수행
    for xml_file in tqdm(xml_files):
        try:
            # XML에서 YOLO 포맷 텍스트 추출 (NEU-DET 데이터는 200x200 크기 고정)
            yolo_data = convert_to_yolo_format(os.path.join(xml_source, xml_file), 200, 200)
            
            # 텍스트 파일 저장
            txt_filename = xml_file.replace('.xml', '.txt')
            with open(os.path.join(label_dest, txt_filename), 'w') as f:
                f.write('\n'.join(yolo_data))
                
            # 이미지 복사
            img_filename_jpg = xml_file.replace('.xml', '.jpg')
            if os.path.exists(os.path.join(img_source, img_filename_jpg)):
                shutil.copy(os.path.join(img_source, img_filename_jpg), os.path.join(img_dest, img_filename_jpg))
                
        except Exception as e:
            print(f'Error processing {xml_file}: {e}')


# ==========================================
# [3단계: 가공 작업 수행 및 data.yaml 설정 생성]
# ==========================================
def prepare_dataset():
    # 학습(train) 및 검증(val) 데이터셋 분류 작업 수행
    process_folder('train', 'train')
    process_folder('val', 'validation')
    print(f'Data is ready at {OUTPUTDIR}')

    # YOLO 학습에 필요한 데이터셋 정보 YAML 파일 자동 작성
    yaml_content = """path: ./datasets/NEU-DET-YOLO
train: images/train
val: images/val

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    print('data.yaml has been created!')


# ==========================================
# [4단계: YOLO 모델 로드 및 학습]
# ==========================================
def train_model():
    # 사전 학습된 YOLOv8n(Nano) 모델 가중치를 인터넷에서 로컬로 다운로드 및 로드
    model = YOLO('yolov8n.pt')
    
    print("Starting training...")
    # data.yaml의 설정을 읽어 30 에폭 동안 학습 (이미지 크기 200x200)
    results = model.train(data='data.yaml', epochs=30, imgsz=200)
    return model


# ==========================================
# [5단계: 학습 완료 모델 검증 및 추론 결과 시각화]
# ==========================================
def validate_and_visualize(model_path='runs/detect/train/weights/best.pt'):
    """
    학습된 최적 가중치를 불러와 검증 데이터셋에 대해 성능을 평가하고 예측 이미지를 시각화합니다.
    """
    model = YOLO(model_path)
    
    # 5.1 검증 데이터셋 예측 수행 및 결과 이미지 자동 저장
    results = model.predict(source='datasets/NEU-DET-YOLO/images/val', conf=0.25, save=True)
    
    # 5.2 저장된 폴더에서 샘플 3장을 가져와 Matplotlib으로 화면에 출력
    save_dir = results[0].save_dir
    files = os.listdir(save_dir)[:3]
    
    plt.figure(figsize=(15, 5))
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(save_dir, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, i+1)
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
    # 훈련에 필요한 데이터 폴더가 있는지 체크 후 가공 단계 실행
    if os.path.exists(BASEDIR):
        prepare_dataset()
        model = train_model()
        validate_and_visualize()
    else:
        print(f"'{BASEDIR}' 폴더가 존재하지 않습니다. 먼저 원본 데이터셋을 다운로드해 주세요.")

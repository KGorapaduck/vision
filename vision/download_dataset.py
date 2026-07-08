import os
import shutil
import sys

# Ensure parent directory is in sys.path to allow imports from shared
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from shared.config import RAW_DATA_DIR

# kagglehub 라이브러리가 없으면 자동 설치
try:
    import kagglehub
except ImportError:
    print("kagglehub 라이브러리가 없어 설치를 시작합니다...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

print("Kaggle에서 NEU Surface Defect Database 다운로드를 시작합니다...")
# kagglehub를 이용해 로그인 없이 public 데이터셋 다운로드
download_path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print(f"다운로드 완료! 임시 경로: {download_path}")

# 복사할 로컬 목적지 폴더 정의
dest_dir = RAW_DATA_DIR
os.makedirs(dest_dir, exist_ok=True)

print(f"임시 경로의 데이터를 {dest_dir} 폴더로 복사합니다...")

# 다운로드된 파일/폴더들을 rawdata 폴더로 복사
for item in os.listdir(download_path):
    source_item = os.path.join(download_path, item)
    dest_item = os.path.join(dest_dir, item)
    
    if os.path.isdir(source_item):
        if os.path.exists(dest_item):
            shutil.rmtree(dest_item)
        shutil.copytree(source_item, dest_item)
        print(f"폴더 복사 완료: {item}")
    else:
        shutil.copy2(source_item, dest_item)
        print(f"파일 복사 완료: {item}")

print("\n🎉 모든 데이터셋 파일이 'rawdata' 폴더에 성공적으로 세팅되었습니다!")
print("이제 'python -m vision.train_pipeline'을 실행하여 학습을 진행할 수 있습니다.")


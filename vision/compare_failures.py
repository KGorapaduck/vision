import os
import shutil
import argparse
import sys

# SCRIPT_DIR와 REPO_ROOT를 sys.path에 추가하여 shared 모듈을 불러올 수 있도록 보정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RUNS_DIR = os.path.join(REPO_ROOT, "runs", "detect")

def get_failure_visual_folders():
    if not os.path.exists(RUNS_DIR):
        return []
    folders = [f for f in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, f)) and f.startswith("failure_visuals_")]
    return sorted(folders)

def main():
    parser = argparse.ArgumentParser(description="두 개의 실패 분석 결과 폴더를 비교하여 개선된 이미지들을 tmp 폴더로 복사합니다.")
    parser.add_argument("--base", type=str, help="기준이 되는 failure_visuals 폴더명 (예: failure_visuals_v1_base)")
    parser.add_argument("--target", type=str, help="비교 대상 failure_visuals 폴더명 (예: failure_visuals_v1_base_conf0.15)")
    args = parser.parse_args()

    folders = get_failure_visual_folders()
    
    base_folder = args.base
    target_folder = args.target

    # 인자가 주어지지 않았을 경우 인터랙티브하게 선택 유도
    if not base_folder or not target_folder:
        if len(folders) < 2:
            print("[ERROR] 비교할 수 있는 failure_visuals_ 폴더가 최소 2개 이상 존재해야 합니다.")
            print(f"현재 탐색된 폴더 목록: {folders}")
            sys.exit(1)
        
        print("\n=== 사용 가능한 실패 시각화 폴더 목록 ===")
        for idx, f in enumerate(folders):
            print(f"[{idx}] {f}")
        
        try:
            base_idx = int(input("\n기준 폴더 (Base, 예: Conf가 높은 폴더) 번호를 선택하세요: "))
            target_idx = int(input("비교 폴더 (Target, 예: Conf를 낮춘 폴더) 번호를 선택하세요: "))
            
            base_folder = folders[base_idx]
            target_folder = folders[target_idx]
        except (ValueError, IndexError):
            print("[ERROR] 올바른 번호를 선택해 주세요.")
            sys.exit(1)

    base_path = os.path.join(RUNS_DIR, base_folder)
    target_path = os.path.join(RUNS_DIR, target_folder)

    if not os.path.exists(base_path) or not os.path.exists(target_path):
        print(f"[ERROR] 지정된 폴더가 존재하지 않습니다.\nBase: {base_path}\nTarget: {target_path}")
        sys.exit(1)

    # 1. TMP 폴더 준비 및 초기화 (runs/detect/tmp)
    tmp_dir = os.path.join(RUNS_DIR, "tmp")
    if os.path.exists(tmp_dir):
        print(f"\n기존 TMP 폴더를 비우는 중: {tmp_dir}")
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"\n비교 시작:")
    print(f"  - 기준 폴더 (Base)  : {base_folder}")
    print(f"  - 비교 폴더 (Target): {target_folder}")
    print(f"  - 결과 저장 폴더 (TMP): {tmp_dir}\n")

    subdirs = ["missed", "misclassified", "low_confidence"]
    total_improved = 0

    for sd in subdirs:
        base_sd_path = os.path.join(base_path, sd)
        target_sd_path = os.path.join(target_path, sd)
        
        base_files = set(os.listdir(base_sd_path)) if os.path.exists(base_sd_path) else set()
        target_files = set(os.listdir(target_sd_path)) if os.path.exists(target_sd_path) else set()
        
        # Base 에는 존재하지만 Target 에는 존재하지 않는 이미지 (즉, 개선됨)
        improved_files = sorted(list(base_files - target_files))
        count = len(improved_files)
        total_improved += count
        
        print(f"[{sd}] 카테고리:")
        print(f"  - 기존 에러: {len(base_files)}장 -> 변경 후 에러: {len(target_files)}장")
        print(f"  - 개선된(오류 탈출) 개수: {count}장")
        
        if count > 0:
            # TMP 내부에 카테고리별 하위 폴더 생성
            tmp_sd_dir = os.path.join(tmp_dir, sd)
            os.makedirs(tmp_sd_dir, exist_ok=True)
            
            for file_name in improved_files:
                src_file = os.path.join(base_sd_path, file_name)
                dst_file = os.path.join(tmp_sd_dir, file_name)
                shutil.copy2(src_file, dst_file)
            print(f"  -> 개선된 이미지 {count}장을 {tmp_sd_dir} 로 복사 완료.")
        print()

    print("=" * 50)
    print(f"비교 분석 및 복사 완료! 총 {total_improved}장의 이미지가 개선되었습니다.")
    print(f"결과 확인 (TMP 폴더): {tmp_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()

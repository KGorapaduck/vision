import sqlite3
import os
from datetime import datetime

DB_PATH = 'steel_defects.db'

def get_db_connection():
    """SQLite 데이터베이스 연결을 생성하여 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """데이터베이스 테이블이 없을 경우 자동으로 테이블을 초기화 생성합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. inference_runs 테이블 생성 (각 훈련/검증 세션 요약)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            total_images INTEGER NOT NULL,
            total_defects INTEGER NOT NULL,
            precision REAL,
            recall REAL,
            map50 REAL,
            inference_time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. defect_details 테이블 생성 (개별 결함 검출 내역)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS defect_details (
            defect_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            image_name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            box_x_center REAL NOT NULL,
            box_y_center REAL NOT NULL,
            box_width REAL NOT NULL,
            box_height REAL NOT NULL,
            severity TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES inference_runs(run_id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    print("SQLite Database initialized successfully.")

def calculate_severity(confidence, box_width, box_height):
    """
    결함의 심각도를 자동 판정하는 로직입니다.
    - YOLOv8 좌표계는 0~1로 정규화되어 있으므로, box_width * box_height 가 면적 비율(area_ratio)이 됩니다.
    - 면적 비율이 10% 이상이거나 신뢰도(Confidence)가 0.9 이상이면 'HIGH'
    - 면적 비율이 5% 이상이거나 신뢰도(Confidence)가 0.7 이상이면 'MEDIUM'
    - 그 외에는 'LOW'로 판정합니다.
    """
    area_ratio = box_width * box_height
    
    if area_ratio >= 0.10 or confidence >= 0.90:
        return 'HIGH'
    elif area_ratio >= 0.05 or confidence >= 0.70:
        return 'MEDIUM'
    else:
        return 'LOW'

def insert_inference_run(model_version, total_images, total_defects, precision, recall, map50):
    """검증 실행 요약 정보를 저장하고 고유 run_id를 반환합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO inference_runs (model_version, total_images, total_defects, precision, recall, map50)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (model_version, total_images, total_defects, precision, recall, map50))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def insert_defect_details(defect_list):
    """
    결함 상세 리스트를 벌크(Bulk)로 데이터베이스에 저장합니다.
    defect_list 원소 튜플 구조: 
    (run_id, image_name, class_name, confidence, x_center, y_center, width, height, severity)
    """
    if not defect_list:
        return
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.executemany('''
        INSERT INTO defect_details (
            run_id, image_name, class_name, confidence, 
            box_x_center, box_y_center, box_width, box_height, severity
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', defect_list)
    
    conn.commit()
    conn.close()
    print(f"Logged {len(defect_list)} defect details to DB.")

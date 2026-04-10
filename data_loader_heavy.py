# coding=utf-8
# 파일명: data_loader_heavy.py
# 대규모 데이터셋(HIGGS, Criteo) 전용 데이터 로더

import os
import urllib.request
import numpy as np
import pandas as pd
from utils import binary_sampler


def data_loader_heavy(data_name, miss_rate, max_samples=100000):
    '''
    대용량 데이터셋 로더. 메모리 보호를 위해 max_samples 지정.
    '''
    os.makedirs('data', exist_ok=True)

    if data_name == 'higgs':
        # HIGGS 데이터셋 (약 1100만 행, 28개 특성)
        file_path = 'data/HIGGS.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                "HIGGS 데이터셋은 직접 다운로드가 필요합니다.\\n"
            )

        print(f'[Heavy Loader] HIGGS 데이터 읽는 중... (최대 {max_samples}행)')
        # 압축된 csv 파일에서 지정된 행만큼만 읽어옵니다 (첫 번째 열은 라벨이므로 제외)
        df = pd.read_csv(file_path, header=None, nrows=max_samples)
        data_x = df.iloc[:, 1:].values.astype(float)

    elif data_name == 'criteo':
        # Criteo 데이터셋 (약 4500만 행)
        # 주의: Criteo는 공식 다운로드 링크가 자주 변경되므로 수동 다운로드를 권장합니다.
        # 캐글(Kaggle) 등에서 다운받은 train.txt (혹은 csv)를 data 폴더에 넣어야 합니다.
        file_path = 'data/criteoDB/train.txt'

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                "Criteo 데이터셋은 직접 다운로드가 필요합니다.\\n"
                "Kaggle(Criteo Display Advertising Challenge)에서 train.txt를 다운받아 "
                "'data/criteoDB/train.txt'로 저장해주세요."
            )

        print(f'[Heavy Loader] Criteo 데이터 읽는 중... (최대 {max_samples}행)')
        # Criteo는 13개의 수치형 변수와 26개의 범주형(Hash) 변수로 구성됨.
        # GAN Imputation의 순수 성능 평가를 위해 13개의 수치형(Integer) 변수만 추출하여 사용
        df = pd.read_csv(file_path, sep='\\t', header=None, nrows=max_samples)
        data_x = df.iloc[:, 1:14].values.astype(float)

    else:
        raise ValueError(f"Unknown heavy dataset: {data_name}")

    # 결측치(Missing Values) 인위적 생성
    no, dim = data_x.shape
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    print(f"[Heavy Loader] {data_name} 로드 완료: 형태 {data_x.shape}, 결측률 {miss_rate * 100}%")
    return data_x, miss_data_x, data_m
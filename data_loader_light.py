# coding=utf-8
# Data loader — 5개 데이터셋 지원
# letter, spam, credit, breast, news

import os
import urllib.request
import numpy as np
from utils import binary_sampler

_UCI_URLS = {
    'spam'  : 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
    'letter': 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
    'credit': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
    'breast': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
    'news'  : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip',
}

def download_data(data_name):
    os.makedirs('data', exist_ok=True)
    file_path = f'data/{data_name}.csv'
    if not os.path.exists(file_path):
        print(f'[data_loader] {data_name}.csv 없음. 다운로드 시작...')
        if data_name in ['spam', 'letter']:
            urllib.request.urlretrieve(_UCI_URLS[data_name], file_path)
        elif data_name == 'breast':
            urllib.request.urlretrieve(_UCI_URLS['breast'], file_path)
        elif data_name == 'credit':
            _download_credit(file_path)
        elif data_name == 'news':
            _download_news(file_path)
        print(f'[data_loader] 완료 → {file_path}')
    else:
        print(f'[data_loader] {file_path} 이미 존재. 스킵.')

def _download_credit(file_path):
    '''Credit 데이터: xls → numpy 변환 후 저장.'''
    try:
        import pandas as pd
        xls_path = 'data/credit.xls'
        urllib.request.urlretrieve(_UCI_URLS['credit'], xls_path)
        df = pd.read_excel(xls_path, header=1)
        df = df.drop(columns=['ID'], errors='ignore')
        df.to_csv(file_path, index=False)
    except ImportError:
        raise ImportError('credit 데이터셋에는 pandas와 openpyxl이 필요합니다.\npip install pandas openpyxl xlrd')

def _download_news(file_path):
    '''News 데이터: zip → csv 변환.'''
    import zipfile, io
    zip_path = 'data/news.zip'
    urllib.request.urlretrieve(_UCI_URLS['news'], zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
        with z.open(names[0]) as f:
            content = f.read().decode('utf-8')
    with open(file_path, 'w') as f:
        f.write(content)

def data_loader(data_name, miss_rate):
    '''
    Returns:
        data_x      : 원본 데이터 (결측 없음)
        miss_data_x : 결측 삽입 데이터
        data_m      : 결측 마스크 (1=관측, 0=결측)
    '''
    download_data(data_name)

    if data_name == 'spam':
        data_x = np.loadtxt('data/spam.csv', delimiter=',')

    elif data_name == 'letter':
        raw    = np.genfromtxt('data/letter.csv', delimiter=',', dtype=str)
        data_x = raw[:, 1:].astype(float)

    elif data_name == 'breast':
        # wdbc.data: col0=ID, col1=diagnosis(M/B) → 둘 다 제거, col2~31 수치형
        raw    = np.genfromtxt('data/breast.csv', delimiter=',', dtype=str)
        data_x = raw[:, 2:].astype(float)

    elif data_name == 'credit':
        try:
            import pandas as pd
            df     = pd.read_csv('data/credit.csv')
            # 수치형 컬럼만 사용
            data_x = df.select_dtypes(include=[np.number]).values.astype(float)
        except ImportError:
            raise ImportError('pip install pandas openpyxl')

    elif data_name == 'news':
        try:
            import pandas as pd
            df = pd.read_csv('data/news.csv')
            # 수치형만, url/timedelta 컬럼 제거
            drop_cols = [c for c in df.columns if 'url' in c.lower() or 'timedelta' in c.lower()]
            df = df.drop(columns=drop_cols, errors='ignore')
            data_x = df.select_dtypes(include=[np.number]).dropna(axis=1).values.astype(float)
        except ImportError:
            raise ImportError('pip install pandas')
    else:
        raise ValueError(f"Unknown dataset: '{data_name}'. "
                         f"Choose from: letter, spam, credit, breast, news")

    no, dim = data_x.shape
    data_m  = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m

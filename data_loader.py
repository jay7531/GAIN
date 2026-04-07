# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Data loader for UCI letter, spam and MNIST datasets (PyTorch version).

- Keras dependency removed: MNIST is now loaded via torchvision.
- Auto-download added: letter / spam CSV files are fetched from UCI
  automatically if they are not found locally.
'''

import os
import urllib.request
import numpy as np
from utils import binary_sampler


# ── UCI download URLs ──────────────────────────────────────────────────────────
_UCI_URLS = {
    'spam'  : 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
    'letter': 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
}


def download_data(data_name):
    '''Download UCI dataset CSV if not already present in ./data/.

    Args:
        - data_name: 'letter' or 'spam'
    '''
    os.makedirs('data', exist_ok=True)
    file_path = f'data/{data_name}.csv'

    if not os.path.exists(file_path):
        url = _UCI_URLS[data_name]
        print(f'[data_loader] {data_name}.csv not found.')
        print(f'[data_loader] Downloading from UCI repository...')
        print(f'[data_loader] URL: {url}')
        urllib.request.urlretrieve(url, file_path)
        print(f'[data_loader] Download complete -> saved to {file_path}')
    else:
        print(f'[data_loader] {file_path} already exists. Skipping download.')


def data_loader(data_name, miss_rate):
    '''Load dataset and introduce missingness.

    Args:
        - data_name: 'letter', 'spam', or 'mnist'
        - miss_rate : probability of a component being missing

    Returns:
        - data_x     : original data  (no missing values)
        - miss_data_x: data with NaN inserted at missing positions
        - data_m     : binary mask  (1 = observed, 0 = missing)
    '''

    # ── Load raw data ──────────────────────────────────────────────
    if data_name == 'spam':
        download_data('spam')
        # spambase.data: no header, 57 numeric features + 1 label
        data_x = np.loadtxt('data/spam.csv', delimiter=',')

    elif data_name == 'letter':
        download_data('letter')
        # first column is a character class label (A-Z) -> drop it
        raw    = np.genfromtxt('data/letter.csv', delimiter=',', dtype=str)
        data_x = raw[:, 1:].astype(float)

    elif data_name == 'mnist':
        try:
            import torchvision
            import torchvision.transforms as transforms

            dataset = torchvision.datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            data_x = dataset.data.numpy().reshape(60000, 28 * 28).astype(float)
        except ImportError:
            raise ImportError(
                'torchvision is required for MNIST. '
                'Install it with: pip install torchvision'
            )
    else:
        raise ValueError(
            f"Unknown dataset: '{data_name}'. "
            f"Choose from 'letter', 'spam', 'mnist'."
        )

    # ── Introduce missing data ─────────────────────────────────────
    no, dim = data_x.shape
    data_m  = binary_sampler(1 - miss_rate, no, dim)

    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m

import subprocess
import sys


PREFIX = ['python', 'main.py']
for MODEL in ['mlp_small', 'mlp_big', 'svm']:
    for DATASET in ['iris', 'glass', 'awa2']:
        for TESTSIZE in ['0.2', '0.4', '0.5', '0.6', '0.8']:
            subprocess.run(PREFIX+[
                f'--model={MODEL}',
                f'--dataset={DATASET}',
                f'--test_size={TESTSIZE}'
            ])

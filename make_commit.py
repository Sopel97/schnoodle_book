from pathlib import Path
import random
import subprocess

for path in Path('./').rglob('*.*'):
    if '.git' in str(path):
        continue
    if random.random() < 0.5:
        subprocess.run(['git', 'add', path])

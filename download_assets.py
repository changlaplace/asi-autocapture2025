import os
from urllib.request import Request, urlopen

root = os.path.join(os.getcwd(), 'assets')
os.makedirs(root, exist_ok=True)

for i in range(1, 21):
    url = f'https://picsum.photos/seed/asi{i}/1600/1600'
    dest = os.path.join(root, f'asset_{i:02d}.jpg')
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req, timeout=30) as response:
        data = response.read()
    with open(dest, 'wb') as f:
        f.write(data)
    print(f'downloaded {dest} ({len(data)} bytes)')

print(f'Total files: {len(os.listdir(root))}')

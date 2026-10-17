from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
for path in sorted((root/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    print('PASS',path.name,flush=True)
print('All seven notebooks executed successfully.')

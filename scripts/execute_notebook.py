"""Execute the guided notebook using this interpreter without global kernel install."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
source = root / 'notebooks/01_guided_end_to_end.ipynb'
notebook = nbformat.read(source, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, timeout=900, km=manager,
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
output = root / 'notebooks/01_guided_end_to_end.executed.ipynb'
nbformat.write(notebook, output)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells: {output}')

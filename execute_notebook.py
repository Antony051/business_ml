import nbformat
from nbclient import NotebookClient

with open('notebooks/credit_risk_analysis.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=1200, kernel_name='python3')
client.execute()

with open('notebooks/credit_risk_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully.")

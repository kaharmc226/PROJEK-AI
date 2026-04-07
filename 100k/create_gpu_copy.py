import nbformat
import sys

nb_path = r'c:\Users\Administrator\Desktop\PROJEK AI\100k\insdata100k_tuned.ipynb'
out_path = r'c:\Users\Administrator\Desktop\PROJEK AI\100k\insdata100k_tuned_gpu.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

markdown_cell_added = False

for cell in nb.cells:
    if cell.cell_type == 'markdown' and not markdown_cell_added:
        instructions = """## **GPU Accelerated Notebook**
To run this notebook efficiently for 100k+ rows:
**Google Colab:**
1. Navigate to **Runtime > Change runtime type**
2. Select **T4 GPU** under Hardware accelerator and click Save.

**Kaggle:**
1. In notebook edit mode, expand the **Notebook Options** panel on the right.
2. Select **GPU T4x2** (or P100) under Accelerator.

The notebook uses `XGBRegressor` on GPU natively for faster training. Linear Regression and Random Forest stay on CPU (though RF could be updated to `cuRF`, XGBoost is usually more powerful).
"""
        new_cell = nbformat.v4.new_markdown_cell(instructions)
        nb.cells.insert(0, new_cell)
        markdown_cell_added = True

    if cell.cell_type == 'code':
        source = cell.source
        
        # Replace imports
        if 'from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor' in source:
            source = source.replace(
                'from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor',
                'from sklearn.ensemble import RandomForestRegressor as RFR\nfrom xgboost import XGBRegressor'
            )
        
        # Replace the class instantiation
        if 'GradientBoostingRegressor(**p)' in source:
            source = source.replace(
                'GradientBoostingRegressor(**p)',
                'XGBRegressor(tree_method="hist", device="cuda", **p)'
            )
        
        cell.source = source
        
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("done")

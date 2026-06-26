import json

with open(r"D:\Parth\Poly\polymer_competition\notebooks\kaggle_pipeline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    st = src[:150].replace("\n", "\\n")
    print(f"Cell {i} ({cell['cell_type']}): {st}")
    if "run_all_folds" in src or "build_ensemble" in src or "merge_submissions" in src:
        print("  *** FULL SOURCE ***")
        print(src)
        print("  *** END ***")

"""Genera lab9_forecasting.ipynb a partir de lab9_src.py (celdas separadas por '# %%')."""
import nbformat

src = open("lab9_src.py", encoding="utf-8").read()
cells = []
for chunk in src.split("\n# %%")[1:] if not src.startswith("# %%") else ("\n" + src).split("\n# %%")[1:]:
    is_md = chunk.startswith(" [markdown]")
    body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
    body = body.strip("\n")
    if is_md:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, "lab9_forecasting.ipynb")
print(f"{len(cells)} celdas")

"""`# %%` hücreli .py dosyasını Jupyter notebook'a çevirir.

Kullanım: python py_to_ipynb.py at_yarisi_siralama.py at_yarisi_siralama.ipynb
"""
import sys

import nbformat


def convert(src, dst):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    blocks, cur = [], None
    for line in open(src, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            cur = {"md": "[markdown]" in line, "lines": []}
            blocks.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    for b in blocks:
        text = "\n".join(b["lines"]).strip("\n")
        if b["md"]:
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        elif text:
            nb.cells.append(nbformat.v4.new_code_cell(text))
    nbformat.write(nb, dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])

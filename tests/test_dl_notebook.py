from pathlib import Path


def test_dl_notebook_compiles():
    path = Path("notebooks/03_deep_learning_model_review.py")
    assert path.exists()
    compile(path.read_text(), str(path), "exec")

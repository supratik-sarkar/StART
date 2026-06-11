# Apply StART v0.5 DL Suite Patch

From Mac terminal:

```bash
cd ~/Downloads
unzip StART-v0.5-dl-suite.zip
rsync -av StART-v0.5-dl-suite/ ~/Desktop/StART/
cd ~/Desktop/StART
source .venv-start/bin/activate
python -m pip install torch matplotlib tabulate
python -m pytest
ruff check src tests
python examples/deep_learning_sequence_demo.py
python notebooks/03_deep_learning_model_review.py
```

If local tests pass, sync to git clone and PR:

```bash
rsync -av --delete --exclude '.git' --exclude '.venv-start' --exclude 'start_output' ~/Desktop/StART/ ~/Desktop/My_Git/StART/
cd ~/Desktop/My_Git/StART
git checkout main
git pull origin main
git checkout -b feature/deep-learning-model-review-suite
git add -A
git commit -m "Add deep learning model review suite"
git push -u origin feature/deep-learning-model-review-suite
gh pr create --base main --head feature/deep-learning-model-review-suite --title "Add deep learning model review suite" --body "Adds a laptop-safe PyTorch MLP model review workflow with train/test/OOS metrics, explainability, robustness, figures, evidence outputs, markdown report, tests, and a Databricks-compatible notebook."
```

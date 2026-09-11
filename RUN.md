# Run Guide

1. `cd dynamic-pricing-india`
2. `python -m venv .venv`
3. Windows: `.venv\\Scripts\\activate`
4. `pip install -r requirements.txt`
5. `python src/train_model.py`
6. `streamlit run app.py`

The first training run downloads the public dataset via kagglehub. The raw dataset is not committed to GitHub.

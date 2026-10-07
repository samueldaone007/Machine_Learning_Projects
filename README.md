# Machine Learning Projects

A collection of machine learning projects, each in its own folder with the dataset, the training notebook, and the exported model.

## Projects

| Project | Problem type | Model | Folder |
|---|---|---|---|
| Student Performance Predictor | Regression | `RandomForestRegressor` | [`StudentPerformance/`](StudentPerformance/) |

---

## Student Performance Predictor

Predicts a student's **math score** from demographic and academic background features using the classic [Students Performance dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (1,000 rows, 8 columns).

### Dataset

`StudentPerformance/StudentsPerformance.csv` — 1,000 students with the following columns:

| Column | Type | Description |
|---|---|---|
| `gender` | categorical | male / female |
| `race/ethnicity` | categorical | group A–E |
| `parental level of education` | categorical | highest parental education level |
| `lunch` | categorical | standard / free-reduced |
| `test preparation course` | categorical | completed / none |
| `math score` | numeric (0–100) | **target** |
| `reading score` | numeric (0–100) | feature |
| `writing score` | numeric (0–100) | feature |

### Workflow (in the notebook)

1. **Load** the CSV with pandas.
2. **Encode** the five categorical columns with `sklearn.preprocessing.LabelEncoder`.
3. **Split** features (`X`) and target (`y = math score`).
4. **Train** a `RandomForestRegressor` on a 90/10 train/test split (`random_state=42`).
5. **Evaluate** with Mean Absolute Error and R².
6. **Round** predictions to whole numbers to match the score format.
7. **Export** the fitted model with `joblib`.

### Results

| Metric | Value |
|---|---|
| Train R² | 0.976 |
| Test R² | 87.06% |
| Mean Absolute Error (test) | 4.73 points |

### Files

```
StudentPerformance/
├── StudentsPerformance.csv          # dataset
├── Untitled2 (1).ipynb              # training notebook (Google Colab)
└── Student_Performance_Predictor.pkl # exported, fitted model
```

---

## Getting started

### Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- joblib
- Jupyter (or Google Colab)

Install the dependencies:

```bash
pip install pandas scikit-learn numpy joblib jupyter
```

### Run the notebook

```bash
jupyter notebook "StudentPerformance/Untitled2 (1).ipynb"
```

Or open it directly in [Google Colab](https://colab.research.google.com/) (it was originally developed there; upload the CSV alongside it).

Run all cells top to bottom — the notebook is self-contained and ends by writing `Student_Performance_Predictor.pkl`.

### Use the saved model

```python
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = joblib.load("StudentPerformance/Student_Performance_Predictor.pkl")

# Reproduce the notebook's encoding: fit a LabelEncoder per categorical
# column on the full dataset, then transform the new record the same way.
data = pd.read_csv("StudentPerformance/StudentsPerformance.csv")
categorical = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
encoders = {col: LabelEncoder().fit(data[col]) for col in categorical}

sample = pd.DataFrame([{
    "gender": "female",
    "race/ethnicity": "group B",
    "parental level of education": "bachelor's degree",
    "lunch": "standard",
    "test preparation course": "none",
    "reading score": 72,
    "writing score": 74,
}])
for col in categorical:
    sample[col] = encoders[col].transform(sample[col])

predicted_math_score = int(round(model.predict(sample)[0]))
print(predicted_math_score)
```

> **Note:** the notebook only persists the model, not the fitted `LabelEncoder`s, so they must be rebuilt from the dataset (as above) before making predictions. A cleaner follow-up is to save the encoders alongside the model, or wrap preprocessing and the estimator in a single scikit-learn `Pipeline`.

# Student Academic Performance Analysis System

A beginner-friendly Streamlit + SQLite + Scikit-learn project for exploring student academic records..

## Features

- Dashboard with student counts, attendance, marks and pass/fail summaries
- Search students by Student ID or name
- Add student profiles and update existing profiles
- Add or update subject-wise semester records
- Subject-wise and semester-wise academic analysis
- K-Means clustering for exploratory student grouping
- Logistic Regression to estimate pass probability
- SQLite database for persistent records

## Project structure

```text
Student_Academic_Performance/
├── dataset/
│   └── students.csv
├── database/
│   └── student_academy.db       # created automatically; ignored by Git
├── models/
│   ├── logistic_regression.joblib  # created after training
│   └── kmeans.joblib               # created after training
├── app.py
├── analytics.py
├── database.py
├── ml_models.py
├── data_analysis.py
├── requirements.txt
└── .gitignore
```

## Windows / VS Code setup

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   python -m venv venv
   ```

5. Activate it:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this once in that terminal:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   ```

6. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Open the local address shown in the terminal, usually `http://localhost:8501`.

The first launch imports `dataset/students.csv` into `database/student_academy.db`. The database is created automatically. CSV import runs only when the database has no students, so later app launches do not overwrite edits.

## Dataset columns

The supplied CSV is expected to contain:

- `Student_ID`
- `Student_Name`
- `Department`
- `Batch`
- `Semester`
- `Attendance_Percentage`
- `Study_Hours`
- `Subject`
- `Marks`
- `Maximum_Marks`

Keep the column names unchanged, or update the importer in `database.py`.

## Pass/fail rule

The initial example uses 40% as the minimum mark in each subject. A student passes a semester only if they reach that threshold in every subject recorded for that semester. This is only a sample rule. Change `PASS_MARK` in `app.py` and use the academy's official rules before reporting real results.

## Machine learning notes

### K-Means

Groups student-semester records using attendance, study hours and average marks. Cluster numbers are arbitrary labels, not grades or diagnoses. Inspect cluster averages before interpreting them.

### Logistic Regression

Estimates pass probability from attendance, study hours, previous-semester average and semester number. The current semester's final marks are deliberately excluded from the input features to avoid target leakage. The target is derived from the sample pass rule.

The included dataset is for a project prototype. Model metrics on a synthetic or small dataset do not prove real-world predictive performance. Use approved historical academy data, validate the target rule, and evaluate false negatives carefully before using predictions to guide student support.

## Important data and privacy notes

- Do not upload identifiable real student data to a public GitHub repository.
- The local SQLite database and trained models are ignored by `.gitignore`.
- The starter CSV is included so the project can run immediately. If it contains real student information, remove it or replace it with anonymized/sample data before publishing.
- This project is an educational prototype, not a system for making high-stakes academic decisions automatically.

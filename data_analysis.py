import pandas as pd
from analytics import prepare_records, semester_summary

df = pd.read_csv("dataset/students.csv")
print("Original shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

clean = prepare_records(df)
print("\nCleaned shape:", clean.shape)
print("\nUnique students:", clean["Student_ID"].nunique())
print("\nDepartments:\n", clean["Department"].value_counts())
print("\nSemester summary:\n", semester_summary(clean).head(20).to_string(index=False))

import numpy as np
import pandas as pd


def prepare_records(df):
    """Clean and enrich subject-level academic records."""
    data = df.copy()
    numeric = [
        "Semester", "Attendance_Percentage", "Study_Hours",
        "Marks", "Maximum_Marks"
    ]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[
        "Student_ID", "Semester", "Attendance_Percentage",
        "Study_Hours", "Marks", "Maximum_Marks", "Subject"
    ])
    data = data[
        data["Maximum_Marks"].gt(0) &
        data["Marks"].ge(0) &
        data["Marks"].le(data["Maximum_Marks"]) &
        data["Attendance_Percentage"].between(0, 100) &
        data["Study_Hours"].ge(0)
    ].copy()

    data["Mark_Percentage"] = (
        data["Marks"] / data["Maximum_Marks"] * 100
    )
    return data


def semester_summary(df, pass_mark=40.0):
    """One row per student per semester; Pass requires passing every subject."""
    data = prepare_records(df)
    if data.empty:
        return pd.DataFrame()

    data["Subject_Pass"] = data["Mark_Percentage"] >= pass_mark
    summary = (
        data.groupby(["Student_ID", "Semester"], as_index=False)
        .agg(
            Attendance_Percentage=("Attendance_Percentage", "mean"),
            Study_Hours=("Study_Hours", "mean"),
            Average_Marks=("Mark_Percentage", "mean"),
            Passed_Subjects=("Subject_Pass", "sum"),
            Total_Subjects=("Subject", "nunique"),
            Student_Name=("Student_Name", "first"),
            Department=("Department", "first"),
            Batch=("Batch", "first")
        )
    )
    summary["Result"] = np.where(
        summary["Passed_Subjects"] == summary["Total_Subjects"],
        "Pass", "Fail"
    )
    summary["Pass_Label"] = (summary["Result"] == "Pass").astype(int)
    summary = summary.sort_values(["Student_ID", "Semester"])
    summary["Previous_Semester_Average"] = (
        summary.groupby("Student_ID")["Average_Marks"].shift(1)
    )
    return summary


def overall_student_summary(df, pass_mark=40.0):
    data = prepare_records(df)
    if data.empty:
        return pd.DataFrame()
    sem = semester_summary(data, pass_mark)
    return (
        sem.groupby("Student_ID", as_index=False)
        .agg(
            Student_Name=("Student_Name", "first"),
            Department=("Department", "first"),
            Batch=("Batch", "first"),
            Average_Attendance=("Attendance_Percentage", "mean"),
            Average_Marks=("Average_Marks", "mean"),
            Semesters=("Semester", "nunique"),
            Passed_Semesters=("Pass_Label", "sum")
        )
    )

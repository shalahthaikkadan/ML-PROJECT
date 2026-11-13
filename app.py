import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from database import (
    initialize_database, import_csv_if_database_empty,
    get_students, get_student, add_student, update_student,
    save_academic_record, get_academic_records, get_all_records
)
from analytics import prepare_records, semester_summary, overall_student_summary
from ml_models import (
    train_logistic, train_kmeans, load_logistic, load_kmeans,
    predict_pass_probability
)

st.set_page_config(
    page_title="Student Academic Performance",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background: #f7f9fc; }
.block-container { padding-top: 1.7rem; }
[data-testid="stMetric"] {
    background: white; border: 1px solid #e5eaf2;
    padding: 16px; border-radius: 13px;
}
.hero {
    padding: 22px 24px; border-radius: 16px;
    background: linear-gradient(115deg, #102a56, #2563eb);
    color: white; margin-bottom: 18px;
}
.hero h1 { color: white; margin: 0; }
.hero p { color: #dbeafe; margin: 6px 0 0 0; }
</style>
""", unsafe_allow_html=True)

PASS_MARK = 40.0  # Change this to the academy's official subject pass percentage.

initialize_database()
import_csv_if_database_empty()


@st.cache_data(ttl=5)
def load_records():
    return get_all_records()


def refresh_data():
    load_records.clear()
    st.rerun()


raw_df = load_records()
if raw_df.empty:
    st.error("No academic records found. Check dataset/students.csv.")
    st.stop()

df = prepare_records(raw_df)
sem_df = semester_summary(df, PASS_MARK)
students_df = get_students()


def page_header(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def show_student_profile(student_id):
    profile = get_student(student_id)
    records = prepare_records(get_academic_records(student_id))
    if profile is None or records.empty:
        st.info("No student profile or academic records found.")
        return

    st.subheader(f"👤 {profile['student_name']}")
    st.caption(
        f"{profile['student_id']}  ·  {profile['department']}  ·  Batch {profile['batch']}"
    )

    attendance = records.groupby("Semester")["Attendance_Percentage"].mean().mean()
    average = records["Marks"].sum() / records["Maximum_Marks"].sum() * 100
    a, b, c = st.columns(3)
    a.metric("Average attendance", f"{attendance:.1f}%")
    b.metric("Overall marks", f"{average:.1f}%")
    c.metric("Semesters", int(records["Semester"].nunique()))

    st.markdown("#### Semester results")
    student_sem = sem_df[sem_df["Student_ID"] == student_id].sort_values("Semester")
    if not student_sem.empty:
        st.dataframe(
            student_sem[["Semester", "Average_Marks", "Passed_Subjects",
                         "Total_Subjects", "Result"]],
            use_container_width=True, hide_index=True
        )

    st.markdown("#### Subject-wise records")
    st.dataframe(
        records[["Semester", "Subject", "Marks", "Maximum_Marks",
                 "Mark_Percentage", "Attendance_Percentage", "Study_Hours"]],
        use_container_width=True, hide_index=True
    )

    trend = records.groupby("Semester", as_index=False)["Mark_Percentage"].mean()
    st.markdown("#### Semester performance trend")
    st.line_chart(trend, x="Semester", y="Mark_Percentage")


st.sidebar.title("🎓 Academy Portal")
st.sidebar.caption("Student Academic Performance System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Student Management", "Student Search",
     "Academic Performance", "ML Analysis"]
)
st.sidebar.divider()
st.sidebar.caption("Prototype · Local SQLite database")

if page == "Dashboard":
    page_header("Student Performance Dashboard",
                "Overview of student outcomes, attendance and academic progress.")

    unique_students = df["Student_ID"].nunique()
    departments = df["Department"].nunique()
    avg_attendance = sem_df["Attendance_Percentage"].mean()
    avg_marks = df["Mark_Percentage"].mean()
    passed = int((sem_df["Result"] == "Pass").sum())
    failed = int((sem_df["Result"] == "Fail").sum())
    total_semester_results = len(sem_df)
    pass_rate = passed / total_semester_results * 100 if total_semester_results else 0
    fail_rate = failed / total_semester_results * 100 if total_semester_results else 0

    f1, f2 = st.columns(2)
    with f1:
        dept_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department filter", dept_options)
    with f2:
        sem_options = ["All"] + sorted(df["Semester"].dropna().unique().tolist())
        selected_sem = st.selectbox("Semester filter", sem_options)

    filtered = df.copy()
    if selected_dept != "All":
        filtered = filtered[filtered["Department"] == selected_dept]
    if selected_sem != "All":
        filtered = filtered[filtered["Semester"] == selected_sem]

    filtered_sem = semester_summary(filtered, PASS_MARK)
    total_sem = len(filtered_sem)
    p_count = int((filtered_sem["Result"] == "Pass").sum()) if total_sem else 0
    f_count = total_sem - p_count
    p_rate = p_count / total_sem * 100 if total_sem else 0
    f_rate = f_count / total_sem * 100 if total_sem else 0
    attendance_value = filtered_sem["Attendance_Percentage"].mean() if total_sem else 0
    marks_value = filtered["Mark_Percentage"].mean() if not filtered.empty else 0

    st.markdown("### Performance summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unique students", filtered["Student_ID"].nunique())
    k2.metric("Departments", filtered["Department"].nunique())
    k3.metric("Average attendance", f"{attendance_value:.1f}%")
    k4.metric("Average marks", f"{marks_value:.1f}%")
    k5, k6 = st.columns(2)
    k5.metric("Pass rate", f"{p_rate:.1f}%")
    k6.metric("Fail rate", f"{f_rate:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("### Department-wise students")
        dept_chart = (filtered.groupby("Department")["Student_ID"]
                      .nunique().reset_index(name="Students"))
        if not dept_chart.empty:
            st.bar_chart(dept_chart, x="Department", y="Students")
    with right:
        st.markdown("### Semester-wise average marks")
        trend = (filtered.groupby("Semester")["Mark_Percentage"]
                 .mean().reset_index(name="Average marks"))
        if not trend.empty:
            st.line_chart(trend, x="Semester", y="Average marks")

    st.markdown("### Pass / fail student-semester results")
    st.bar_chart(pd.DataFrame({
        "Result": ["Pass", "Fail"],
        "Count": [p_count, f_count]
    }), x="Result", y="Count")

    st.caption(
        f"Pass rule used in this prototype: every subject must reach "
        f"{PASS_MARK:.0f}% to pass a semester. Confirm this with the academy."
    )

elif page == "Student Management":
    page_header("Student Management",
                "Register students, update profiles and maintain semester records.")
    tab_add, tab_update, tab_record, tab_view = st.tabs(
        ["➕ Add student", "✏️ Update profile",
         "📚 Add / update academic record", "👥 View students"]
    )

    with tab_add:
        with st.form("add_student_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            sid = c1.text_input("Student ID")
            name = c2.text_input("Student name")
            department = c1.text_input("Department", placeholder="e.g. Computer Science")
            batch = c2.text_input("Batch", placeholder="e.g. 2023-2027")
            submitted = st.form_submit_button("Add student", use_container_width=True)
        if submitted:
            if not all([sid.strip(), name.strip(), department.strip(), batch.strip()]):
                st.error("Please complete all fields.")
            else:
                ok, message = add_student(sid, name, department, batch)
                (st.success if ok else st.error)(message)
                if ok:
                    refresh_data()

    with tab_update:
        if students_df.empty:
            st.info("No students are registered.")
        else:
            chosen = st.selectbox(
                "Choose student",
                students_df["student_id"].tolist(),
                format_func=lambda x: (
                    f"{x} — " +
                    students_df.loc[students_df.student_id == x, "student_name"].iloc[0]
                )
            )
            profile = get_student(chosen)
            with st.form("update_student_form"):
                name_new = st.text_input("Student name", value=profile["student_name"])
                dept_new = st.text_input("Department", value=profile["department"])
                batch_new = st.text_input("Batch", value=profile["batch"])
                submitted = st.form_submit_button("Save profile changes",
                                                   use_container_width=True)
            if submitted:
                if not all([name_new.strip(), dept_new.strip(), batch_new.strip()]):
                    st.error("All fields are required.")
                elif update_student(chosen, name_new, dept_new, batch_new):
                    st.success("Student profile updated.")
                    refresh_data()
                else:
                    st.error("No profile was updated.")

    with tab_record:
        if students_df.empty:
            st.info("Add a student before adding academic records.")
        else:
            chosen = st.selectbox("Student", students_df["student_id"].tolist(),
                                  key="record_student")
            with st.form("academic_record_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                semester = c1.selectbox("Semester", list(range(1, 9)))
                subject = c2.text_input("Subject")
                attendance = c1.number_input("Attendance (%)", 0.0, 100.0, 75.0)
                hours = c2.number_input("Study hours per day", 0.0, 24.0, 3.0)
                maximum = c1.number_input("Maximum marks", 1.0, 1000.0, 100.0)
                marks = c2.number_input("Obtained marks", 0.0, float(maximum), 50.0)
                submitted = st.form_submit_button("Save academic record",
                                                  use_container_width=True)
            if submitted:
                ok, message = save_academic_record(
                    chosen, semester, subject, attendance, hours, marks, maximum
                )
                (st.success if ok else st.error)(message)
                if ok:
                    refresh_data()
            st.caption(
                "If the same student, semester and subject already exist, "
                "the record is updated rather than duplicated."
            )

    with tab_view:
        current_students = get_students()
        st.metric("Registered students", len(current_students))
        st.dataframe(current_students, use_container_width=True, hide_index=True)

elif page == "Student Search":
    page_header("Find a Student",
                "Search using Student ID or student name.")
    query = st.text_input("Search student", placeholder="Type a name or Student ID...")
    if query.strip():
        matches = get_students(query)
        if matches.empty:
            st.warning("No matching student found.")
        else:
            chosen = st.selectbox(
                "Select a matching student",
                matches["student_id"].tolist(),
                format_func=lambda x: (
                    f"{x} — " +
                    matches.loc[matches.student_id == x, "student_name"].iloc[0]
                )
            )
            show_student_profile(chosen)
    else:
        st.info("Start typing to search for a student.")

elif page == "Academic Performance":
    page_header("Academic Performance",
                "Explore semester-wise and subject-wise student results.")
    c1, c2 = st.columns(2)
    dept_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
    dept = c1.selectbox("Department", dept_options, key="academic_dept")
    sem_options = ["All"] + sorted(df["Semester"].dropna().unique().tolist())
    sem = c2.selectbox("Semester", sem_options, key="academic_sem")
    view = df.copy()
    if dept != "All":
        view = view[view["Department"] == dept]
    if sem != "All":
        view = view[view["Semester"] == sem]

    a, b = st.columns(2)
    with a:
        st.markdown("### Subject-wise average marks")
        subject_chart = (view.groupby("Subject")["Mark_Percentage"]
                         .mean().sort_values(ascending=False)
                         .reset_index(name="Average marks"))
        if not subject_chart.empty:
            st.bar_chart(subject_chart, x="Subject", y="Average marks")
    with b:
        st.markdown("### Semester results")
        result_view = semester_summary(view, PASS_MARK)
        if not result_view.empty:
            counts = result_view["Result"].value_counts().rename_axis("Result").reset_index(name="Count")
            st.bar_chart(counts, x="Result", y="Count")

    st.markdown("### Student-semester summary")
    summary = semester_summary(view, PASS_MARK)
    if not summary.empty:
        st.dataframe(
            summary[["Student_ID", "Student_Name", "Department", "Semester",
                     "Attendance_Percentage", "Study_Hours", "Average_Marks",
                     "Passed_Subjects", "Total_Subjects", "Result"]],
            use_container_width=True, hide_index=True
        )

elif page == "ML Analysis":
    page_header("Machine Learning Analysis",
                "Train K-Means groups and Logistic Regression pass-probability model.")
    st.warning(
        "This is an academic prototype. Predictions are estimates, not official "
        "results or decisions about students. Confirm the pass rule and validate "
        "models with academy-approved historical data."
    )
    st.markdown("### Model training")
    pass_mark = st.number_input("Minimum subject pass percentage",
                                min_value=1.0, max_value=100.0,
                                value=float(PASS_MARK), step=1.0)
    clusters = st.slider("K-Means number of clusters", 2, 6, 3)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train Logistic Regression", use_container_width=True):
            try:
                metrics, _ = train_logistic(raw_df, pass_mark)
                st.session_state["logistic_metrics"] = metrics
                st.success("Logistic Regression model trained and saved.")
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Train K-Means", use_container_width=True):
            try:
                clusters_df, _ = train_kmeans(raw_df, clusters, pass_mark)
                st.session_state["clusters_df"] = clusters_df
                st.success("K-Means model trained and saved.")
            except Exception as exc:
                st.error(str(exc))

    metrics = st.session_state.get("logistic_metrics")
    if metrics:
        st.markdown("### Logistic Regression evaluation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall", f"{metrics['recall']:.1%}")
        m4.metric("F1 score", f"{metrics['f1']:.1%}")
        st.write("Confusion matrix (rows = actual, columns = predicted):")
        st.dataframe(pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual Fail", "Actual Pass"],
            columns=["Predicted Fail", "Predicted Pass"]
        ))

    cluster_result = st.session_state.get("clusters_df")
    if cluster_result is not None:
        st.markdown("### K-Means cluster distribution")
        distribution = cluster_result["Cluster"].value_counts().sort_index().reset_index()
        distribution.columns = ["Cluster", "Student-semester records"]
        st.bar_chart(distribution, x="Cluster", y="Student-semester records")
        st.dataframe(cluster_result, use_container_width=True, hide_index=True)

    st.markdown("### Estimate a student's pass probability")
    logistic_artifact = load_logistic()
    if logistic_artifact is None:
        st.info("Train Logistic Regression first to enable probability estimates.")
    else:
        student_options = get_students()
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            student_id = c1.selectbox("Student", student_options["student_id"].tolist())
            semester = c2.selectbox("Semester to estimate", list(range(1, 9)))
            attendance = c1.number_input("Expected attendance (%)", 0.0, 100.0, 75.0)
            study_hours = c2.number_input("Study hours per day", 0.0, 24.0, 3.0)
            previous_avg = c1.number_input(
                "Previous semester average (%) — use 0 if unavailable",
                0.0, 100.0, 0.0
            )
            predict = st.form_submit_button("Estimate pass probability",
                                            use_container_width=True)
        if predict:
            previous_value = None if previous_avg == 0 else previous_avg
            probability = predict_pass_probability(
                logistic_artifact, attendance, study_hours,
                previous_value, semester
            )
            st.metric("Estimated pass probability", f"{probability:.1%}")
            st.caption(
                "Model estimate only. It is not a guarantee or an official result. "
                "A previous average of 0 is treated as unavailable and replaced "
                "with the model's training median."
            )

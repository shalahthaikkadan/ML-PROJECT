from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "database"
DB_PATH = DB_DIR / "student_academy.db"
CSV_PATH = ROOT / "dataset" / "students.csv"


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            department TEXT NOT NULL,
            batch TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS academic_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            semester INTEGER NOT NULL,
            subject TEXT NOT NULL,
            attendance_percentage REAL NOT NULL,
            study_hours REAL NOT NULL,
            marks REAL NOT NULL,
            maximum_marks REAL NOT NULL,
            UNIQUE(student_id, semester, subject),
            FOREIGN KEY(student_id) REFERENCES students(student_id)
                ON UPDATE CASCADE ON DELETE CASCADE
        );
        """)


def import_csv_if_database_empty():
    """Import the starter CSV once. Existing database records are preserved."""
    initialize_database()
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        if count:
            return

        if not CSV_PATH.exists():
            return

        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
        required = [
            "Student_ID", "Student_Name", "Department", "Batch",
            "Semester", "Attendance_Percentage", "Study_Hours",
            "Subject", "Marks", "Maximum_Marks"
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

        df = df[required].copy()
        df["Student_ID"] = df["Student_ID"].astype(str).str.strip()
        df["Student_Name"] = df["Student_Name"].astype(str).str.strip()
        df["Department"] = df["Department"].astype(str).str.strip()
        df["Batch"] = df["Batch"].astype(str).str.strip()
        for col in [
            "Semester", "Attendance_Percentage", "Study_Hours",
            "Marks", "Maximum_Marks"
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required)
        df = df[
            (df["Semester"] >= 1) &
            (df["Attendance_Percentage"].between(0, 100)) &
            (df["Study_Hours"] >= 0) &
            (df["Maximum_Marks"] > 0) &
            (df["Marks"].between(0, df["Maximum_Marks"]))
        ]

        profiles = df[
            ["Student_ID", "Student_Name", "Department", "Batch"]
        ].drop_duplicates("Student_ID")

        conn.executemany(
            """INSERT OR IGNORE INTO students
               (student_id, student_name, department, batch)
               VALUES (?, ?, ?, ?)""",
            profiles.itertuples(index=False, name=None)
        )

        records = df[
            ["Student_ID", "Semester", "Subject", "Attendance_Percentage",
             "Study_Hours", "Marks", "Maximum_Marks"]
        ]
        conn.executemany(
            """INSERT OR IGNORE INTO academic_records
               (student_id, semester, subject, attendance_percentage,
                study_hours, marks, maximum_marks)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records.itertuples(index=False, name=None)
        )


def get_students(search=""):
    with get_connection() as conn:
        if search.strip():
            term = f"%{search.strip()}%"
            rows = conn.execute(
                """SELECT * FROM students
                   WHERE student_id LIKE ? OR student_name LIKE ?
                   ORDER BY student_name""",
                (term, term)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM students ORDER BY student_name"
            ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_student(student_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?",
            (student_id,)
        ).fetchone()
    return dict(row) if row else None


def add_student(student_id, student_name, department, batch):
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO students
                   (student_id, student_name, department, batch)
                   VALUES (?, ?, ?, ?)""",
                (student_id.strip(), student_name.strip(),
                 department.strip(), str(batch).strip())
            )
        return True, "Student added successfully."
    except sqlite3.IntegrityError:
        return False, "That Student ID already exists."


def update_student(student_id, student_name, department, batch):
    with get_connection() as conn:
        cur = conn.execute(
            """UPDATE students
               SET student_name = ?, department = ?, batch = ?
               WHERE student_id = ?""",
            (student_name.strip(), department.strip(),
             str(batch).strip(), student_id)
        )
    return cur.rowcount > 0


def save_academic_record(student_id, semester, subject, attendance,
                         study_hours, marks, maximum_marks):
    if not get_student(student_id):
        return False, "Student ID was not found."
    if not subject.strip():
        return False, "Subject is required."
    if not 0 <= attendance <= 100:
        return False, "Attendance must be between 0 and 100."
    if study_hours < 0:
        return False, "Study hours cannot be negative."
    if maximum_marks <= 0 or not 0 <= marks <= maximum_marks:
        return False, "Marks must be between 0 and maximum marks."

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO academic_records
               (student_id, semester, subject, attendance_percentage,
                study_hours, marks, maximum_marks)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(student_id, semester, subject)
               DO UPDATE SET
                 attendance_percentage = excluded.attendance_percentage,
                 study_hours = excluded.study_hours,
                 marks = excluded.marks,
                 maximum_marks = excluded.maximum_marks""",
            (student_id, int(semester), subject.strip(), float(attendance),
             float(study_hours), float(marks), float(maximum_marks))
        )
    return True, "Academic record saved. Existing subject record is updated."


def get_academic_records(student_id=None):
    query = """
        SELECT
            a.record_id AS Record_ID,
            a.student_id AS Student_ID,
            s.student_name AS Student_Name,
            s.department AS Department,
            s.batch AS Batch,
            a.semester AS Semester,
            a.subject AS Subject,
            a.attendance_percentage AS Attendance_Percentage,
            a.study_hours AS Study_Hours,
            a.marks AS Marks,
            a.maximum_marks AS Maximum_Marks
        FROM academic_records a
        JOIN students s ON s.student_id = a.student_id
    """
    params = ()
    if student_id is not None:
        query += " WHERE a.student_id = ?"
        params = (student_id,)
    query += " ORDER BY a.student_id, a.semester, a.subject"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_all_records():
    return get_academic_records()

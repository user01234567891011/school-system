
# Запуск: streamlit run app.py
# Требования:
#   pip install streamlit pandas plotly numpy scikit-learn google-genai
# Секрет (.streamlit/secrets.toml):
#   GEMINI_API_KEY = "ваш_ключ"

import os
import sqlite3 as sq3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from google import genai

# Конфигурация страницы
st.set_page_config(
    page_title="Школьная ИС",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="auto",
)

# 1. Адаптивный CSS
#    - ≤ 768px: колонки в столбик, уменьшенные отступы, компактные таблицы
#    - ≥ 769px: обычная раскладка
st.markdown(
    """
    <style>
    /* ---------- База ---------- */
    .stApp { overflow-wrap: break-word; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* Таблицы: аккуратные границы и читаемый шрифт */
    div[data-testid="stDataFrame"] { font-size: 0.92rem; }

    /* Метрики: чуть крупнее значение */
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }

    /* ---------- Мобильный режим ---------- */
    @media (max-width: 768px) {
        /* Все горизонтальные блоки — в столбик */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.4rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        /* Меньше отступов по краям */
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 3rem !important;
        }

        /* Заголовки компактнее */
        h1 { font-size: 1.45rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        /* Таблицы: мельче шрифт, чтобы влезало больше колонок */
        div[data-testid="stDataFrame"] { font-size: 1.2rem; }

        /* Кнопки на всю ширину — удобнее нажимать */
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100% !important;
        }

        /* Вкладки: горизонтальный скролл, не переносим */
        div[data-testid="stTabs"] div[role="tablist"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            scrollbar-width: none;
        }
        div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {
            display: none;
        }

        /* Чат: поле ввода на всю ширину */
        div[data-testid="stChatInput"] { padding: 0 0.4rem; }

        /* Диалог: почти на весь экран */
        div[data-testid="stDialog"] > div {
            width: 92vw !important;
            max-width: 92vw !important;
        }
    }

    /* ---------- Очень узкие экраны ---------- */
    @media (max-width: 420px) {
        .block-container { padding-left: 0.4rem !important; padding-right: 0.4rem !important; }
        h1 { font-size: 1.25rem !important; }
        div[data-testid="stDataFrame"] { font-size: 0.76rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "school.db"

# 2. API-ключ
def get_api_key() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

# 3. БД + миграции
def get_conn() -> sq3.Connection:
    con = sq3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA foreign_keys = ON")
    return con

def _ensure_column(cur: sq3.Cursor, table: str, column: str, coltype: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if not cols:
        return
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

def _run_migrations(cur: sq3.Cursor) -> None:
    for t, cols in {
        "teachers": [("name", "TEXT"), ("phone", "INTEGER"), ("date", "DATE"), ("passwords", "TEXT")],
        "students": [("name", "TEXT"), ("class_id", "INTEGER"), ("phone", "INTEGER"), ("language", "TEXT"),
                     ("date", "DATE"), ("grades", "DECIMAL"), ("attendance", "DECIMAL"), ("passwords", "TEXT")],
        "admins": [("passwords", "TEXT")],
        "classes": [("class", "TEXT"), ("students_number", "INTEGER"), ("teacher_id", "INTEGER")],
        "subjects": [("subject", "TEXT"), ("teacher_id", "INTEGER"), ("class_id", "INTEGER")],
        "grades": [("student_id", "INTEGER"), ("name", "TEXT"), ("subject_id", "INTEGER"),
                   ("subject", "TEXT"), ("grade", "DECIMAL"), ("cw_grade", "DECIMAL")],
        "grading": [("student_id", "INTEGER"), ("name", "TEXT"), ("grade", "INTEGER"),
                    ("subject", "TEXT"), ("subject_id", "INTEGER"), ("date", "DATE")],
        "cw_grading": [("student_id", "INTEGER"), ("name", "TEXT"), ("cw_grade", "INTEGER"),
                       ("subject", "TEXT"), ("subject_id", "INTEGER"), ("date", "DATE")],
        "attendance": [("student_id", "INTEGER"), ("name", "TEXT"), ("was", "BOOL"), ("date", "DATE")],
        "rasp": [("subject_id", "INTEGER"), ("subject", "TEXT"), ("class_id", "INTEGER"),
                 ("class", "TEXT"), ("date", "DATE")],
    }.items():
        for col, typ in cols:
            _ensure_column(cur, t, col, typ)

def _backfill_passwords(cur: sq3.Cursor) -> None:
    cur.execute("UPDATE teachers SET passwords='psword-t' || teacher_id WHERE passwords IS NULL OR passwords=''")
    cur.execute("UPDATE students SET passwords='psword-p' || student_id WHERE passwords IS NULL OR passwords=''")

def init_db() -> None:
    con = get_conn()
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS admins(admin_id INTEGER PRIMARY KEY AUTOINCREMENT, passwords TEXT)")
    cur.execute("""CREATE TABLE IF NOT EXISTS students(student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT, class_id INTEGER, phone INTEGER, language TEXT, date DATE,
                   grades DECIMAL, attendance DECIMAL, passwords TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS subjects(subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   subject TEXT, teacher_id INTEGER, class_id INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS classes(class_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   class TEXT, students_number INTEGER, teacher_id INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS teachers(teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT, phone INTEGER, date DATE, passwords TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS grades(grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   student_id INTEGER, name TEXT, subject_id INTEGER, subject TEXT,
                   grade DECIMAL, cw_grade DECIMAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS grading(student_id INTEGER, name TEXT, grade INTEGER,
                   subject TEXT, subject_id INTEGER, date DATE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS cw_grading(student_id INTEGER, name TEXT, cw_grade INTEGER,
                   subject TEXT, subject_id INTEGER, date DATE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS attendance(student_id INTEGER, name TEXT, was BOOL, date DATE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS rasp(subject_id INTEGER, subject TEXT, class_id INTEGER,
                   class TEXT, date DATE)""")

    _run_migrations(cur)

    cur.executemany(
        "INSERT OR IGNORE INTO teachers(teacher_id, name, phone, date, passwords) VALUES(?,?,?,?,?)",
        [
            (1, "Владимир Владимирович Путин", 77761561, "1952-10-07", "psword-t1"),
            (2, "Иосиф Виссарионович Джугашвили", 77766262, "1990-05-05", "psword-t2"),
            (3, "Наполеон Бонапарт", 77766363, "1970-05-05", "psword-t3"),
            (4, "Пупкин Иван Васильевич", 77766464, "2000-05-05", "psword-t4"),
        ],
    )
    cur.executemany("INSERT OR IGNORE INTO admins(admin_id, passwords) VALUES(?,?)", [(1, "1234")])
    cur.executemany(
        "INSERT OR IGNORE INTO classes(class_id, class, students_number, teacher_id) VALUES(?,?,?,?)",
        [(1, "9А", 30, 1), (2, "9Б", 29, 2), (3, "9В", 32, 3), (4, "9Г", 28, 4)],
    )
    cur.executemany(
        """INSERT OR IGNORE INTO students(student_id, name, class_id, phone, language, date, grades, attendance, passwords)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        [
            (1, "Дятлов Александр Сергеевич", 4, 77955151, "Английский", "2010-05-05", 0, 0, "psword-p1"),
            (2, "Иванов Иван Иванович", 4, 77955252, "Английский", "2010-05-06", 0, 0, "psword-p2"),
            (3, "Смит Джон Олегович", 4, 77955353, "Английский", "2010-05-07", 0, 0, "psword-p3"),
            (4, "Ким Чен Ын", 4, 77955454, "Английский", "2010-05-08", 0, 0, "psword-p4"),
            (5, "Коровкина Анастасия Владимировна", 4, 77955656, "Английский", "2010-05-09", 0, 0, "psword-p5"),
            (6, "Петров Петр Петрович", 1, 77955757, "Английский", "2010-01-15", 0, 0, "psword-p6"),
            (7, "Сидорова Анна Сергеевна", 1, 77955858, "Английский", "2010-03-22", 0, 0, "psword-p7"),
            (8, "Васильев Василий Васильевич", 1, 77955959, "Английский", "2010-07-11", 0, 0, "psword-p8"),
            (9, "Смирнова Елена Дмитриевна", 1, 77956060, "Английский", "2010-09-19", 0, 0, "psword-p9"),
            (10, "Кузнецов Алексей Павлович", 1, 77956161, "Английский", "2010-12-03", 0, 0, "psword-p10"),
            (11, "Морозов Дмитрий Олегович", 3, 77956262, "Французский", "2010-04-14", 0, 0, "psword-p11"),
            (12, "Попова Ольга Николаевна", 3, 77956363, "Французский", "2010-06-18", 0, 0, "psword-p12"),
            (13, "Соколова Мария Андреевна", 3, 77956464, "Французский", "2010-08-25", 0, 0, "psword-p13"),
            (14, "Лебедев Максим Сергеевич", 3, 77956565, "Французский", "2010-02-10", 0, 0, "psword-p14"),
            (15, "Козлова Екатерина Игоревна", 3, 77956666, "Французский", "2010-10-30", 0, 0, "psword-p15"),
            (16, "Новиков Артем Павлович", 2, 77956767, "Немецкий", "2010-01-05", 0, 0, "psword-p16"),
            (17, "Федоров Никита Валерьевич", 2, 77956868, "Немецкий", "2010-03-17", 0, 0, "psword-p17"),
            (18, "Морозова Татьяна Павловна", 2, 77956969, "Немецкий", "2010-07-21", 0, 0, "psword-p18"),
            (19, "Волков Сергей Михайлович", 2, 77957070, "Немецкий", "2010-09-09", 0, 0, "psword-p19"),
            (20, "Петухов Кирилл Андреевич", 2, 77957171, "Немецкий", "2010-11-12", 0, 0, "psword-p20"),
        ],
    )
    cur.executemany(
        "INSERT OR IGNORE INTO subjects(subject_id, subject, teacher_id, class_id) VALUES(?,?,?,?)",
        [
            (1, "Математика", 4, 1), (2, "Русский язык", 2, 1), (3, "Английский язык", 2, 1),
            (4, "Физика", 1, 1), (5, "Информатика", 2, 1), (6, "Геометрия", 4, 1),
            (7, "Математика", 4, 2), (8, "Русский язык", 2, 2), (9, "Английский язык", 2, 2),
            (10, "Физика", 1, 2), (11, "Информатика", 2, 2), (12, "Геометрия", 4, 2),
            (13, "Математика", 4, 3), (14, "Русский язык", 3, 3), (15, "Французский язык", 2, 3),
            (16, "Физика", 1, 3), (17, "Информатика", 2, 3), (18, "Геометрия", 4, 3),
            (19, "Математика", 4, 4), (20, "Русский язык", 2, 4), (21, "Немецкий язык", 1, 4),
            (22, "Физика", 1, 4), (23, "Информатика", 2, 4), (24, "Геометрия", 4, 4),
        ],
    )
    _backfill_passwords(cur)
    con.commit()
    con.close()

init_db()

# 4. Сессия
DEFAULTS = {
    "student_id": 0,
    "autho": False,
    "role": None,
    "role_check": 0,
    "reg_id": None,
    "reg_password": None,
    "flash": None,
    "chat_school": [{"role": "assistant", "text": "Добрый день! Чем могу помочь по школьной базе?"}],
    "chat_tutor": [{"role": "assistant", "text": "Привет! Давай разберёмся с учёбой. С чего начнём?"}],
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def flash(msg: str, kind: str = "success") -> None:
    st.session_state.flash = (kind, msg)
# 5. Авторизация
ROLE_MAP = {"Ученик": "student", "Учитель": "teacher", "Администратор": "admin"}
ROLE_CHECK = {"student": 1, "teacher": 2, "admin": 3}
ROLE_LABEL = {1: "Ученик", 2: "Учитель", 3: "Администратор"}
def login_screen() -> None:
    st.title("🔐 Вход в систему")
    col1, col2 = st.columns([1, 1])
    with col1:
        role_label = st.selectbox("Роль", list(ROLE_MAP.keys()))
        role = ROLE_MAP[role_label]
        reg_id = st.number_input("ID", min_value=1, step=1)
        reg_password = st.text_input("Пароль", max_chars=32, type="password")
        if st.button("Войти", type="primary", use_container_width=True):
            con = get_conn()
            try:
                if role == "admin":
                    table, id_col = "admins", "admin_id"
                elif role == "teacher":
                    table, id_col = "teachers", "teacher_id"
                else:
                    table, id_col = "students", "student_id"
                df = pd.read_sql(
                    f"SELECT * FROM {table} WHERE {id_col}=? AND passwords=?",
                    con, params=(reg_id, reg_password),
                )
            finally:
                con.close()
            if df.empty:
                st.error("Неверно введён ID или пароль")
            else:
                st.session_state.autho = True
                st.session_state.role = role
                st.session_state.role_check = ROLE_CHECK[role]
                st.session_state.reg_id = int(reg_id)
                st.session_state.reg_password = reg_password
                flash(f"Вы вошли как {role_label}")
                st.rerun()
    with col2:
        st.info(
            "**Демо-доступы**\n\n"
            "- Ученик: ID `1`, пароль `psword-p1`\n"
            "- Учитель: ID `1`, пароль `psword-t1`\n"
            "- Админ: ID `1`, пароль `1234`"
        )

if not st.session_state.autho:
    login_screen()
    st.stop()
# 6. Шапка + sidebar
st.title("🏫 Школьная информационная система")
st.caption(f"Вы вошли как **{ROLE_LABEL[st.session_state.role_check]}** (ID {st.session_state.reg_id})")

with st.sidebar:
    st.markdown(f"### 👤 {ROLE_LABEL[st.session_state.role_check]}")
    st.caption(f"ID: {st.session_state.reg_id}")
    if st.button("🚪 Выйти", use_container_width=True):
        for k in ("autho", "role", "role_check", "reg_id", "reg_password"):
            st.session_state[k] = DEFAULTS[k]
        st.rerun()
    st.divider()

if st.session_state.flash:
    kind, msg = st.session_state.flash
    st.session_state.flash = None
    getattr(st, kind)(msg)


# 7.

def add_grade(student_id: int, subject_id: int, grade: int, grade_date: date, is_cw: bool) -> None:
    table = "cw_grading" if is_cw else "grading"
    col = "cw_grade" if is_cw else "grade"
    con = get_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Ученик с ID {student_id} не найден")
        student_name = row[0]

        cur.execute("SELECT subject FROM subjects WHERE subject_id=?", (subject_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Предмет с ID {subject_id} не найден")
        subject_name = row[0]

        cur.execute(
            f"INSERT INTO {table}(student_id, name, {col}, subject, subject_id, date) VALUES(?,?,?,?,?,?)",
            (student_id, student_name, grade, subject_name, subject_id, str(grade_date)),
        )

        cur.execute(
            """SELECT ROUND(AVG(m), 2) FROM (
                   SELECT grade AS m FROM grading WHERE student_id=?
                   UNION ALL
                   SELECT cw_grade AS m FROM cw_grading WHERE student_id=?)""",
            (student_id, student_id),
        )
        avg_all = cur.fetchone()[0] or 0
        cur.execute("UPDATE students SET grades=? WHERE student_id=?", (avg_all, student_id))

        cur.execute(
            """SELECT ROUND(AVG(m), 2) FROM (
                   SELECT grade AS m FROM grading WHERE student_id=? AND subject_id=?
                   UNION ALL
                   SELECT cw_grade AS m FROM cw_grading WHERE student_id=? AND subject_id=?)""",
            (student_id, subject_id, student_id, subject_id),
        )
        avg_subj = cur.fetchone()[0] or 0
        cur.execute("SELECT 1 FROM grades WHERE student_id=? AND subject_id=?", (student_id, subject_id))
        if cur.fetchone():
            cur.execute("UPDATE grades SET grade=? WHERE student_id=? AND subject_id=?",
                        (avg_subj, student_id, subject_id))
        else:
            cur.execute(
                "INSERT INTO grades(student_id, name, subject_id, subject, grade, cw_grade) VALUES(?,?,?,?,?,?)",
                (student_id, student_name, subject_id, subject_name, avg_subj, 0),
            )
        con.commit()
    finally:
        con.close()

# 8. Навигация (адаптивная)
NAV_ITEMS = [
    "📊 Таблицы",
    "✍️ Оценки",
    "🔎 Запросы",
    "📈 Аналитика",
    "🤖 ИИ-агент",
    "🎓 Репетитор",
]
if st.session_state.role_check == 3:
    NAV_ITEMS.insert(1, "👥 Справочники")

with st.sidebar:
    st.markdown("### 🧭 Навигация")
    nav = st.radio("Раздел", NAV_ITEMS, label_visibility="collapsed")
    st.divider()
    st.caption("Данные обновляются в реальном времени.")

# 9. Таблицы
def page_tables() -> None:
    st.header("📊 Просмотр таблиц")

    base = {
        "Ученики": "students",
        "Предметы": "subjects",
        "Классы": "classes",
        "Учителя": "teachers",
        "Средний балл": "grades",
        "Текущие оценки": "grading",
        "Оценки за КР": "cw_grading",
        "Посещаемость": "attendance",
    }
    if st.session_state.role_check == 1:
        allowed = ("Ученики", "Предметы", "Учителя", "Средний балл", "Текущие оценки", "Оценки за КР", "Посещаемость")
        base = {k: v for k, v in base.items() if k in allowed}

    table_label = st.selectbox("Таблица", list(base.keys()))
    table = base[table_label]

    rc = st.session_state.role_check
    con = get_conn()
    try:
        if rc == 3:
            if table == "students":
                df = pd.read_sql(
                    "SELECT student_id, name, class_id, phone, language, date, grades, attendance FROM students", con
                )
            else:
                df = pd.read_sql(f"SELECT * FROM {table}", con)
        elif rc == 2:
            tid = st.session_state.reg_id
            if table == "students":
                df = pd.read_sql(
                    """SELECT s.student_id, s.name, s.class_id, s.phone, s.language, s.date, s.grades, s.attendance
                       FROM students s
                       WHERE s.class_id IN (SELECT class_id FROM subjects WHERE teacher_id=?)""",
                    con, params=(tid,),
                )
            elif table == "teachers":
                df = pd.read_sql("SELECT teacher_id, name, phone, date FROM teachers WHERE teacher_id=?", con, params=(tid,))
            elif table == "subjects":
                df = pd.read_sql("SELECT * FROM subjects WHERE teacher_id=?", con, params=(tid,))
            elif table in ("grading", "cw_grading", "grades"):
                df = pd.read_sql(
                    f"SELECT * FROM {table} WHERE subject_id IN (SELECT subject_id FROM subjects WHERE teacher_id=?)",
                    con, params=(tid,),
                )
            elif table == "attendance":
                df = pd.read_sql(
                    """SELECT a.* FROM attendance a
                       WHERE a.student_id IN (SELECT s.student_id FROM students s
                           WHERE s.class_id IN (SELECT class_id FROM subjects WHERE teacher_id=?))""",
                    con, params=(tid,),
                )
            elif table == "classes":
                df = pd.read_sql(
                    "SELECT * FROM classes WHERE class_id IN (SELECT class_id FROM subjects WHERE teacher_id=?)",
                    con, params=(tid,),
                )
            else:
                df = pd.read_sql(f"SELECT * FROM {table}", con)
        else:
            sid = st.session_state.reg_id
            if table == "students":
                df = pd.read_sql(
                    "SELECT student_id, name, class_id, phone, language, date, grades, attendance FROM students WHERE student_id=?",
                    con, params=(sid,),
                )
            elif table == "teachers":
                df = pd.read_sql(
                    """SELECT DISTINCT t.teacher_id, t.name, t.phone, t.date FROM teachers t
                       JOIN subjects sub ON sub.teacher_id = t.teacher_id
                       WHERE sub.class_id = (SELECT class_id FROM students WHERE student_id=?)""",
                    con, params=(sid,),
                )
            elif table == "subjects":
                df = pd.read_sql(
                    "SELECT * FROM subjects WHERE class_id=(SELECT class_id FROM students WHERE student_id=?)",
                    con, params=(sid,),
                )
            elif table in ("grading", "cw_grading", "grades"):
                df = pd.read_sql(f"SELECT * FROM {table} WHERE student_id=?", con, params=(sid,))
            elif table == "attendance":
                df = pd.read_sql("SELECT * FROM attendance WHERE student_id=?", con, params=(sid,))
            else:
                df = pd.DataFrame()

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True

        )
    finally:
        con.close()


# 10. Справочники (только админ): ученики/учителя/классы/предметы

def page_directory() -> None:
    st.header("👥 Справочники")
    tab_s, tab_t, tab_c, tab_sub = st.tabs(["Ученики", "Учителя", "Классы", "Предметы"])

    # ---- Ученики ----
    with tab_s:
        con = get_conn()
        try:
            st.dataframe(pd.read_sql("SELECT * FROM students", con), use_container_width=True, hide_index=True)
        finally:
            con.close()
        with st.form("add_student"):
            st.subheader("Добавить ученика")
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("ФИО")
                class_id = st.number_input("ID класса", min_value=1, step=1, value=1)
                phone = st.number_input("Телефон", step=1, min_value=0)
            with c2:
                language = st.selectbox("Язык", ["Английский", "Немецкий", "Французский"])
                birth = st.date_input("Дата рождения", value=date(2010, 1, 1))
                password = st.text_input("Пароль", value="psword-new", max_chars=32)
            if st.form_submit_button("Добавить", type="primary"):
                if not name.strip():
                    st.error("Введите ФИО")
                else:
                    con = get_conn()
                    try:
                        con.execute(
                            """INSERT INTO students(name, class_id, phone, language, date, grades, attendance, passwords)
                               VALUES(?,?,?,?,?,?,?,?)""",
                            (name.strip(), int(class_id), int(phone), language, str(birth), 0, 0, password),
                        )
                        con.commit()
                        flash("Ученик добавлен")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                    finally:
                        con.close()
        with st.form("del_student"):
            st.subheader("Удалить ученика")
            del_name = st.text_input("ФИО для удаления")
            if st.form_submit_button("Удалить", type="secondary"):
                con = get_conn()
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM students WHERE name=?", (del_name.strip(),))
                    if cur.rowcount:
                        con.commit()
                        flash(f"Удалено: {cur.rowcount}")
                        st.rerun()
                    else:
                        st.warning("Не найдено")
                finally:
                    con.close()

    #  Учителя
    with tab_t:
        con = get_conn()
        try:
            st.dataframe(pd.read_sql("SELECT * FROM teachers", con), use_container_width=True, hide_index=True)
        finally:
            con.close()
        with st.form("add_teacher"):
            st.subheader("Добавить учителя")
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("ФИО")
                phone = st.number_input("Телефон", step=1, min_value=0)
            with c2:
                birth = st.date_input("Дата рождения", value=date(1980, 1, 1))
                password = st.text_input("Пароль", value="psword-new", max_chars=32)
            if st.form_submit_button("Добавить", type="primary"):
                if not name.strip():
                    st.error("Введите ФИО")
                else:
                    con = get_conn()
                    try:
                        con.execute("INSERT INTO teachers(name, phone, date, passwords) VALUES(?,?,?,?)",
                                    (name.strip(), int(phone), str(birth), password))
                        con.commit()
                        flash("Учитель добавлен")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                    finally:
                        con.close()
        with st.form("del_teacher"):
            st.subheader("Удалить учителя")
            del_name = st.text_input("ФИО для удаления")
            if st.form_submit_button("Удалить", type="secondary"):
                con = get_conn()
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM teachers WHERE name=?", (del_name.strip(),))
                    if cur.rowcount:
                        con.commit()
                        flash(f"Удалено: {cur.rowcount}")
                        st.rerun()
                    else:
                        st.warning("Не найдено")
                finally:
                    con.close()

    # Классы
    with tab_c:
        con = get_conn()
        try:
            st.dataframe(pd.read_sql("SELECT * FROM classes", con), use_container_width=True, hide_index=True)
        finally:
            con.close()
        with st.form("add_class"):
            st.subheader("Добавить класс")
            c1, c2, c3 = st.columns(3)
            with c1:
                class_name = st.text_input("Название (9Д)")
            with c2:
                students_number = st.number_input("Кол-во", min_value=1, step=1, value=25)
            with c3:
                teacher_id = st.number_input("ID кл. рук.", min_value=1, step=1, value=1)
            if st.form_submit_button("Добавить", type="primary"):
                if not class_name.strip():
                    st.error("Введите название")
                else:
                    con = get_conn()
                    try:
                        con.execute("INSERT INTO classes(class, students_number, teacher_id) VALUES(?,?,?)",
                                    (class_name.strip(), int(students_number), int(teacher_id)))
                        con.commit()
                        flash("Класс добавлен")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                    finally:
                        con.close()
        with st.form("del_class"):
            st.subheader("Удалить класс")
            del_name = st.text_input("Название для удаления")
            if st.form_submit_button("Удалить", type="secondary"):
                con = get_conn()
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM classes WHERE class=?", (del_name.strip(),))
                    if cur.rowcount:
                        con.commit()
                        flash(f"Удалено: {cur.rowcount}")
                        st.rerun()
                    else:
                        st.warning("Не найдено")
                finally:
                    con.close()

    #Предметы
    with tab_sub:
        con = get_conn()
        try:
            st.dataframe(pd.read_sql("SELECT * FROM subjects", con), use_container_width=True, hide_index=True)
        finally:
            con.close()
        with st.form("add_subject"):
            st.subheader("Добавить предмет")
            c1, c2, c3 = st.columns(3)
            with c1:
                subject_name = st.text_input("Название")
            with c2:
                teacher_id = st.number_input("ID учителя", min_value=1, step=1, value=1)
            with c3:
                class_id = st.number_input("ID класса", min_value=1, step=1, value=1)
            if st.form_submit_button("Добавить", type="primary"):
                if not subject_name.strip():
                    st.error("Введите название")
                else:
                    con = get_conn()
                    try:
                        con.execute("INSERT INTO subjects(subject, teacher_id, class_id) VALUES(?,?,?)",
                                    (subject_name.strip(), int(teacher_id), int(class_id)))
                        con.commit()
                        flash("Предмет добавлен")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                    finally:
                        con.close()
        with st.form("del_subject"):
            st.subheader("Удалить предмет")
            del_id = st.number_input("ID предмета", min_value=1, step=1, value=1)
            if st.form_submit_button("Удалить", type="secondary"):
                con = get_conn()
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM subjects WHERE subject_id=?", (int(del_id),))
                    if cur.rowcount:
                        con.commit()
                        flash("Предмет удалён")
                        st.rerun()
                    else:
                        st.warning("Не найдено")
                finally:
                    con.close()



# 11. Оценки

def _confirm_and_add(student_id: int, subject_id: int, grade: int, gdate: date, is_cw: bool) -> None:
    kind = "за КР" if is_cw else "текущую"

    @st.dialog("Подтверждение")
    def dialog() -> None:
        st.write(f"Выставить {kind} оценку **{grade}** ученику ID {student_id}?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Нет", use_container_width=True):
                st.rerun()
        with c2:
            if st.button("Да", type="primary", use_container_width=True):
                try:
                    add_grade(int(student_id), int(subject_id), int(grade), gdate, is_cw)
                    st.success("Оценка выставлена")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    dialog()


def page_grades() -> None:
    st.header("✍️ Выставление оценок")

    tab1, tab2, tab3 = st.tabs(["Текущая оценка", "Оценка за КР", "Посещаемость"])

    today = date.today()
    min_allowed = today - timedelta(days=2)

    with tab1:
        with st.form("form_grade"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                sid = st.number_input("ID ученика", min_value=1, step=1, key="g_sid")
            with c2:
                subj = st.number_input("ID предмета", min_value=1, step=1, key="g_sub")
            with c3:
                grade = st.number_input("Оценка", min_value=1, max_value=5, step=1, key="g_val")
            with c4:
                gdate = st.date_input("Дата", value=today, min_value=min_allowed, max_value=today, key="g_date")
            submitted = st.form_submit_button("Выставить", type="primary", use_container_width=True)
        if submitted:
            _confirm_and_add(sid, subj, int(grade), gdate, is_cw=False)
        con = get_conn()
        try:
            st.subheader("Последние текущие оценки")
            st.dataframe(
                pd.read_sql("SELECT * FROM grading ORDER BY date DESC LIMIT 50", con),
                use_container_width=True, hide_index=True,
            )
        finally:
            con.close()

    with tab2:
        with st.form("form_cw"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                sid = st.number_input("ID ученика", min_value=1, step=1, key="cw_sid")
            with c2:
                subj = st.number_input("ID предмета", min_value=1, step=1, key="cw_sub")
            with c3:
                grade = st.number_input("Оценка", min_value=1, max_value=5, step=1, key="cw_val")
            with c4:
                gdate = st.date_input("Дата", value=today, min_value=min_allowed, max_value=today, key="cw_date")
            submitted = st.form_submit_button("Выставить КР", type="primary", use_container_width=True)
        if submitted:
            _confirm_and_add(sid, subj, int(grade), gdate, is_cw=True)
        con = get_conn()
        try:
            st.subheader("Последние оценки за КР")
            st.dataframe(
                pd.read_sql("SELECT * FROM cw_grading ORDER BY date DESC LIMIT 50", con),
                use_container_width=True, hide_index=True,
            )
        finally:
            con.close()

    with tab3:
        con = get_conn()
        try:
            st.subheader("Последние отметки посещаемости")
            st.dataframe(
                pd.read_sql("SELECT * FROM attendance ORDER BY date DESC LIMIT 50", con),
                use_container_width=True, hide_index=True,
            )
        finally:
            con.close()
        with st.form("form_att"):
            c1, c2, c3 = st.columns(3)
            with c1:
                att_sid = st.number_input("ID ученика", min_value=1, step=1, key="att_sid")
            with c2:
                att_status = st.selectbox("Статус", ["Был", "Отсутствовал"], key="att_status")
            with c3:
                att_date = st.date_input("Дата", value=today, min_value=today, max_value=today, key="att_date")
            if st.form_submit_button("Сохранить", type="primary", use_container_width=True):
                was = 1 if att_status == "Был" else 0
                con = get_conn()
                try:
                    cur = con.cursor()
                    cur.execute("SELECT 1 FROM attendance WHERE student_id=? AND date=?", (int(att_sid), str(att_date)))
                    if cur.fetchone():
                        st.error("Уже отмечено за эту дату")
                    else:
                        cur.execute("SELECT name FROM students WHERE student_id=?", (int(att_sid),))
                        row = cur.fetchone()
                        if not row:
                            st.error("Ученик не найден")
                        else:
                            cur.execute(
                                "INSERT INTO attendance(student_id, name, was, date) VALUES(?,?,?,?)",
                                (int(att_sid), row[0], was, str(att_date)),
                            )
                            con.commit()
                            flash("Посещаемость сохранена")
                            st.rerun()
                finally:
                    con.close()



# 12. Запросы
ALLOWED_TABLES = {"students", "subjects", "classes", "teachers", "grades", "grading", "cw_grading", "attendance"}
ALLOWED_COLUMNS = {
    "students": ["student_id", "name", "class_id", "phone", "language", "date", "grades", "attendance"],
    "subjects": ["subject_id", "subject", "teacher_id", "class_id"],
    "classes": ["class_id", "class", "students_number", "teacher_id"],
    "teachers": ["teacher_id", "name", "phone", "date"],
    "grades": ["grade_id", "student_id", "name", "subject_id", "subject", "grade", "cw_grade"],
    "grading": ["student_id", "name", "grade", "subject", "subject_id", "date"],
    "cw_grading": ["student_id", "name", "cw_grade", "subject", "subject_id", "date"],
    "attendance": ["student_id", "name", "was", "date"],
}
NUMERIC_COLS = {"student_id", "class_id", "phone", "subject_id", "teacher_id", "students_number",
                "grade", "cw_grade", "grades", "attendance", "was"}


def page_queries() -> None:
    st.header("🔎 Запросы")

    with st.expander("Конструктор (безопасный)", expanded=True):
        tbl_label = st.selectbox("Таблица", list(ALLOWED_TABLES))
        cols = ALLOWED_COLUMNS[tbl_label]
        col = st.selectbox("Колонка", cols)
        op = st.selectbox("Оператор", ["=", "<", ">", "<=", ">=", "LIKE"])

        if col in NUMERIC_COLS:
            val = st.number_input("Значение", step=1)
        elif col == "date":
            val = str(st.date_input("Значение", value=date.today()))
        else:
            val = st.text_input("Значение")

        if st.button("Выполнить", type="primary", key="q_builder", use_container_width=True):
            sql = f"SELECT * FROM {tbl_label} WHERE {col} {op} ?"
            con = get_conn()
            try:
                df = pd.read_sql(sql, con, params=(val,))
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Ошибка: {e}")
            finally:
                con.close()

    if st.session_state.role_check == 3:
        with st.expander("Произвольный SQL (админ)", expanded=False):
            sql_text = st.text_area("SQL", height=120, placeholder="SELECT * FROM students LIMIT 5;")
            if st.button("Выполнить SQL", key="q_raw", use_container_width=True):
                if not sql_text.strip():
                    st.error("Пустой запрос")
                else:
                    con = get_conn()
                    try:
                        if sql_text.strip().lower().startswith("select"):
                            df = pd.read_sql(sql_text, con)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            cur = con.cursor()
                            cur.execute(sql_text)
                            con.commit()
                            st.success(f"Выполнено. Строк: {cur.rowcount}")
                    except Exception as e:
                        st.error(f"Ошибка SQL: {e}")
                    finally:
                        con.close()
# 13. ИИ
def build_db_context_for_role() -> str:
    rc = st.session_state.role_check
    rid = st.session_state.reg_id
    con = get_conn()
    try:
        if rc == 3:
            parts = []
            for t in ["students", "teachers", "classes", "subjects", "grades", "grading", "cw_grading", "attendance"]:
                parts.append(f"[{t}]\n" + pd.read_sql(f"SELECT * FROM {t}", con).to_string())
            return "\n\n".join(parts)

        if rc == 2:
            tid = rid
            subjects = pd.read_sql("SELECT * FROM subjects WHERE teacher_id=?", con, params=(tid,))
            classes_ids = tuple(subjects["class_id"].unique().tolist()) or (-1,)
            q = ",".join("?" * len(classes_ids))
            students = pd.read_sql(
                f"SELECT student_id, name, class_id, phone, language, date, grades, attendance FROM students WHERE class_id IN ({q})",
                con, params=classes_ids,
            )
            grading = pd.read_sql(
                "SELECT * FROM grading WHERE subject_id IN (SELECT subject_id FROM subjects WHERE teacher_id=?)",
                con, params=(tid,),
            )
            cw = pd.read_sql(
                "SELECT * FROM cw_grading WHERE subject_id IN (SELECT subject_id FROM subjects WHERE teacher_id=?)",
                con, params=(tid,),
            )
            att = pd.read_sql(
                f"SELECT * FROM attendance WHERE student_id IN (SELECT student_id FROM students WHERE class_id IN ({q}))",
                con, params=classes_ids,
            )
            classes = pd.read_sql(f"SELECT * FROM classes WHERE class_id IN ({q})", con, params=classes_ids)
            return (
                f"[subjects]\n{subjects.to_string()}\n\n"
                f"[students]\n{students.to_string()}\n\n"
                f"[classes]\n{classes.to_string()}\n\n"
                f"[grading]\n{grading.to_string()}\n\n"
                f"[cw_grading]\n{cw.to_string()}\n\n"
                f"[attendance]\n{att.to_string()}"
            )

        sid = rid
        student = pd.read_sql(
            "SELECT student_id, name, class_id, phone, language, date, grades, attendance FROM students WHERE student_id=?",
            con, params=(sid,),
        )
        subjects = pd.read_sql(
            "SELECT * FROM subjects WHERE class_id=(SELECT class_id FROM students WHERE student_id=?)",
            con, params=(sid,),
        )
        grading = pd.read_sql("SELECT * FROM grading WHERE student_id=?", con, params=(sid,))
        cw = pd.read_sql("SELECT * FROM cw_grading WHERE student_id=?", con, params=(sid,))
        att = pd.read_sql("SELECT * FROM attendance WHERE student_id=?", con, params=(sid,))
        return (
            f"[students]\n{student.to_string()}\n\n"
            f"[subjects]\n{subjects.to_string()}\n\n"
            f"[grading]\n{grading.to_string()}\n\n"
            f"[cw_grading]\n{cw.to_string()}\n\n"
            f"[attendance]\n{att.to_string()}"
        )
    finally:
        con.close()


def _call_gemini(prompt: str) -> str:
    api_key = get_api_key()
    if not api_key:
        return "⚠️ API-ключ Gemini не настроен. Добавьте GEMINI_API_KEY в .streamlit/secrets.toml."
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return resp.text or "(пустой ответ)"


def _render_chat(chat_key: str, placeholder: str, build_prompt) -> None:
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    user_text = st.chat_input(placeholder, key=f"input_{chat_key}")
    if user_text:
        st.session_state[chat_key].append({"role": "user", "text": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    answer = _call_gemini(build_prompt(user_text))
                except Exception as e:
                    answer = f"Ошибка ИИ: {e}"
                st.markdown(answer)
        st.session_state[chat_key].append({"role": "assistant", "text": answer})
        st.rerun()


def page_chat_school() -> None:
    st.header("🤖 ИИ-агент по школьной базе")
    st.caption("Задавайте вопросы по данным школы. Доступ ограничен вашей ролью.")

    def build(user_text: str) -> str:
        ctx = build_db_context_for_role()
        history = "\n".join(f"{m['role']}: {m['text']}" for m in st.session_state["chat_school"][-10:])
        role = ROLE_LABEL[st.session_state.role_check]
        return (
            f"Ты — школьный ИИ-агент. Роль пользователя: {role}. ID: {st.session_state.reg_id}.\n"
            "Правила: ученик видит только свои данные; учитель — свои классы и учеников; "
            "администратор — всё. Пароли не выводить никому, кроме администратора. "
            "Если данных для ответа нет — честно скажи об этом. "
            "Если ученик просит решить задание — объясни метод, не давай списать.\n\n"
            f"КОНТЕКСТ БД:\n{ctx}\n\n"
            f"ИСТОРИЯ:\n{history}\n\n"
            f"ВОПРОС: {user_text}"
        )

    _render_chat("chat_school", "Ваш вопрос по школе...", build)


def page_chat_tutor() -> None:
    st.header("🎓 ИИ-репетитор")
    st.caption("Помогу разобраться в теме, но не дам списать.")

    def build(user_text: str) -> str:
        history = "\n".join(f"{m['role']}: {m['text']}" for m in st.session_state["chat_tutor"][-10:])
        role = ROLE_LABEL[st.session_state.role_check]
        return (
            f"Ты — школьный ИИ-репетитор. Роль: {role}.\n"
            "Не давай готовых ответов на домашку — объясняй метод и шаги. "
            "Если просят составить задания — проверь их на решаемость. "
            "Отвечай дружелюбно и по делу.\n\n"
            f"ИСТОРИЯ:\n{history}\n\n"
            f"ЗАПРОС: {user_text}"
        )

    _render_chat("chat_tutor", "Спросите репетитора...", build)


# 14. Аналитика (3 группы → подвкладки)
def page_analytics() -> None:
    st.header("📈 Аналитика")

    group = st.radio(
        "Раздел аналитики",
        ["📊 Оценки", "🏆 Обзор"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Группа: Оценки
    if group == "📊 Оценки":
        sub = st.radio(
            "Подраздел",
            ["Динамика", "Средний балл", "Прогноз (среднее)", "Прогноз (ML)", "Освоение"],
            horizontal=True,
            label_visibility="collapsed",
            key="sub_grades",
        )

        if sub == "Динамика":
            st.subheader("Динамика оценок ученика")
            c1, c2 = st.columns(2)
            with c1:
                st_id = st.number_input("ID ученика", min_value=1, step=1, value=2, key="g1_st")
            with c2:
                sub_id = st.number_input("ID предмета", min_value=1, step=1, value=1, key="g1_sub")

            con = get_conn()
            try:
                df = pd.read_sql(
                    """SELECT grade AS mark, date, 'Текущая' AS type FROM grading
                       WHERE student_id=? AND subject_id=?
                       UNION ALL
                       SELECT cw_grade AS mark, date, 'КР' AS type FROM cw_grading
                       WHERE student_id=? AND subject_id=?
                       ORDER BY date ASC""",
                    con, params=(st_id, sub_id, st_id, sub_id),
                )
            finally:
                con.close()

            if df.empty:
                st.warning("Нет оценок.")
            else:
                st.write(f"**Все оценки:** {', '.join(map(str, df['mark'].tolist()))}")
                st.dataframe(df, use_container_width=True, hide_index=True)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                fig = px.line(df, x="date", y="mark", color="type", markers=True,
                              title="Динамика", range_y=[1, 5])
                st.plotly_chart(fig, use_container_width=True)

        elif sub == "Средний балл":
            st.subheader("Средний балл по предмету")
            sub_id_g2 = st.number_input("ID предмета", min_value=1, step=1, value=1, key="g2_sub")
            con = get_conn()
            try:
                df = pd.read_sql(
                    """SELECT grade AS mark FROM grading WHERE subject_id=?
                       UNION ALL
                       SELECT cw_grade AS mark FROM cw_grading WHERE subject_id=?""",
                    con, params=(sub_id_g2, sub_id_g2),
                )
            finally:
                con.close()
            if df.empty:
                st.warning("Нет оценок.")
            else:
                st.metric("Средний балл", round(float(df["mark"].mean()), 2))
                st.dataframe(df, use_container_width=True, hide_index=True)

        elif sub == "Прогноз (среднее)":
            st.subheader("Прогноз за четверть (по среднему)")
            c1, c2 = st.columns(2)
            with c1:
                st_id_q = st.number_input("ID ученика", min_value=1, step=1, value=1, key="g3_st")
            with c2:
                sub_id_q = st.number_input("ID предмета", min_value=1, step=1, value=1, key="g3_sub")

            con = get_conn()
            try:
                df = pd.read_sql(
                    """SELECT grade AS mark, date, 'Текущая' AS type FROM grading
                       WHERE student_id=? AND subject_id=?
                       UNION ALL
                       SELECT cw_grade AS mark, date, 'КР' AS type FROM cw_grading
                       WHERE student_id=? AND subject_id=?""",
                    con, params=(st_id_q, sub_id_q, st_id_q, sub_id_q),
                )
                st_name = pd.read_sql("SELECT name FROM students WHERE student_id=?", con, params=(st_id_q,))
                sub_name = pd.read_sql("SELECT subject FROM subjects WHERE subject_id=?", con, params=(sub_id_q,))
            finally:
                con.close()

            student_name = st_name.iloc[0]["name"] if not st_name.empty else "—"
            subject_name = sub_name.iloc[0]["subject"] if not sub_name.empty else "—"

            if df.empty:
                st.warning(f"У {student_name} нет оценок по «{subject_name}».")
            else:
                raw = float(df["mark"].mean())
                rounded = max(1, min(5, round(raw)))
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Средний балл", round(raw, 2))
                with c2:
                    st.metric("Прогноз", rounded)
                if raw >= 4.5:
                    st.success("Идёт на «5»")
                elif raw >= 3.5:
                    st.info("Идёт на «4»")
                elif raw >= 2.5:
                    st.warning("Идёт на «3»")
                else:
                    st.error("Риск «2»")
                st.dataframe(df, use_container_width=True, hide_index=True)

        elif sub == "Прогноз (ML)":
            st.subheader("Прогноз ML (линейная регрессия)")
            c1, c2 = st.columns(2)
            with c1:
                st_id_ml = st.number_input("ID ученика", min_value=1, step=1, value=1, key="g4_st")
            with c2:
                sub_id_ml = st.number_input("ID предмета", min_value=1, step=1, value=1, key="g4_sub")

            con = get_conn()
            try:
                df = pd.read_sql(
                    """SELECT date, grade AS mark, 1 AS weight FROM grading
                       WHERE student_id=? AND subject_id=?
                       UNION ALL
                       SELECT date, cw_grade AS mark, 2 AS weight FROM cw_grading
                       WHERE student_id=? AND subject_id=?
                       ORDER BY date ASC""",
                    con, params=(st_id_ml, sub_id_ml, st_id_ml, sub_id_ml),
                )
            finally:
                con.close()

            if df.empty or len(df) < 3:
                st.warning("Нужно ≥ 3 оценок.")
            else:
                df["lesson_index"] = range(1, len(df) + 1)
                X = df[["lesson_index", "weight"]].values
                y = df["mark"].values
                model = LinearRegression().fit(X, y)
                future = np.array([[len(df) + 1, 1], [len(df) + 2, 1], [len(df) + 3, 2]])
                pred_raw = float(model.predict(future).mean())
                pred = max(2.0, min(5.0, pred_raw))
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Сырой прогноз", round(pred_raw, 2))
                with c2:
                    st.metric("Итог", round(pred))
                coef = model.coef_[0]
                if coef > 0.05:
                    st.success("Тренд: рост")
                elif coef < -0.05:
                    st.error("Тренд: снижение")
                else:
                    st.info("Тренд: стабильно")
                st.dataframe(df[["date", "mark", "weight"]], use_container_width=True, hide_index=True)

        elif sub == "Освоение":
            st.subheader("Анализ освоения и посещаемости")
            c1, c2 = st.columns(2)
            with c1:
                an_sid = st.number_input("ID ученика", min_value=1, step=1, value=1, key="an_sid")
            with c2:
                an_sub = st.number_input("ID предмета", min_value=1, step=1, value=1, key="an_sub")

            if st.button("Провести аналитику", key="btn_an", use_container_width=True):
                con = get_conn()
                try:
                    df_g = pd.read_sql(
                        """SELECT grade AS mark, date FROM grading WHERE student_id=? AND subject_id=?
                           UNION ALL
                           SELECT cw_grade AS mark, date FROM cw_grading WHERE student_id=? AND subject_id=?
                           ORDER BY date ASC""",
                        con, params=(an_sid, an_sub, an_sid, an_sub),
                    )
                    df_a = pd.read_sql("SELECT was FROM attendance WHERE student_id=?", con, params=(an_sid,))
                finally:
                    con.close()

                if df_g.empty:
                    st.warning("Нет данных.")
                else:
                    st.dataframe(df_g, use_container_width=True, hide_index=True)
                    avg = float(df_g["mark"].mean())
                    last = int(df_g["mark"].iloc[-1])
                    drop = last < avg - 0.5
                    absence = bool((df_a["was"] == 0).any()) if not df_a.empty else False
                    st.write(f"**Средний:** {round(avg, 2)} | **Последняя:** {last}")
                    if drop and absence:
                        st.error("Просадка + пропуски: часть программы не усвоена.")
                    elif drop and not absence:
                        st.error("Просадка без пропусков: материал сложный.")
                    elif not drop and absence:
                        st.warning("Пропуски без просадки: возможно, догнал сам.")
                    else:
                        st.success("Всё в норме.")

    #Группа: Обзор
    elif group == "🏆 Обзор":
        st.subheader("Комплексная аналитика")
        con = get_conn()
        try:
            df_grading = pd.read_sql("SELECT * FROM grading", con)
            df_cw = pd.read_sql("SELECT * FROM cw_grading", con)
            df_students = pd.read_sql("SELECT * FROM students", con)
            df_subjects = pd.read_sql("SELECT * FROM subjects", con)
            df_classes = pd.read_sql("SELECT * FROM classes", con)
            df_att = pd.read_sql("SELECT * FROM attendance", con)
        finally:
            con.close()

        with st.expander("1. Средний текущий балл по предметам", expanded=False):
            if not df_grading.empty:
                agg = df_grading.groupby("subject")["grade"].mean().reset_index()
                st.plotly_chart(px.bar(agg, x="subject", y="grade", color="subject", text_auto=".2f"),
                                use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("2. Доли текущих оценок"):
            if not df_grading.empty:
                st.plotly_chart(px.pie(df_grading, names="grade", hole=0.3), use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("3. Количество оценок по датам"):
            if not df_grading.empty:
                cnt = df_grading.groupby("date").size().reset_index(name="count")
                st.plotly_chart(px.line(cnt, x="date", y="count", markers=True), use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("4. Средний балл за КР по предметам"):
            if not df_cw.empty:
                agg = df_cw.groupby("subject")["cw_grade"].mean().reset_index()
                st.plotly_chart(px.bar(agg, x="subject", y="cw_grade", color="subject", text_auto=".2f"),
                                use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("5. Распределение учеников по классам"):
            if not df_students.empty and not df_classes.empty:
                merged = df_students.merge(df_classes, on="class_id", how="left", suffixes=("", "_cl"))
                counts = merged["class"].value_counts().reset_index()
                counts.columns = ["class", "count"]
                st.plotly_chart(px.pie(counts, names="class", values="count", hole=0.6),
                                use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("6. Рейтинг учеников"):
            if not df_students.empty:
                top = df_students.sort_values("grades", ascending=False).head(10)
                st.plotly_chart(px.bar(top, x="grades", y="name", orientation="h", text_auto=True),
                                use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("7. Посещаемость по ученикам"):
            if not df_att.empty:
                att_agg = df_att.groupby(["name", "was"]).size().reset_index(name="cnt")
                att_agg["Статус"] = att_agg["was"].map({1: "Был", 0: "Нет"})
                st.plotly_chart(px.bar(att_agg, x="name", y="cnt", color="Статус", barmode="group"),
                                use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("8. Общий процент посещаемости"):
            if not df_att.empty:
                df_att["status"] = df_att["was"].map({1: "Был", 0: "Пропуск"})
                st.plotly_chart(px.pie(df_att, names="status"), use_container_width=True)
            else:
                st.info("Нет данных.")

        with st.expander("9. Распределение оценок по ученикам"):
            if not df_grading.empty:
                st.plotly_chart(
                    px.scatter(df_grading, x="student_id", y="grade", color="subject",
                               hover_data=["name", "date"]),
                    use_container_width=True,
                )
            else:
                st.info("Нет данных.")

        with st.expander("10. Количество предметов по классам"):
            if not df_subjects.empty and not df_classes.empty:
                merged = df_subjects.merge(df_classes, on="class_id", how="left", suffixes=("", "_cl"))
                counts = merged["class"].value_counts().reset_index()
                counts.columns = ["class", "cnt"]
                st.plotly_chart(px.bar(counts, x="class", y="cnt", color="class", text_auto=True),
                                use_container_width=True)
            else:
                st.info("Нет данных.")




if nav == "📊 Таблицы":
    page_tables()
elif nav == "👥 Справочники":
    page_directory()
elif nav == "✍️ Оценки":
    page_grades()
elif nav == "🔎 Запросы":
    page_queries()
elif nav == "📈 Аналитика":
    page_analytics()
elif nav == "🤖 ИИ-агент":
    page_chat_school()
elif nav == "🎓 Репетитор":
    page_chat_tutor()
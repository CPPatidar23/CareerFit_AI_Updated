import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# CareerFit — AI-Powered Career Guidance Expert System
# ============================================================

st.set_page_config(
    page_title="CareerFit | AI Career Advisor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- Paths -----------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "careerfit_logistic_regression.pkl"
VECTORIZER_PATH = BASE_DIR / "careerfit_tfidf_vectorizer.pkl"
DATA_PATH = BASE_DIR / "Career_Dataset.csv"

# ------------------------- Styling ----------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(99,102,241,.13), transparent 28%),
            radial-gradient(circle at 95% 15%, rgba(14,165,233,.11), transparent 26%),
            #f7f9fc;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        padding: 30px 34px;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #312e81 55%, #0f766e 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(17,24,39,.18);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        line-height: 1.05;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1.5px;
    }

    .hero p {
        font-size: 16px;
        opacity: .88;
        margin: 12px 0 0;
        max-width: 850px;
    }

    .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.13);
        border: 1px solid rgba(255,255,255,.18);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 14px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,.94);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(15,23,42,.06);
        margin-bottom: 15px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(15,23,42,.05);
        min-height: 105px;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .metric-value {
        color: #111827;
        font-size: 26px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Recommendation */
    .recommendation {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 21px;
        margin: 12px 0;
        box-shadow: 0 8px 25px rgba(15,23,42,.06);
    }

    .rank {
        font-size: 13px;
        font-weight: 800;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .career-name {
        font-size: 23px;
        font-weight: 800;
        color: #111827;
        margin: 4px 0 7px;
    }

    .confidence {
        font-size: 15px;
        color: #475569;
        font-weight: 600;
    }

    .bar-bg {
        height: 9px;
        border-radius: 99px;
        background: #e5e7eb;
        overflow: hidden;
        margin-top: 10px;
    }

    .bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
    }

    .tag {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
        font-weight: 600;
    }

    .info-box {
        border-radius: 15px;
        padding: 15px 17px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        margin: 10px 0 18px;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 25px 0 5px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #172554 100%);
    }

    div[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    div[data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,.08);
        color: white;
        border: 1px solid rgba(255,255,255,.15);
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- Load model ---------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Model files could not be loaded.")
    st.code(str(e))
    st.info(
        "Keep careerfit_logistic_regression.pkl and "
        "careerfit_tfidf_vectorizer.pkl in the same folder as app.py."
    )
    st.stop()

dataset = load_dataset()

# ---------------------- Career metadata -----------------------

CAREER_INFO = {
    "AI/ML Engineer": {
        "icon": "🤖",
        "desc": "Build intelligent systems using machine learning, deep learning and AI techniques.",
        "skills": ["Python", "Machine Learning", "Deep Learning", "Statistics", "AI"]
    },
    "Business Analyst": {
        "icon": "📊",
        "desc": "Bridge business needs and technology through requirements, analysis and data-driven insights.",
        "skills": ["SQL", "Excel", "Data Analysis", "Communication", "Business Analysis"]
    },
    "Cloud Engineer": {
        "icon": "☁️",
        "desc": "Design, deploy and maintain scalable cloud infrastructure and services.",
        "skills": ["Cloud", "Linux", "Networking", "DevOps", "Security"]
    },
    "Cybersecurity Analyst": {
        "icon": "🛡️",
        "desc": "Monitor systems, investigate threats and help protect applications, networks and data.",
        "skills": ["Cybersecurity", "Networking", "Security Analysis", "Linux", "Risk Management"]
    },
    "Data Scientist": {
        "icon": "🔬",
        "desc": "Extract insights from data using statistics, machine learning and analytical methods.",
        "skills": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"]
    },
    "Database Administrator": {
        "icon": "🗄️",
        "desc": "Manage databases, performance, security, availability and data reliability.",
        "skills": ["SQL", "Database Management", "Oracle", "Backup", "Security"]
    },
    "Digital Marketing Specialist": {
        "icon": "📣",
        "desc": "Use digital channels, content and analytics to build audiences and business growth.",
        "skills": ["SEO", "Content Marketing", "Social Media", "Analytics", "Communication"]
    },
    "Network Engineer": {
        "icon": "🌐",
        "desc": "Design, configure and troubleshoot computer networks and communication infrastructure.",
        "skills": ["Networking", "TCP/IP", "Routing", "Switching", "Network Security"]
    },
    "Software Developer": {
        "icon": "💻",
        "desc": "Design, develop, test and maintain software applications and systems.",
        "skills": ["Programming", "Java", "Python", "SQL", "Problem Solving"]
    },
    "UI/UX Designer": {
        "icon": "🎨",
        "desc": "Create useful, accessible and visually engaging digital experiences.",
        "skills": ["UI Design", "UX Research", "Wireframing", "Figma", "Creativity"]
    }
}

# ---------------------- Sidebar -------------------------------

with st.sidebar:
    st.markdown("## 🎯 CareerFit")
    st.caption("AI-Powered Career Guidance Expert System")

    st.markdown("---")
    st.markdown("### 🧭 How it works")
    st.markdown("""
    1. Enter your academic background.
    2. Describe your skills and interests.
    3. Add your personality and career goal.
    4. CareerFit converts the text into TF-IDF features.
    5. Logistic Regression generates career probabilities.
    6. The top career matches are displayed.
    """)

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown("**Algorithm:** Logistic Regression")
    st.markdown("**Features:** TF-IDF")
    st.markdown("**Classes:** 10 career paths")

    if dataset is not None:
        st.markdown("---")
        st.markdown("### 📁 Dataset")
        st.markdown(f"**Records:** {len(dataset):,}")
        st.markdown(f"**Fields:** {len(dataset.columns)}")


# ---------------------- Header --------------------------------

st.markdown("""
<div class="hero">
    <div class="pill">AI-POWERED CAREER GUIDANCE</div>
    <h1>Find a Career That Fits You.</h1>
    <p>
        Tell CareerFit about your skills, interests, personality and goals.
        The trained machine-learning model analyzes your profile and returns
        personalized career recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Dashboard metrics --------------------

classes = list(model.classes_)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Career Paths</div>
        <div class="metric-value">10</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    vocab_size = len(getattr(vectorizer, "vocabulary_", {}))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TF-IDF Vocabulary</div>
        <div class="metric-value">{vocab_size:,}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    rows = len(dataset) if dataset is not None else 50_000
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Dataset</div>
        <div class="metric-value">{rows:,}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Prediction Type</div>
        <div class="metric-value">Top 3</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------- Main tabs -----------------------------

tab1, tab2, tab3 = st.tabs([
    "🎯 Career Assessment",
    "📈 Career Explorer",
    "ℹ️ About CareerFit"
])

# ====================== TAB 1 =================================

with tab1:
    st.markdown("## Build Your Career Profile")
    st.markdown(
        '<div class="info-box">'
        '💡 <b>Tip:</b> Use specific keywords such as Python, SQL, Machine Learning, '
        'Figma, Networking, SEO, Cloud, etc. More descriptive profile information '
        'helps the text model use more relevant signals.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("career_form"):
        st.markdown("### 👤 Personal & Academic Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            student_name = st.text_input(
                "Student Name",
                placeholder="e.g. Rahul Kumar"
            )

        with c2:
            education_options = [
                "B.Tech", "B.E.", "B.Sc", "BCA", "MCA", "M.Tech", "M.Sc",
                "Diploma", "BA", "B.A.", "BFA", "B.Des", "MBA", "BBA",
                "B.Com", "M.Com"
            ]
            education = st.selectbox("Education", education_options)

        with c3:
            stream_options = [
                "Computer Science", "Information Technology", "Science",
                "Mathematics", "Statistics", "Design", "Arts", "Commerce"
            ]
            academic_stream = st.selectbox(
                "Academic Stream", stream_options
            )

        c4, c5 = st.columns([1, 2])

        with c4:
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.01
            )

        with c5:
            career_goal = st.text_input(
                "Career Goal",
                placeholder="e.g. Artificial Intelligence, Software Development"
            )

        st.markdown("### 🧩 Your Skills & Interests")

        skills = st.text_area(
            "Skills",
            placeholder="e.g. Python, SQL, Machine Learning, Data Analysis, Communication",
            height=100
        )

        interests = st.text_area(
            "Interests",
            placeholder="e.g. Artificial Intelligence, Technology, Problem Solving, Research",
            height=100
        )

        favorite_subjects = st.text_area(
            "Favorite Subjects",
            placeholder="e.g. Mathematics, Computer Science, Statistics, Database Management",
            height=90
        )

        personality = st.text_input(
            "Personality",
            placeholder="e.g. Logical, Problem Solver, Curious, Team-oriented"
        )

        st.markdown("")
        submitted = st.form_submit_button(
            "🚀 Analyze My Career Profile",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        # Match the notebook's exact text-feature construction.
        combined_text = " ".join([
            str(skills or ""),
            str(interests or ""),
            str(favorite_subjects or ""),
            str(personality or ""),
            str(career_goal or "")
        ])

        combined_text = " ".join(
            combined_text.replace(",", " ")
                         .replace(";", " ")
                         .replace("/", " ")
                         .replace("|", " ")
                         .split()
        )

        if not combined_text.strip():
            st.warning(
                "Please enter at least some skills, interests, subjects, "
                "personality traits or a career goal."
            )
        else:
            with st.spinner("Analyzing your profile..."):
                vector = vectorizer.transform([combined_text])
                probabilities = model.predict_proba(vector)[0]
                order = np.argsort(probabilities)[::-1][:3]

            st.markdown("---")
            st.markdown("## 🎯 Your Career Matches")

            if student_name.strip():
                st.markdown(
                    f"### Hello, **{student_name.strip()}** 👋"
                )

            for rank, idx in enumerate(order, start=1):
                career = model.classes_[idx]
                confidence = float(probabilities[idx]) * 100
                info = CAREER_INFO.get(
                    career,
                    {
                        "icon": "🎯",
                        "desc": "Career recommendation generated by the trained model.",
                        "skills": []
                    }
                )

                st.markdown(f"""
                <div class="recommendation">
                    <div class="rank">MATCH #{rank}</div>
                    <div class="career-name">
                        {info['icon']} {career}
                    </div>
                    <div class="confidence">
                        Model probability: {confidence:.2f}%
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width:{min(confidence,100):.2f}%"></div>
                    </div>
                    <p style="margin-top:13px;color:#475569;">
                        {info['desc']}
                    </p>
                    <div>
                        {''.join([f"<span class='tag'>{x}</span>" for x in info['skills']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Full probability table
            st.markdown("### 📊 Model Probability Distribution")

            result_df = pd.DataFrame({
                "Career": model.classes_,
                "Probability (%)": probabilities * 100
            }).sort_values(
                "Probability (%)", ascending=False
            ).reset_index(drop=True)

            result_df["Probability (%)"] = result_df["Probability (%)"].round(2)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Career": st.column_config.TextColumn("Career"),
                    "Probability (%)": st.column_config.ProgressColumn(
                        "Probability (%)",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
                }
            )

            st.success(
                "Assessment complete. These results represent model probabilities "
                "based on the profile information provided."
            )


# ====================== TAB 2 =================================

with tab2:
    st.markdown("## 📈 Explore Career Paths")
    st.write("Browse the career categories available in the trained CareerFit model.")

    selected_career = st.selectbox(
        "Select a career",
        classes,
        format_func=lambda x: f"{CAREER_INFO.get(x, {}).get('icon', '🎯')} {x}"
    )

    info = CAREER_INFO.get(selected_career, {})

    c1, c2 = st.columns([1.25, 1])

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="rank">CAREER PROFILE</div>
            <div class="career-name">
                {info.get('icon', '🎯')} {selected_career}
            </div>
            <p style="color:#475569;font-size:15px;">
                {info.get('desc', 'Career path available in the model.')}
            </p>
            <h4>Core Skills</h4>
            {''.join([f"<span class='tag'>{x}</span>" for x in info.get('skills', [])])}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if dataset is not None and "Target Career" in dataset.columns:
            count = int((dataset["Target Career"] == selected_career).sum())
            pct = count / len(dataset) * 100

            st.markdown(f"""
            <div class="card">
                <h4>Dataset Representation</h4>
                <div style="font-size:34px;font-weight:800;color:#111827;">
                    {count:,}
                </div>
                <div style="color:#64748b;">records labelled as this career</div>
                <div style="margin-top:12px;color:#475569;">
                    {pct:.2f}% of the provided dataset
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Dataset file not found, so dataset statistics are unavailable.")

    if dataset is not None and "Target Career" in dataset.columns:
        st.markdown("### 📊 Career Distribution in Dataset")
        counts = (
            dataset["Target Career"]
            .value_counts()
            .rename_axis("Career")
            .reset_index(name="Students")
        )
        st.bar_chart(counts.set_index("Career"))

# ====================== TAB 3 =================================

with tab3:
    st.markdown("## ℹ️ About CareerFit")

    st.markdown("""
    <div class="card">
        <h3>🎯 What is CareerFit?</h3>
        <p style="color:#475569;line-height:1.7;">
        CareerFit is an AI-powered career guidance prototype that maps a student's
        profile information to career categories using Natural Language Processing
        and Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="card">
            <h3>📝 1. Profile</h3>
            <p style="color:#64748b;">
            Skills, interests, favorite subjects, personality and career goals
            are collected as text.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <h3>🔤 2. TF-IDF</h3>
            <p style="color:#64748b;">
            The profile text is transformed into numerical TF-IDF features using
            the saved vectorizer from training.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="card">
            <h3>🤖 3. Prediction</h3>
            <p style="color:#64748b;">
            Logistic Regression calculates probabilities for the ten trained
            career classes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧠 Model Details")

    model_details = pd.DataFrame({
        "Component": [
            "Text Representation",
            "Vectorizer",
            "Classifier",
            "Career Classes",
            "Recommendation Output"
        ],
        "Configuration": [
            "Combined profile text",
            "TF-IDF, unigram + bigram",
            "Logistic Regression",
            "10",
            "Top 3 + full probability distribution"
        ]
    })

    st.dataframe(
        model_details,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="info-box">
        <b>Important:</b> CareerFit is an educational/project prototype.
        A model probability is not a guarantee of career success. Career decisions
        should also consider personal goals, real-world experience, opportunities,
        financial considerations and guidance from qualified people.
    </div>
    """, unsafe_allow_html=True)

# ---------------------- Footer --------------------------------

st.markdown("""
<div class="footer">
    CareerFit • AI-Powered Career Guidance Expert System<br>
    Built with Streamlit • TF-IDF • Logistic Regression
</div>
""", unsafe_allow_html=True)

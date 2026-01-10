import streamlit as st

st.set_page_config(
    page_title="Bank Customer Churn Prediction Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

/* ===== SIDEBAR (MAROON) ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #5B0F1B, #7A1526);
    border-right: 1px solid #E5E7EB;
}

/* Sidebar title */
section[data-testid="stSidebar"] h2 {
    color: #FDECEC;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #FDECEC;
}

/* Sidebar caption */
section[data-testid="stSidebar"] .stCaption {
    color: #F8D7DA;
}

/* ===== RADIO MENU ===== */
div[role="radiogroup"] label {
    padding: 0.6rem 0.75rem;
    border-radius: 10px;
    margin-bottom: 6px;
    font-weight: 500;
    transition: all 0.15s ease;
}

/* Hover */
div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.18);
}

/* Active */
div[role="radiogroup"] label[data-selected="true"] {
    background: #FFFFFF !important;
    color: #5B0F1B !important;
    font-weight: 700;
}

/* ===== MAIN CONTENT ===== */
h1, h2, h3, h4 {
    color: #5B0F1B;
}

p, span, li {
    color: #334155;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        #7A1526,
        transparent
    );
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

MENU = {
    "📘 About Dataset": "About Dataset",
    "📊 EDA Visualization": "EDA Visualization",
    "🧠 Model Development": "Machine Learning",
    "🧮 Prediction App": "Prediction App",
    "📩 Contact Me": "Contact Me",
}

with st.sidebar:
    st.markdown("## 🧭 Navigation")

    selected_label = st.radio(
        "Pilih halaman:",
        list(MENU.keys()),
        index=1
    )
    menu = MENU[selected_label]

    st.caption("🏦 Bank Customer Churn App")
    st.caption("Machine-Learning Case Study")

st.title("🏦 Bank Customer Churn Prediction Analysis")
st.caption("A Machine-Learning Case Study • 2026")

st.markdown("---")

if menu == "About Dataset":
    import about
    about.about_dataset()

elif menu == "EDA Visualization":
    import visualisasi
    visualisasi.chart()

elif menu == "Machine Learning":
    import machine_learning
    machine_learning.ml_model()

elif menu == "Prediction App":
    import prediction
    prediction.prediction_app()

elif menu == "Contact Me":
    import kontak
    kontak.contact_me()

st.markdown("---")
st.caption("© 2026 • Bank Customer Churn Prediction Analysis")

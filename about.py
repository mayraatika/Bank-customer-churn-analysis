import streamlit as st

def about_dataset():
    # ===== CSS lokal: hanya card yang perlu (tanpa section-card) =====
    st.markdown("""
    <style>
    .mini-card {
        background: #FFFFFF;
        padding: 0.95rem 1.05rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
    }
    .mini-title {
        color: #636E72;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .mini-value {
        font-size: 1.15rem;
        font-weight: 800;
        color: #2D3436;
        line-height: 1.25;
        margin: 0;
    }
    .mini-note {
        color: #636E72;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .rq-box {
        background: #FFFFFF;
        padding: 1.05rem 1.15rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        min-height: 140px;
        height: 100%;
        box-sizing: border-box;
    }
    .rq-title {
        font-weight: 800;
        margin-bottom: 0.35rem;
        color: #2D3436;
    }
    .rq-desc {
        color: #636E72;
        font-size: 0.95rem;
        line-height: 1.35;
    }

    /* Mengatur bullet list */
    .kpi-list ul { padding-left: 1.1rem; margin: 0.2rem 0 0 0; }
    .kpi-list li { margin: 0.35rem 0; }
    </style>
    """, unsafe_allow_html=True)

    # ===== SECTION 1: IMAGE + BACKGROUND =====
    col_img, col_text = st.columns([1, 1.25], gap="large")

    with col_img:
        st.image("bank_cust.jpg", use_container_width=True)
        st.caption("Source: Freepik")

    with col_text:
        st.subheader("Background")
        st.markdown(
            """
            <div style="text-align: justify;">
            Customer churn merupakan isu strategis dalam industri perbankan karena berpengaruh langsung
            terhadap stabilitas pendapatan dan efisiensi biaya operasional. Kehilangan nasabah tidak hanya
            menurunkan pendapatan, tetapi juga meningkatkan biaya akuisisi pelanggan baru.<br><br>

            Dengan memahami karakteristik serta pola perilaku nasabah, perusahaan dapat mengidentifikasi
            pelanggan berisiko churn lebih awal dan mengambil langkah preventif secara proaktif.<br><br>

            Dataset ini digunakan untuk menganalisis dan memprediksi potensi customer churn berdasarkan
            faktor demografis dan perilaku nasabah. Penerapan pendekatan machine learning memungkinkan
            perusahaan menyusun strategi retensi yang lebih tepat sasaran, mengoptimalkan nilai nasabah,
            serta mendukung pengambilan keputusan berbasis data guna menjaga pertumbuhan bisnis yang berkelanjutan.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ===== SECTION 2: KPI FOCUS (3 CARDS) =====
    k1, k2, k3 = st.columns(3, gap="medium")

    with k1:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">Primary KPI</div>
                <div class="mini-value">Churn Rate</div>
                <div class="mini-note">Deteksi risiko lebih awal</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">Retention Focus</div>
                <div class="mini-value">Retention Rate</div>
                <div class="mini-note">Strategi retensi tepat sasaran</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">Value & Efficiency</div>
                <div class="mini-value">CLV & CAC</div>
                <div class="mini-note">Optimasi nilai & biaya akuisisi</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ===== SECTION 3: OBJECTIVE =====
    st.subheader("Objective")
    o1, o2 = st.columns([1.35, 1], gap="large")

    with o1:
        st.markdown(
            """
            <div style="text-align: justify;">
            Mengukur dan menurunkan tingkat customer churn melalui analisis berbasis data dengan fokus
            pada peningkatan retensi nasabah, optimalisasi nilai pelanggan (customer lifetime value),
            serta efisiensi biaya akuisisi. Pendekatan ini bertujuan mendukung keputusan strategis yang
            berdampak langsung pada stabilitas pendapatan dan pertumbuhan bisnis.
            </div>
            """,
            unsafe_allow_html=True
        )

    with o2:
        st.markdown("**KPI Targets:**")
        st.markdown(
            """
            <div class="kpi-list">
            <ul>
              <li>📉 Turunkan <b>Churn Rate</b></li>
              <li>📈 Naikkan <b>Retention Rate</b></li>
              <li>💰 Tingkatkan <b>CLV</b></li>
              <li>💸 Tekan <b>CAC</b></li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ===== SECTION 4: RESEARCH QUESTION =====
    st.subheader("Research Question")

    r1, r2, r3 = st.columns(3, gap="medium")

    with r1:
        st.markdown(
            """
            <div class="rq-box">
                <div class="rq-title">🔍 Faktor Utama</div>
                <div class="rq-desc">
                    Faktor apa saja yang paling berpengaruh terhadap terjadinya customer churn?
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            """
            <div class="rq-box">
                <div class="rq-title">👥 Profil Nasabah</div>
                <div class="rq-desc">
                    Bagaimana karakteristik nasabah yang memiliki risiko churn tinggi?
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r3:
        st.markdown(
            """
            <div class="rq-box">
                <div class="rq-title">📊 Variabel Kunci</div>
                <div class="rq-desc">
                    Variabel apa yang paling signifikan dalam memprediksi customer churn?
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

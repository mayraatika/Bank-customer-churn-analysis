import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components


# =========================
# PALETTE / THEME
# =========================
PALETTE = {
    "cream": "#FFFDE1",
    "yellow": "#FBE580",
    "green": "#93BD57",
    "red": "#980404",
    "text": "#1f2937",
    "muted": "rgba(31,41,55,0.70)",
    "card": "rgba(255,255,255,0.80)"
}


# =========================
# RISK LOGIC
# =========================
def risk_level(p: float) -> str:
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    return "High"


def risk_badge(level: str) -> str:
    if level == "Low":
        return f"""
        <div class="badge" style="background:{PALETTE['green']}22;border:1px solid {PALETTE['green']}55;color:#1f3b08;">
            🟢 Low Risk
        </div>
        """
    if level == "Medium":
        return f"""
        <div class="badge" style="background:{PALETTE['yellow']}55;border:1px solid {PALETTE['yellow']};color:#5a4a00;">
            🟡 Medium Risk
        </div>
        """
    return f"""
    <div class="badge" style="background:{PALETTE['red']}22;border:1px solid {PALETTE['red']}55;color:{PALETTE['red']};">
        🔴 High Risk
    </div>
    """


def churn_badge(is_alert: bool, thr: float) -> str:
    if is_alert:
        return f"""
        <div class="badge" style="background:{PALETTE['red']}22;border:1px solid {PALETTE['red']}55;color:{PALETTE['red']};">
            ⚠️ Churn Alert (p ≥ {thr:.2f})
        </div>
        """
    return f"""
    <div class="badge" style="background:{PALETTE['green']}22;border:1px solid {PALETTE['green']}55;color:#1f3b08;">
        ✅ No Alert (p < {thr:.2f})
    </div>
    """


def recommended_actions(level: str) -> list[str]:
    if level == "High":
        return [
            "Prioritaskan outreach 1:1 (telepon/relationship manager) dalam 24–48 jam.",
            "Tawarkan incentive relevan (fee waiver / upgrade / loyalty points) — batasi durasi agar efisien.",
            "Review produk yang dimiliki: dorong bundling yang meningkatkan switching cost.",
            "Pastikan service recovery jika ada indikasi dissatisfaction (keluhan/low engagement).",
        ]
    if level == "Medium":
        return [
            "Masukkan ke campaign retensi bertarget (email/push) dengan penawaran ringan.",
            "Ajak aktivasi fitur/produk yang meningkatkan engagement (autodebit, saving goal, dsb).",
            "Monitor 2–4 minggu: jika probability naik → eskalasi ke high-risk flow.",
        ]
    return [
        "Lakukan nurturing (edukasi produk / reminder manfaat) tanpa diskon agresif.",
        "Pantau perubahan perilaku (aktivitas & jumlah produk).",
        "Jadikan baseline untuk segmentasi campaign berikutnya.",
    ]


# =========================
# UI CSS
# =========================
def _inject_css():
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }}

        .pred-wrap {{
            padding: 1.2rem 1.25rem;
            border-radius: 18px;
            background: {PALETTE['card']};
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 10px 22px rgba(0,0,0,0.06);
        }}

        .title {{
            font-weight: 900;
            font-size: 1.35rem;
            color: {PALETTE['text']};
            margin-bottom: 0.25rem;
        }}

        .sub {{
            color: rgba(31,41,55,0.72);
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 0.5rem;
            margin-top: 0.25rem;
        }}

        .kpi {{
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(0,0,0,0.05);
        }}

        .kpi-label {{
            color: rgba(31,41,55,0.72);
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }}

        .kpi-value {{
            font-weight: 900;
            font-size: 1.55rem;
            color: {PALETTE['text']};
            line-height: 1.1;
        }}

        .mini {{
            color: rgba(31,41,55,0.72);
            font-size: 0.85rem;
            margin-top: 0.35rem;
        }}

        .list {{
            margin-top: 0.6rem;
            padding-left: 1.1rem;
            color: rgba(31,41,55,0.85);
        }}

        .hint {{
            background: {PALETTE['cream']};
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================
# LOAD ARTIFACTS (MODEL + SCALER!)
# =========================
def _load_artifacts():
    model = joblib.load("model_churn.pkl")
    feature_names = joblib.load("model_features.pkl")     # list columns after encoding
    numeric_cols = joblib.load("numeric_columns.pkl")     # numeric columns scaled in training
    best_thr = float(joblib.load("best_threshold.pkl"))
    scaler = joblib.load("scaler.pkl")                    # ✅ scaler saved from training
    return model, feature_names, numeric_cols, best_thr, scaler


def prediction_app():
    _inject_css()

    st.markdown(
        """
        <div class="pred-wrap">
            <div class="title">🏦 Churn Prediction</div>
            <div class="sub">Masukkan fitur nasabah → sistem menghitung probabilitas churn, risk level, dan rekomendasi aksi.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load model artifacts
    try:
        model, model_features, numeric_cols, best_thr, scaler = _load_artifacts()
    except Exception as e:
        st.error(
            "Model belum tersedia. Pastikan kamu sudah klik **Save Model** di tab Machine Learning "
            "dan file berikut ada di folder yang sama:\n\n"
            "- model_churn.pkl\n- model_features.pkl\n- numeric_columns.pkl\n- best_threshold.pkl\n- scaler.pkl\n\n"
            f"Detail error: {e}"
        )
        st.stop()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # =========================
    # Preset (biar gampang test Low Risk / High Risk)
    # =========================
    presets = {
        "Low Risk (recommended)": {
            "CreditScore": 820, "Geography": "France", "Gender": "Female", "Age": 27,
            "Tenure": 9, "Balance": 0.0, "NumOfProducts": 3, "HasCrCard": 1,
            "IsActiveMember": 1, "EstimatedSalary": 70000.0
        },
        "High Risk (example)": {
            "CreditScore": 620, "Geography": "Germany", "Gender": "Male", "Age": 45,
            "Tenure": 2, "Balance": 90000.0, "NumOfProducts": 1, "HasCrCard": 1,
            "IsActiveMember": 0, "EstimatedSalary": 80000.0
        }
    }

    preset_choice = st.selectbox("🎛️ Quick Preset", list(presets.keys()), index=0)
    preset = presets[preset_choice]

    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown(
            "<div class='pred-wrap'><div class='title'>🧾 Customer Profile</div>"
            "<div class='sub'>Input data nasabah</div></div>",
            unsafe_allow_html=True
        )

        with st.form("pred_form", clear_on_submit=False):
            c1, c2 = st.columns(2, gap="medium")

            with c1:
                credit_score = st.number_input("CreditScore", 300, 900, int(preset["CreditScore"]), 1)
                age = st.number_input("Age", 18, 100, int(preset["Age"]), 1)
                tenure = st.number_input("Tenure (years)", 0, 10, int(preset["Tenure"]), 1)
                num_products = st.number_input("NumOfProducts", 1, 4, int(preset["NumOfProducts"]), 1)
                has_crcard = st.selectbox("HasCrCard", [0, 1], index=int(preset["HasCrCard"]))

            with c2:
                geography = st.selectbox("Geography", ["France", "Germany", "Spain"],
                                        index=["France","Germany","Spain"].index(preset["Geography"]))
                gender = st.selectbox("Gender", ["Male", "Female"],
                                      index=["Male","Female"].index(preset["Gender"]))
                balance = st.number_input("Balance", min_value=0.0, value=float(preset["Balance"]), step=500.0)
                is_active = st.selectbox("IsActiveMember", [0, 1], index=int(preset["IsActiveMember"]))
                est_salary = st.number_input("EstimatedSalary", min_value=0.0, value=float(preset["EstimatedSalary"]), step=500.0)

            submitted = st.form_submit_button("🔮 Predict Churn")

    # =========================
    # Predict
    # =========================
    if submitted:
        raw = pd.DataFrame([{
            "CreditScore": int(credit_score),
            "Geography": geography,
            "Gender": gender,
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(num_products),
            "HasCrCard": int(has_crcard),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": float(est_salary),
        }])

        # One-hot encoding to match training
        X_in = pd.get_dummies(raw, columns=["Geography", "Gender"], drop_first=True)

        # Ensure same columns exist (reindex)
        X_in = X_in.reindex(columns=model_features, fill_value=0)

        # ✅ Apply saved scaler to numeric cols (consistent with training)
        X_in[numeric_cols] = scaler.transform(X_in[numeric_cols])

        proba = float(model.predict_proba(X_in)[0, 1])
        level = risk_level(proba)
        is_alert = proba >= best_thr

        with right:
            st.markdown(
                "<div class='pred-wrap'><div class='title'>📈 Prediction Result</div>"
                "<div class='sub'>Output model + recommended next steps</div></div>",
                unsafe_allow_html=True
            )

            k1, k2 = st.columns(2, gap="medium")
            with k1:
                st.markdown(
                    f"""
                    <div class="kpi">
                        <div class="kpi-label">Churn Probability</div>
                        <div class="kpi-value">{proba*100:.1f}%</div>
                        <div class="mini">Model: Random Forest</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with k2:
                st.markdown(
                    f"""
                    <div class="kpi">
                        <div class="kpi-label">Decision</div>
                        <div class="kpi-value">{'ALERT' if is_alert else 'OK'}</div>
                        <div class="mini">Threshold tuned: {best_thr:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(risk_badge(level), unsafe_allow_html=True)
            st.markdown(churn_badge(is_alert, best_thr), unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='pred-wrap'><div class='title'>✅ Recommended Actions</div>"
                "<div class='sub'>Otomatis berdasarkan risk level</div></div>",
                unsafe_allow_html=True
            )

            actions = recommended_actions(level)
            st.markdown("<ul class='list'>" + "".join([f"<li>{a}</li>" for a in actions]) + "</ul>", unsafe_allow_html=True)

            with st.expander("Lihat ringkasan input", expanded=False):
                st.dataframe(raw, use_container_width=True)

    else:
        with right:
            st.markdown(
                f"""
                <div class="pred-wrap">
                    <div class="title">✨ Tips</div>
                    <div class="sub">Agar hasil prediksi lebih berguna untuk aksi bisnis</div>
                    <ul class="list">
                        <li>Gunakan preset <b>Low Risk</b> untuk memastikan pipeline prediksi berjalan.</li>
                        <li>Threshold tuned (<b>{best_thr:.2f}</b>) dipakai sebagai early warning churn.</li>
                        <li>Jika hasil masih sering ALERT, cek: <b>IsActiveMember</b>, <b>NumOfProducts</b>, dan <b>Balance</b>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hint">
            <b>Catatan:</b> Prediksi ini adalah estimasi berbasis data historis. Untuk operasional,
            prioritaskan intervensi pada <b>High Risk</b>, lalu evaluasi dampak campaign untuk iterasi threshold.
        </div>
        """,
        unsafe_allow_html=True
    )

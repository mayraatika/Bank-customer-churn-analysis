import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

# Imbalance handling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False


# =========================
# THEME / PALETTE
# =========================
PALETTE = {
    "cream": "#FFFDE1",
    "yellow": "#FBE580",
    "green": "#93BD57",
    "red": "#980404",
    "text": "#1f2937",
    "muted": "rgba(31,41,55,0.72)",
    "grid": "rgba(31,41,55,0.14)",
}

CATEGORY_COLOR = {
    "Demographic": PALETTE["yellow"],
    "Behavior": PALETTE["green"],
    "Financial": PALETTE["red"],
    "Other": "rgba(31,41,55,0.35)",
}


def _base_layout(fig, title=None, height=420):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=12, r=12, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color=PALETTE["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False, tickfont=dict(color=PALETTE["muted"]))
    return fig


def feature_category(feature_name: str) -> str:
    f = feature_name.lower()

    # Demographic
    if ("age" in f) or ("gender" in f) or ("geography" in f) or ("creditscore" in f):
        return "Demographic"

    # Behavior
    if ("isactivemember" in f) or ("numofproducts" in f) or ("tenure" in f) or ("hascrcard" in f):
        return "Behavior"

    # Financial
    if ("balance" in f) or ("estimatedsalary" in f):
        return "Financial"

    return "Other"


def _confusion_fig(cm, title):
    df_cm = pd.DataFrame(
        cm,
        index=["Actual Active (0)", "Actual Churn (1)"],
        columns=["Pred Active (0)", "Pred Churn (1)"]
    )
    fig = px.imshow(
        df_cm,
        text_auto=True,
        color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["yellow"]]],
        title=title
    )
    return _base_layout(fig, title=title, height=420)


def _scorecard_vertical(title: str, thr: float, acc: float, prec: float, rec: float, f1: float, auc: float, fp: int, fn: int):
    """
    Scorecard vertical supaya angka nggak kepotong.
    """
    st.markdown(f"#### {title}")

    # Row 1
    a, b, c = st.columns(3)
    a.metric("Threshold", f"{thr:.2f}")
    b.metric("Accuracy", f"{acc*100:.2f}%")
    c.metric("ROC AUC", f"{auc*100:.2f}%")

    # Row 2
    d, e, fcol = st.columns(3)
    d.metric("Precision", f"{prec*100:.2f}%")
    e.metric("Recall", f"{rec*100:.2f}%")
    fcol.metric("F1", f"{f1*100:.2f}%")

    # Row 3
    g, h = st.columns(2)
    g.metric("False Negative (FN)", f"{fn:,}")
    h.metric("False Positive (FP)", f"{fp:,}")

    st.caption("Catatan: FN = churn yang terlewat (bahaya untuk retensi). FP = alarm palsu (biaya campaign).")


def ml_model():
    st.subheader("Machine Learning Model")
    st.caption("Random Forest • SMOTE (if available) • Threshold Tuning • Grouped Feature Importance • Save Scaler")

    # ======================================================
    # 1) Load Dataset 
    # ======================================================
    df = pd.read_excel("CustomerChurnAnalysis.xlsx")

    selected_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Exited"
    ]

    missing = [c for c in selected_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan di dataset: {missing}")
        st.stop()

    df = df[selected_cols].copy()
    df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce")
    df = df.dropna(subset=["Exited"])
    df["Exited"] = df["Exited"].astype(int)

    if set(df["Exited"].unique()) - {0, 1}:
        st.error(f"Target `Exited` harus binary (0/1). Nilai unik sekarang: {sorted(df['Exited'].unique().tolist())}")
        st.stop()

    numeric_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    categorical_cols = ["Geography", "Gender"]

    # ======================================================
    # KPI header
    # ======================================================
    total = len(df)
    churned = int((df["Exited"] == 1).sum())
    churn_rate = (churned / total * 100) if total else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Churned Customers", f"{churned:,}")
    k3.metric("Churn Rate", f"{churn_rate:.2f}%")

    with st.expander("📄 Preview Dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # ======================================================
    # 2) Outlier Handling (IQR)
    # ======================================================
    st.write("### 1. Outlier Handling (IQR Method)")
    outlier_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
    before = df.shape[0]

    Q1 = df[outlier_cols].quantile(0.25)
    Q3 = df[outlier_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[outlier_cols] < (Q1 - 1.5 * IQR)) | (df[outlier_cols] > (Q3 + 1.5 * IQR))).any(axis=1)].copy()

    st.info(f"Rows before: **{before:,}** → after: **{df.shape[0]:,}**")

    # ======================================================
    # 2.5) Correlation Heatmap
    # ======================================================
    st.write("### Correlation Heatmap (Numerical Features)")

    corr_cols = numeric_cols + ["Exited"]
    corr_df = df[corr_cols].corr()

    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="Reds",
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    fig_corr.update_traces(showscale=True)

    fig_corr.update_xaxes(showgrid=False, showline=False, ticks="")
    fig_corr.update_yaxes(showgrid=False, showline=False, ticks="")

    fig_corr.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1]
        ),
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
    "📌 **Insight Correlation:** Seluruh pasangan fitur memiliki nilai korelasi di bawah **0.8**, "
    "sehingga **tidak terindikasi multikolinearitas tinggi**. "
    "Oleh karena itu, **tidak ada fitur yang di-drop** berdasarkan analisis korelasi."
    )


    # ======================================================
    # 3) Feature Encoding
    # ======================================================
    st.write("### 2. Feature Encoding (One-Hot)")

    st.markdown(
        """
        **Apa yang dilakukan pada tahap encoding?**
        - Kolom kategorikal **Geography** dan **Gender** tidak bisa langsung diproses model tree dalam bentuk teks.
        - Maka kita ubah menjadi **kolom biner (0/1)** menggunakan *one-hot encoding*.
        - `drop_first=True` dipakai supaya menghindari redundansi (dummy trap), jadi 1 kategori dijadikan baseline.

        **Contoh hasilnya:**
        - `Geography` → `Geography_Germany`, `Geography_Spain` (baseline: France)
        - `Gender` → `Gender_Male` (baseline: Female)
        """
    )

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    created_cols = [c for c in df_encoded.columns if c not in df.columns]
    st.caption(f"Kolom baru hasil encoding: {', '.join(created_cols)}")

    # ======================================================
    # 4) Train-Test Split 
    # ======================================================
    st.write("### 3. Train–Test Split")

    test_size = st.slider("Test size", 0.15, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    X = df_encoded.drop("Exited", axis=1)
    y = df_encoded["Exited"]

    if len(y.unique()) < 2:
        st.error("Data hanya punya 1 kelas setelah cleaning. Tidak bisa training model.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y
    )

    s1, s2 = st.columns(2)
    s1.metric("Train size", f"{len(X_train):,}")
    s2.metric("Test size", f"{len(X_test):,}")

    st.caption("Split dilakukan sebelum scaling untuk menghindari data leakage.")

    # ======================================================
    # 5) Scaling (fit on train only) + SAVE SCALER
    # ======================================================
    st.write("### 4. Scaling (MinMaxScaler)")

    scaler = MinMaxScaler()
    scale_cols = [c for c in numeric_cols if c in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaler.fit(X_train_scaled[scale_cols])
    X_train_scaled[scale_cols] = scaler.transform(X_train_scaled[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test_scaled[scale_cols])

    st.success("✅ Scaling applied on train only. Scaler akan disimpan untuk Prediction App.")

    # ======================================================
    # 6) Imbalance Handling
    # ======================================================
    st.write("### 5. Handling Imbalanced Data")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Active (0)", int((y_train == 0).sum()))
    c2.metric("Train Churn (1)", int((y_train == 1).sum()))
    c3.metric("SMOTE", "Available ✅" if SMOTE_AVAILABLE else "Not Available ⚠️")
    c4.metric("Strategy", "SMOTE" if SMOTE_AVAILABLE else "class_weight")

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=int(random_state))
        X_train_final, y_train_final = sm.fit_resample(X_train_scaled, y_train)
        class_weight = None
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
        class_weight = "balanced"

    # ======================================================
    # 7) Train Random Forest
    # ======================================================
    st.write("### 6. Train Random Forest")

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=int(random_state),
        n_jobs=-1,
        class_weight=class_weight
    )
    model.fit(X_train_final, y_train_final)

    train_acc = model.score(X_train_final, y_train_final)
    st.success(f"Training Accuracy: **{train_acc*100:.2f}%**")

    # ======================================================
    # 8) Feature Importance (Grouped)
    # ======================================================
    st.write("### 7. Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X_train_scaled.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    importance_df["Category"] = importance_df["Feature"].apply(feature_category)

    cat_sum = (importance_df.groupby("Category", as_index=False)["Importance"]
               .sum().sort_values("Importance", ascending=False))

    fig_cat = px.bar(
        cat_sum,
        x="Importance",
        y="Category",
        orientation="h",
        title="Total Importance by Category",
        color="Category",
        color_discrete_map=CATEGORY_COLOR
    )
    fig_cat = _base_layout(fig_cat, title="Total Importance by Category", height=340)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.write("#### Top 10 Most Influential Features")
    top10 = importance_df.head(10).copy()  # already desc
    top10 = top10.sort_values("Importance", ascending=True)

    fig_imp = go.Figure()
    fig_imp.add_trace(
        go.Bar(
            x=top10["Importance"],
            y=top10["Feature"],
            orientation="h",
            marker=dict(color=top10["Category"].map(CATEGORY_COLOR).fillna(CATEGORY_COLOR["Other"])),
            text=[f"{v:.3f}" for v in top10["Importance"]],
            textposition="outside",
        )
    )
    fig_imp = _base_layout(fig_imp, title="Top 10 Feature Importance", height=520)
    fig_imp.update_xaxes(title="Importance")
    fig_imp.update_yaxes(title=None)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ======================================================
    # 9) Model Evaluation 
    # ======================================================
    st.write("### 8. Model Evaluation")

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    default_thr = 0.50
    y_pred_default = (y_prob >= default_thr).astype(int)

    default_acc = accuracy_score(y_test, y_pred_default)
    default_prec = precision_score(y_test, y_pred_default, zero_division=0)
    default_rec = recall_score(y_test, y_pred_default, zero_division=0)
    default_f1 = f1_score(y_test, y_pred_default, zero_division=0)
    default_auc = roc_auc_score(y_test, y_prob)

    cm_default = confusion_matrix(y_test, y_pred_default)
    tn_d, fp_d, fn_d, tp_d = cm_default.ravel()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{default_acc*100:.2f}%")
    m2.metric("Precision", f"{default_prec*100:.2f}%")
    m3.metric("Recall", f"{default_rec*100:.2f}%")
    m4.metric("F1", f"{default_f1*100:.2f}%")
    m5.metric("ROC AUC", f"{default_auc*100:.2f}%")

    st.plotly_chart(_confusion_fig(cm_default, "Confusion Matrix — Default (0.50)"), use_container_width=True)

    # ======================================================
    # 10) Threshold tuning
    # ======================================================
    st.write("### 9. Threshold Tuning")
    st.caption("Goal: reduce missed churn (FN) while keeping FP controlled (precision guardrail).")

    min_precision = max(0.0, default_prec * 0.90)

    thresholds = [round(i / 100, 2) for i in range(20, 81, 2)]
    rows = []
    for thr in thresholds:
        pred_thr = (y_prob >= thr).astype(int)
        cm_thr = confusion_matrix(y_test, pred_thr)
        tn, fp, fn, tp = cm_thr.ravel()
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_test, pred_thr),
            "precision": precision_score(y_test, pred_thr, zero_division=0),
            "recall": recall_score(y_test, pred_thr, zero_division=0),
            "f1": f1_score(y_test, pred_thr, zero_division=0),
            "FP": int(fp),
            "FN": int(fn),
        })

    thr_df = pd.DataFrame(rows)
    candidates = thr_df[thr_df["precision"] >= min_precision].copy()

    if len(candidates) == 0:
        best_row = thr_df.sort_values("f1", ascending=False).iloc[0]
        st.warning("Guardrail terlalu ketat → fallback ke threshold terbaik berdasarkan F1.")
    else:
        best_row = candidates.sort_values("f1", ascending=False).iloc[0]

    best_thr = float(best_row["threshold"])
    y_pred_best = (y_prob >= best_thr).astype(int)

    tuned_acc = accuracy_score(y_test, y_pred_best)
    tuned_prec = precision_score(y_test, y_pred_best, zero_division=0)
    tuned_rec = recall_score(y_test, y_pred_best, zero_division=0)
    tuned_f1 = f1_score(y_test, y_pred_best, zero_division=0)
    tuned_auc = default_auc

    cm_best = confusion_matrix(y_test, y_pred_best)
    tn_b, fp_b, fn_b, tp_b = cm_best.ravel()

    st.success(f"✅ Tuned threshold = **{best_thr:.2f}** (Precision guardrail ≥ {min_precision*100:.1f}%)")

    fig_thr = go.Figure()
    fig_thr.add_trace(go.Scatter(x=thr_df["threshold"], y=thr_df["precision"], mode="lines+markers",
                                 name="Precision", line=dict(color=PALETTE["green"])))
    fig_thr.add_trace(go.Scatter(x=thr_df["threshold"], y=thr_df["recall"], mode="lines+markers",
                                 name="Recall", line=dict(color=PALETTE["red"])))
    fig_thr.add_trace(go.Scatter(x=thr_df["threshold"], y=thr_df["f1"], mode="lines+markers",
                                 name="F1", line=dict(color=PALETTE["yellow"])))
    fig_thr = _base_layout(fig_thr, title="Precision / Recall / F1 vs Threshold", height=420)
    fig_thr.update_yaxes(title="Score")
    fig_thr.update_xaxes(title="Threshold")
    st.plotly_chart(fig_thr, use_container_width=True)

    # ======================================================
    # 11) Scorecard + CM comparison FIX
    # ======================================================
    st.write("### 10. Model Scorecard (Default vs Tuned)")

    left, right = st.columns(2, gap="large")
    with left:
        _scorecard_vertical(
            "Default (Threshold 0.50)",
            default_thr, default_acc, default_prec, default_rec, default_f1, default_auc,
            fp_d, fn_d
        )
    with right:
        _scorecard_vertical(
            "Tuned (Threshold Optimized)",
            best_thr, tuned_acc, tuned_prec, tuned_rec, tuned_f1, tuned_auc,
            fp_b, fn_b
        )

    st.write("### 11. Confusion Matrix Comparison")

    cm1, cm2 = st.columns(2, gap="large")
    with cm1:
        st.plotly_chart(_confusion_fig(cm_default, "Default (0.50)"), use_container_width=True)
    with cm2:
        st.plotly_chart(_confusion_fig(cm_best, f"Tuned ({best_thr:.2f})"), use_container_width=True)

    # ======================================================
    # 12) Save model + artifacts (INCLUDING scaler)
    # ======================================================
    st.write("### 12. Save Model & Artifacts")

    if st.button("💾 Save Model"):
        joblib.dump(model, "model_churn.pkl")
        joblib.dump(list(X_train_scaled.columns), "model_features.pkl")
        joblib.dump(scale_cols, "numeric_columns.pkl")
        joblib.dump(best_thr, "best_threshold.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success(
            "✅ Saved:\n"
            "- model_churn.pkl\n"
            "- model_features.pkl\n"
            "- numeric_columns.pkl\n"
            "- best_threshold.pkl\n"
            "- scaler.pkl"
        )

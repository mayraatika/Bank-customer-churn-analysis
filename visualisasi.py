import streamlit as st
import pandas as pd
import plotly.express as px


PALETTE = {
    "cream": "#FFFDE1",
    "yellow": "#FBE580",
    "green": "#93BD57",
    "red": "#980404",
    "text": "#1f2937",
    "muted": "rgba(31,41,55,0.65)",
    "grid": "rgba(31,41,55,0.12)",
}


def _pct(x) -> float:
    return float(x) * 100.0


def _safe_mean(series):
    try:
        return float(series.mean())
    except Exception:
        return None


def _format_money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def _base_layout(fig, title=None):
    fig.update_layout(
        title=title,
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(color=PALETTE["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        showline=False,
        tickfont=dict(color=PALETTE["muted"]),
    )
    return fig


def _soft_bar(fig, bar_colors=None):
    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.65)")),
        opacity=0.92,
        cliponaxis=False,
        textposition="outside",
    )
    if bar_colors is not None:
        fig.update_traces(marker_color=bar_colors)
    return fig


def _top_churn_by_group(df_in, group_col):
    if group_col not in df_in.columns or df_in.empty:
        return None
    tmp = df_in.groupby(group_col)["Exited"].mean()
    if tmp.empty:
        return None
    g = tmp.idxmax()
    return g, float(tmp.loc[g])


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def chart():
    st.subheader("EDA Visualization")
    st.caption("Exploratory Data Analysis for Bank Customer Churn")

    df = pd.read_excel("CustomerChurnAnalysis.xlsx")

    if "Exited" not in df.columns:
        st.error("Kolom target `Exited` tidak ditemukan.")
        st.stop()

    with st.expander("🔎 Filter Data", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            geo = "All"
            if "Geography" in df.columns:
                geo = st.selectbox(
                    "Geography",
                    ["All"] + sorted(df["Geography"].dropna().astype(str).unique().tolist()),
                )

        with c2:
            gender = "All"
            if "Gender" in df.columns:
                gender = st.selectbox(
                    "Gender",
                    ["All"] + sorted(df["Gender"].dropna().astype(str).unique().tolist()),
                )

        with c3:
            exited_filter = st.selectbox("Exited", ["All", 0, 1])

    dff = df.copy()
    if geo != "All" and "Geography" in dff.columns:
        dff = dff[dff["Geography"].astype(str) == str(geo)]
    if gender != "All" and "Gender" in dff.columns:
        dff = dff[dff["Gender"].astype(str) == str(gender)]
    if exited_filter != "All":
        dff = dff[dff["Exited"] == exited_filter]

    total = len(dff)
    churned = int((dff["Exited"] == 1).sum())
    active = int((dff["Exited"] == 0).sum())
    churn_rate = (churned / total * 100) if total else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Churned Customers", f"{churned:,}")
    k3.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Demographics", "Behavior", "Financial"])

    with tab1:
        st.markdown("### Customer Status Overview")

        if total == 0:
            st.warning("Tidak ada data pada filter saat ini.")
        else:
            dist = pd.DataFrame({"Label": ["Active", "Churn"], "Count": [active, churned]})
            fig = px.bar(dist, x="Label", y="Count", text="Count", title="Customer Distribution (Active vs Churn)")
            fig = _soft_bar(fig, [PALETTE["green"], PALETTE["red"]])
            fig = _base_layout(fig, title="Customer Distribution (Active vs Churn)")
            fig.update_yaxes(title="Count")
            fig.update_xaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

            if churn_rate < 15:
                st.success(
                    f"📌 **Overview Insight:** Churn berada di level **{churn_rate:.2f}%** (relatif sehat). "
                    "Prioritas utama adalah mempertahankan kualitas pengalaman nasabah dan menjaga segmen bernilai tinggi."
                )
            elif churn_rate < 30:
                st.warning(
                    f"📌 **Overview Insight:** Churn sebesar **{churn_rate:.2f}%** menunjukkan risiko yang mulai berdampak. "
                    "Intervensi retensi berbasis segmen (non-aktif/wilayah/produk) berpotensi memberi dampak cepat."
                )
            else:
                st.error(
                    f"📌 **Overview Insight:** Churn tinggi (**{churn_rate:.2f}%**) adalah sinyal kritis. "
                    "Diperlukan strategi retensi agresif, tersegmentasi, dan berbasis prioritas risiko."
                )

    with tab2:
        st.markdown("### Demographics (Who churns?)")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            if "Geography" in dff.columns and total:
                geo_churn = dff.groupby("Geography")["Exited"].mean().reset_index()
                geo_churn["ChurnRate"] = geo_churn["Exited"] * 100

                fig = px.pie(
                    geo_churn,
                    names="Geography",
                    values="ChurnRate",
                    hole=0.55,
                    title="Churn Rate by Geography (%)",
                    color_discrete_sequence=[PALETTE["green"], PALETTE["yellow"], PALETTE["red"], "#6BAED6"],
                )
                fig.update_traces(textposition="outside", textinfo="percent+label")
                fig = _base_layout(fig, title="Churn Rate by Geography (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom `Geography` tidak tersedia / data kosong.")

        with c2:
            if "Gender" in dff.columns and total:
                gen_churn = dff.groupby("Gender")["Exited"].mean().reset_index()
                gen_churn["ChurnRate"] = gen_churn["Exited"] * 100

                fig = px.pie(
                    gen_churn,
                    names="Gender",
                    values="ChurnRate",
                    hole=0.55,
                    title="Churn Rate by Gender (%)",
                    color_discrete_sequence=[PALETTE["green"], PALETTE["red"]],
                )
                fig.update_traces(textposition="outside", textinfo="percent+label")
                fig = _base_layout(fig, title="Churn Rate by Gender (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom `Gender` tidak tersedia / data kosong.")

        if "Age" in dff.columns and total:
            tmp = dff.copy()
            tmp["Status"] = tmp["Exited"].map({0: "Active", 1: "Churn"}).astype(str)

            fig = px.box(
                tmp,
                x="Status",
                y="Age",
                color="Status",
                title="Age Distribution by Status",
                color_discrete_map={"Active": PALETTE["green"], "Churn": PALETTE["red"]},
            )
            fig = _base_layout(fig, title="Age Distribution by Status")
            fig.update_xaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📌 Insight (Demographics):**")
        dem_ins = []

        top_geo = _top_churn_by_group(dff, "Geography")
        if top_geo:
            dem_ins.append(
                f"🌍 **Geographic Risk:** Wilayah **{top_geo[0]}** memiliki churn tertinggi "
                f"(**{_pct(top_geo[1]):.1f}%**). Hal ini bisa mengindikasikan faktor lokal: kompetisi, kualitas layanan, atau mismatch produk."
            )

        if "Gender" in dff.columns and total:
            g_rate = dff.groupby("Gender")["Exited"].mean()
            if len(g_rate) >= 2:
                max_g = g_rate.idxmax()
                dem_ins.append(
                    f"👥 **Gender Pattern:** Segmen **{max_g}** menunjukkan churn lebih tinggi "
                    f"(**{_pct(g_rate[max_g]):.1f}%**). Hal ini dapat menjadi dasar segmentasi komunikasi dan penawaran benefit."
                )

        if "Age" in dff.columns and total:
            churn_age = _safe_mean(dff.loc[dff["Exited"] == 1, "Age"])
            active_age = _safe_mean(dff.loc[dff["Exited"] == 0, "Age"])
            if churn_age is not None and active_age is not None:
                if churn_age > active_age:
                    dem_ins.append(
                        f"👤 **Age Trend:** Nasabah churn cenderung lebih tua "
                        f"(avg **{churn_age:.1f} tahun**) vs aktif (**{active_age:.1f} tahun**). "
                        "Implikasi: butuh pendekatan layanan & edukasi produk yang lebih sesuai segmen senior."
                    )
                else:
                    dem_ins.append(
                        f"👤 **Age Trend:** Perbedaan usia tidak terlalu kuat (avg churn **{churn_age:.1f}** vs aktif **{active_age:.1f}**). "
                        "Fokus retensi kemungkinan lebih kuat pada perilaku/produk."
                    )

        if dem_ins:
            for i in dem_ins:
                st.info(i)
        else:
            st.caption("Tidak ditemukan pola demografis yang cukup kuat pada filter saat ini.")

    with tab3:
        st.markdown("### Behavior (Why churns?)")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            if "IsActiveMember" in dff.columns and total:
                act = dff.groupby("IsActiveMember")["Exited"].mean().reset_index()
                act["ChurnRate"] = act["Exited"] * 100
                act["Label"] = act["IsActiveMember"].map({0: "Not Active", 1: "Active"}).astype(str)

                fig = px.bar(act, x="Label", y="ChurnRate", text=act["ChurnRate"].round(2), title="Churn Rate by Activity (%)")
                fig = _soft_bar(fig, PALETTE["yellow"])
                fig = _base_layout(fig, title="Churn Rate by Activity (%)")
                fig.update_yaxes(title="Churn Rate (%)")
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom `IsActiveMember` tidak tersedia / data kosong.")

        with c2:
            if "NumOfProducts" in dff.columns and total:
                prod = dff.groupby("NumOfProducts")["Exited"].mean().reset_index()
                prod["ChurnRate"] = prod["Exited"] * 100
                prod = prod.sort_values("NumOfProducts")

                fig = px.bar(prod, x="NumOfProducts", y="ChurnRate", text=prod["ChurnRate"].round(2), title="Churn Rate by Number of Products (%)")
                fig = _soft_bar(fig, PALETTE["green"])
                fig = _base_layout(fig, title="Churn Rate by Number of Products (%)")
                fig.update_yaxes(title="Churn Rate (%)")
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom `NumOfProducts` tidak tersedia / data kosong.")

        st.markdown("**📌 Insight (Behavior):**")
        beh_ins = []

        if "IsActiveMember" in dff.columns and total:
            act_rate = dff.groupby("IsActiveMember")["Exited"].mean()
            if 0 in act_rate and 1 in act_rate:
                ratio = (act_rate[0] / act_rate[1]) if act_rate[1] != 0 else None
                if ratio:
                    beh_ins.append(
                        f"🔴 **Engagement Risk:** Nasabah tidak aktif churn **{_pct(act_rate[0]):.1f}%** vs aktif **{_pct(act_rate[1]):.1f}%** "
                        f"(≈ **{ratio:.1f}×** lebih tinggi). Engagement adalah salah satu driver churn paling kuat."
                    )
                else:
                    beh_ins.append(
                        f"🔴 **Engagement Risk:** Nasabah tidak aktif churn **{_pct(act_rate[0]):.1f}%**, sementara churn pada aktif sangat rendah. "
                        "Prioritaskan program re-engagement."
                    )

        if "NumOfProducts" in dff.columns and total:
            p_rate = dff.groupby("NumOfProducts")["Exited"].mean()
            if not p_rate.empty:
                worst_p = p_rate.idxmax()
                beh_ins.append(
                    f"📦 **Product Holding:** Churn tertinggi pada **NumOfProducts={worst_p}** "
                    f"(**{_pct(p_rate[worst_p]):.1f}%**). Ini bisa terkait mismatch benefit atau kurangnya cross-sell relevan."
                )

        if beh_ins:
            for i in beh_ins:
                st.info(i)
        else:
            st.caption("Belum terlihat pola perilaku yang konsisten terhadap churn.")

    with tab4:
        st.markdown("### Financial (Value Impact)")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            if "Balance" in dff.columns and total:
                tmp = dff.copy()
                tmp["Status"] = tmp["Exited"].map({0: "Active", 1: "Churn"}).astype(str)
                fig = px.box(
                    tmp,
                    x="Status",
                    y="Balance",
                    color="Status",
                    title="Balance by Status",
                    color_discrete_map={"Active": PALETTE["green"], "Churn": PALETTE["red"]},
                )
                fig = _base_layout(fig, title="Balance by Status")
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom `Balance` tidak tersedia / data kosong.")

        with c2:
            if "EstimatedSalary" in dff.columns and total:
                tmp = dff.copy()
                tmp["Status"] = tmp["Exited"].map({0: "Active", 1: "Churn"}).astype(str)
                fig = px.box(
                    tmp,
                    x="Status",
                    y="EstimatedSalary",
                    color="Status",
                    title="Estimated Salary by Status",
                    color_discrete_map={"Active": PALETTE["green"], "Churn": PALETTE["red"]},
                )
                fig = _base_layout(fig, title="Estimated Salary by Status")
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom `EstimatedSalary` tidak tersedia / data kosong.")

        st.markdown("**📌 Insight (Financial):**")
        fin_ins = []

        if "Balance" in dff.columns and total:
            churn_bal = _safe_mean(dff.loc[dff["Exited"] == 1, "Balance"])
            active_bal = _safe_mean(dff.loc[dff["Exited"] == 0, "Balance"])
            if churn_bal is not None and active_bal is not None:
                if churn_bal > active_bal:
                    fin_ins.append(
                        f"💎 **Value at Risk:** Saldo rata-rata churn **lebih tinggi** ({_format_money(churn_bal)}) dibanding aktif ({_format_money(active_bal)}). "
                        "Hal ini berarti churn berpotensi menggerus value lebih besar → proteksi high-value customer penting."
                    )
                else:
                    fin_ins.append(
                        f"💰 **Balance Pattern:** Saldo churn ({_format_money(churn_bal)}) lebih rendah dari aktif ({_format_money(active_bal)}). "
                        "Fokus retensi dapat diarahkan ke segmen mass untuk menekan churn rate."
                    )

        if "EstimatedSalary" in dff.columns and total:
            churn_sal = _safe_mean(dff.loc[dff["Exited"] == 1, "EstimatedSalary"])
            active_sal = _safe_mean(dff.loc[dff["Exited"] == 0, "EstimatedSalary"])
            if churn_sal is not None and active_sal is not None:
                fin_ins.append(
                    f"💼 **Income Profile:** Estimasi pendapatan churn ({_format_money(churn_sal)}) vs aktif ({_format_money(active_sal)}). "
                    "Temuan ini dapat membantu menentukan positioning benefit & channel komunikasi."
                )

        if fin_ins:
            for i in fin_ins:
                st.info(i)
        else:
            st.caption("Tidak ditemukan perbedaan finansial yang signifikan.")

    st.divider()
    st.markdown("## 🔎 Key Insights Summary")

    insights = []
    actions = []

    if total:
        if "IsActiveMember" in dff.columns:
            act_rate = dff.groupby("IsActiveMember")["Exited"].mean()
            if 0 in act_rate and 1 in act_rate:
                insights.append(
                    f"🔴 Engagement driver kuat: non-aktif churn **{_pct(act_rate[0]):.1f}%** vs aktif **{_pct(act_rate[1]):.1f}%**."
                )
                if act_rate[0] > act_rate[1]:
                    actions.append("🎯 Jalankan **re-engagement** (promo personal, reminder transaksi, lifecycle messaging).")

        top_geo = _top_churn_by_group(dff, "Geography")
        if top_geo:
            insights.append(f"🌍 Wilayah risiko tertinggi: **{top_geo[0]}** (churn **{_pct(top_geo[1]):.1f}%**).")
            actions.append(f"🌍 Fokus retensi di **{top_geo[0]}**: audit pain points lokal, evaluasi produk & kompetisi.")

        if "Balance" in dff.columns:
            churn_bal = _safe_mean(dff.loc[dff["Exited"] == 1, "Balance"])
            active_bal = _safe_mean(dff.loc[dff["Exited"] == 0, "Balance"])
            if churn_bal is not None and active_bal is not None and churn_bal > active_bal:
                insights.append("💎 Risiko kehilangan high-value customer (saldo churn > saldo aktif).")
                actions.append("💎 Buat **VIP retention playbook**: prioritas layanan + loyalty benefit + outreach proaktif.")

    insights = _dedupe_keep_order(insights)
    actions = _dedupe_keep_order(actions)

    if insights:
        for ins in insights:
            st.success(ins)
    else:
        st.info("Tidak cukup data untuk menghasilkan insight otomatis pada filter saat ini.")

    st.markdown("## ✅ Recommended Actions")
    if actions:
        for a in actions[:6]:
            st.warning(a)
    else:
        st.info("Belum ada rekomendasi yang dapat diturunkan dari data pada filter saat ini.")

    with st.expander("📄 Sample Data"):
        st.dataframe(dff.head(), use_container_width=True)

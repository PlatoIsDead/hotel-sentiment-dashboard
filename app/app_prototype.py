"""
Hotel Analytics Dashboard
Run: streamlit run app.py
Reads from: booking_risk.parquet, guest_enriched.parquet, hotel_summary.parquet
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Отели — Аналитика",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

RISK_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
TOPIC_RU = {
    "прочее": "Прочее",
    "вопросы": "Вопросы",
    "заезд-выезд": "Заезд / Выезд",
    "чистота": "Чистота",
    "интернет": "Интернет",
    "платежи": "Платежи",
    "ремонт": "Ремонт",
    "температура": "Температура",
    "парковка": "Парковка",
    "шум": "Шум",
    "удобства": "Удобства",
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    br = pd.read_parquet(BASE / "booking_risk.parquet")
    ge = pd.read_parquet(BASE / "guest_enriched.parquet")
    hs = pd.read_parquet(BASE / "hotel_summary.parquet")
    ge["month"] = ge["date_add"].dt.to_period("M").astype(str)
    ge["year"]  = ge["date_add"].dt.year
    return br, ge, hs

br, ge, hs = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🏨 Фильтры")

all_hotels = sorted(br["hotel_id"].unique().tolist())
hotel_labels = {h: f"Отель {h}" for h in all_hotels}

selected_hotels = st.sidebar.multiselect(
    "Отели",
    options=all_hotels,
    default=all_hotels,
    format_func=lambda h: hotel_labels[h],
)

selected_risk = st.sidebar.multiselect(
    "Уровень риска",
    options=["LOW", "MEDIUM", "HIGH"],
    default=["LOW", "MEDIUM", "HIGH"],
    format_func=lambda x: {"LOW": "✅ Низкий", "MEDIUM": "⚠️ Средний", "HIGH": "🔴 Высокий"}[x],
)

# Filter
br_f = br[br["hotel_id"].isin(selected_hotels) & br["risk_level"].isin(selected_risk)]
ge_f = ge[ge["hotel_id"].isin(selected_hotels)]
hs_f = hs[hs["hotel_id"].isin(selected_hotels)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Аналитика гостевых коммуникаций")
st.caption(f"Данные: {ge['date_add'].min().date()} — {ge['date_add'].max().date()} · "
           f"{br['ID_booking'].nunique():,} бронирований · {len(ge):,} сообщений · "
           f"{len(all_hotels)} отелей")

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)

total_bookings = br_f["ID_booking"].nunique()
high_risk = (br_f["risk_level"] == "HIGH").sum()
high_pct   = high_risk / len(br_f) * 100 if len(br_f) else 0
med_reply  = br_f["reply_time_min"].median()
threats    = hs_f["threats"].sum()
complaints = hs_f["complaints"].sum()

c1.metric("Бронирований", f"{total_bookings:,}")
c2.metric("Высокий риск", f"{high_risk:,}", f"{high_pct:.1f}%")
c3.metric("Медианное время ответа", f"{med_reply:.0f} мин")
c4.metric("Угрозы", f"{int(threats)}")
c5.metric("Жалобы", f"{int(complaints)}")

st.markdown("---")

# ── Row 1: Risk breakdown + Reply time ───────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Распределение риска по отелям")
    risk_counts = (
        br_f.groupby(["hotel_id", "risk_level"])
        .size()
        .reset_index(name="n")
    )
    risk_counts["hotel"] = risk_counts["hotel_id"].map(lambda h: f"Отель {h}")
    fig = px.bar(
        risk_counts,
        x="hotel",
        y="n",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        barmode="stack",
        labels={"n": "Бронирований", "hotel": "Отель", "risk_level": "Риск"},
        category_orders={"risk_level": ["HIGH", "MEDIUM", "LOW"]},
    )
    fig.update_layout(margin=dict(t=10, b=10), legend_title="Риск", height=320)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Время ответа администратора")
    bucket_order = ["<=5m", "5-15m", "15-60m", ">60m"]
    bucket_labels = {"<=5m": "≤5 мин", "5-15m": "5–15 мин", "15-60m": "15–60 мин", ">60m": ">60 мин"}
    reply_counts = (
        br_f.groupby(["hotel_id", "reply_bucket"])
        .size()
        .reset_index(name="n")
    )
    reply_counts["hotel"] = reply_counts["hotel_id"].map(lambda h: f"Отель {h}")
    reply_counts["bucket_label"] = reply_counts["reply_bucket"].map(bucket_labels)
    fig2 = px.bar(
        reply_counts,
        x="hotel",
        y="n",
        color="bucket_label",
        barmode="stack",
        labels={"n": "Бронирований", "hotel": "Отель", "bucket_label": "Время ответа"},
        category_orders={"bucket_label": [bucket_labels[b] for b in bucket_order]},
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig2.update_layout(margin=dict(t=10, b=10), legend_title="Время ответа", height=320)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Topics + Sentiment trend ──────────────────────────────────────────
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Топ тем обращений")
    topic_counts = (
        br_f.groupby("top_topic")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=True)
    )
    topic_counts["topic_label"] = topic_counts["top_topic"].map(
        lambda t: TOPIC_RU.get(t, t)
    )
    # Exclude "прочее" from chart — it's a catch-all
    topic_counts = topic_counts[topic_counts["top_topic"] != "прочее"]
    fig3 = px.bar(
        topic_counts.tail(12),
        x="n",
        y="topic_label",
        orientation="h",
        labels={"n": "Бронирований", "topic_label": "Тема"},
        color="n",
        color_continuous_scale="Blues",
    )
    fig3.update_layout(margin=dict(t=10, b=10), showlegend=False,
                       coloraxis_showscale=False, height=340)
    st.plotly_chart(fig3, use_container_width=True)

with col_right2:
    st.subheader("Доля негативных сообщений по месяцам")
    ge_sel = ge_f.copy()
    monthly = (
        ge_sel.groupby(["hotel_id", "month", "hf_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["NEG", "NEU", "POS"]:
        if col not in monthly.columns:
            monthly[col] = 0
    monthly["total"] = monthly[["NEG", "NEU", "POS"]].sum(axis=1)
    monthly["NEG_%"] = (monthly["NEG"] / monthly["total"] * 100).round(1)
    monthly["hotel"] = monthly["hotel_id"].map(lambda h: f"Отель {h}")
    monthly_sorted = monthly.sort_values("month")

    fig4 = px.line(
        monthly_sorted,
        x="month",
        y="NEG_%",
        color="hotel",
        labels={"NEG_%": "% негативных", "month": "Месяц", "hotel": "Отель"},
        markers=True,
    )
    fig4.update_layout(margin=dict(t=10, b=10), height=340,
                       xaxis_tickangle=-45, legend_title="Отель")
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Hotel summary table ────────────────────────────────────────────────
st.subheader("Сводка по отелям")
display_hs = hs_f.copy()
display_hs["hotel_id"] = display_hs["hotel_id"].map(lambda h: f"Отель {h}")
display_hs = display_hs.rename(columns={
    "hotel_id":           "Отель",
    "n_bookings":         "Бронирований",
    "avg_risk":           "Ср. риск",
    "high_risk_bookings": "Высокий риск",
    "high_risk_share_%":  "% высокого риска",
    "median_reply_min":   "Медиана ответа (мин)",
    "avg_neg_share":      "% негатива",
    "threats":            "Угрозы",
    "complaints":         "Жалобы",
    "problems":           "Проблемы",
})
st.dataframe(
    display_hs.set_index("Отель"),
    use_container_width=True,
)

# ── Row 4: High-risk bookings table ──────────────────────────────────────────
st.markdown("---")
st.subheader("🔴 Бронирования с высоким риском")

high_risk_df = br_f[br_f["risk_level"] == "HIGH"].copy()
high_risk_df = high_risk_df.sort_values("risk_score", ascending=False)

high_risk_df["hotel"] = high_risk_df["hotel_id"].map(lambda h: f"Отель {h}")
high_risk_df["topic_label"] = high_risk_df["top_topic"].map(lambda t: TOPIC_RU.get(t, t))

show_cols = {
    "hotel":           "Отель",
    "ID_booking":      "ID бронирования",
    "risk_score":      "Риск",
    "topic_label":     "Тема",
    "n_guest_msgs":    "Сообщений гостя",
    "neg_share":       "% негатива",
    "reply_time_min":  "Ответ (мин)",
    "THREAT":          "Угрозы",
    "COMPLAINT":       "Жалобы",
    "last_sentiment":  "Последний тон",
}
display_hr = high_risk_df[list(show_cols.keys())].rename(columns=show_cols)

st.dataframe(
    display_hr.head(50),
    use_container_width=True,
    column_config={
        "Риск": st.column_config.ProgressColumn(
            "Риск", min_value=0, max_value=100, format="%.0f"
        ),
        "% негатива": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.caption(f"Показано {min(50, len(display_hr))} из {len(display_hr)} бронирований с высоким риском")

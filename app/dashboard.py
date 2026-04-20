"""
Дашборд гостиничных коммуникаций
Запуск: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Аналитика отелей",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).parent.parent

HOTEL_NAMES = {1: "БС74", 2: "МК16", 3: "You&Co", 4: "М73", 5: "О-44"}
BUCKET_ORDER  = ["<=5m", "5-15m", "15-60m", ">60m"]
BUCKET_LABELS = {"<=5m": "≤5 мин", "5-15m": "5–15 мин", "15-60m": "15–60 мин", ">60m": ">60 мин"}
BUCKET_COLORS = ["#22c55e", "#86efac", "#fbbf24", "#ef4444"]


@st.cache_data
def load_all():
    br    = pd.read_parquet(BASE / "data/booking_response.parquet")
    hs    = pd.read_parquet(BASE / "data/hotel_summary_v2.parquet")
    daily = pd.read_parquet(BASE / "data/daily_response.parquet")
    th    = pd.read_parquet(BASE / "data/threads_sample_classified.parquet")
    ht    = pd.read_parquet(BASE / "data/hotel_topics.parquet")
    ba    = pd.read_parquet(BASE / "data/bot_answerability.parquet")
    br["first_guest_time"] = pd.to_datetime(br["first_guest_time"])
    br["year"] = br["first_guest_time"].dt.year
    daily["date"] = pd.to_datetime(daily["date"])
    th["thread_start"] = pd.to_datetime(th["thread_start"])
    th["hotel_name"] = th["hotel_id"].map(HOTEL_NAMES)
    return br, hs, daily, th, ht, ba

br, hs, daily, th, ht, ba = load_all()
ALL_YEARS  = sorted(br["year"].dropna().unique().tolist())
ALL_HOTELS = sorted(br["hotel_id"].unique().tolist())


st.title("🏨 Аналитика гостевых коммуникаций")
tab1, tab2, tab3 = st.tabs(["📊 Статистика по отелям", "❓ Топ темы", "🤖 Бот vs Оператор"])

# ── Вкладка 1 ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Сводка по отелям")
    st.dataframe(
        hs.rename(columns={
            "hotel_name": "Отель", "n_bookings": "Бронирований",
            "response_rate": "% с ответом", "median_reply": "Медиана (мин)",
            "mean_reply": "Среднее (мин)", "p90_reply": "P90 (мин)",
            "no_reply": "Без ответа", "avg_rating": "Рейтинг",
            "n_reviews": "Отзывов", "low_rating_pct": "% низких оценок",
        }).drop(columns=["hotel_id", "n_with_text"]).set_index("Отель"),
        use_container_width=True)

    st.markdown("---")
    st.subheader("Динамика времени ответа")

    sel_hotel_t1 = st.selectbox("Отель", sorted(daily["hotel_name"].unique()))
    d = daily[daily["hotel_name"] == sel_hotel_t1].copy()

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=d["date"], y=d["median_reply"], mode="markers", name="Медиана по дням",
        opacity=0.35, marker=dict(size=4, color="#93c5fd")))
    fig_daily.add_trace(go.Scatter(
        x=d["date"], y=d["rolling_30d"], mode="lines", name="Тренд 30д",
        line=dict(width=2.5, color="#ef4444")))
    fig_daily.update_layout(height=320, margin=dict(t=10, b=10),
                             yaxis_title="Минуты", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_daily, use_container_width=True)

    st.markdown("---")

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        sel_years = st.multiselect("Годы", ALL_YEARS, default=ALL_YEARS, format_func=str)
    with fc2:
        pass

    br_y = br[br["year"].isin(sel_years)].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Боксплот времени ответа")
        fig_box = go.Figure()
        for hid in ALL_HOTELS:
            data = br_y[br_y["hotel_id"] == hid]["reply_time_min"].dropna()
            cap  = data[data <= data.quantile(0.95)]
            fig_box.add_trace(go.Box(y=cap, name=HOTEL_NAMES[hid], boxmean=True,
                                     marker_color=px.colors.qualitative.Set2[hid % 8]))
        fig_box.update_layout(yaxis_title="Минуты (до 95‑го перцентиля)",
                               showlegend=False, height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.subheader("Скорость ответа (% бронирований)")
        bdata = (br_y.groupby(["hotel_name","reply_bucket"]).size().reset_index(name="n"))
        bdata["pct"] = (bdata["n"] / bdata.groupby("hotel_name")["n"].transform("sum") * 100).round(1)
        bdata["bucket_ru"] = bdata["reply_bucket"].map(BUCKET_LABELS)
        fig_buck = px.bar(bdata, x="hotel_name", y="pct", color="bucket_ru", barmode="stack",
                          color_discrete_sequence=BUCKET_COLORS,
                          category_orders={"bucket_ru": [BUCKET_LABELS[b] for b in BUCKET_ORDER]},
                          labels={"pct": "% бронирований", "hotel_name": "Отель", "bucket_ru": ""})
        fig_buck.update_layout(height=340, margin=dict(t=10, b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_buck, use_container_width=True)

# ── Вкладка 2 ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Топ темы обращений гостей")

    all_hotels_ht = sorted(ht["hotel_name"].unique().tolist())
    sel_hotel = st.selectbox("Отель", all_hotels_ht)
    view_ht = ht[ht["hotel_name"] == sel_hotel].copy()

    fig_topics = px.bar(
        view_ht, x="% от отеля", y="Ключевые слова", orientation="h",
        color="% от отеля", color_continuous_scale="Blues",
        hover_data=["Сообщений", "Пример"],
        labels={"% от отеля": "% сообщений", "Ключевые слова": ""},
    )
    fig_topics.update_layout(
        height=max(380, len(view_ht) * 32),
        margin=dict(t=10, b=10),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_topics, use_container_width=True)

    st.dataframe(
        view_ht[["Ключевые слова", "Сообщений", "% от отеля", "Пример"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Сообщений": st.column_config.ProgressColumn(
                "Сообщений", min_value=0,
                max_value=int(view_ht["Сообщений"].max()), format="%d"),
            "Пример": st.column_config.TextColumn("Пример", width="large"),
        }
    )


# ── Вкладка 3 ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Бот vs Оператор")
    st.caption(f"{ba['thread_start'].min().date()} — {ba['thread_start'].max().date()} · "
               f"{len(ba)} тредов · {ba['ID_booking'].nunique()} бронирований · "
               f"{ba['hotel_name'].nunique()} отеля")

    bot_pct = (ba["label"] == "BOT").mean() * 100
    human_pct = 100 - bot_pct
    m1, m2, m3 = st.columns(3)
    m1.metric("Всего тредов", len(ba))
    m2.metric("🤖 Бот закрывает", f"{bot_pct:.1f}%")
    m3.metric("👤 Нужен оператор", f"{human_pct:.1f}%")

    st.markdown("---")

    hotel_stats = (ba.groupby(["hotel_name", "label"])
                   .size().reset_index(name="n"))
    hotel_stats["pct"] = (
        hotel_stats["n"] /
        hotel_stats.groupby("hotel_name")["n"].transform("sum") * 100
    ).round(1)
    hotel_stats["label_ru"] = hotel_stats["label"].map({"BOT": "🤖 Бот", "HUMAN": "👤 Оператор"})

    fig_ba = px.bar(
        hotel_stats, x="hotel_name", y="pct", color="label_ru", barmode="stack",
        color_discrete_map={"🤖 Бот": "#22c55e", "👤 Оператор": "#ef4444"},
        labels={"pct": "% тредов", "hotel_name": "Отель", "label_ru": ""},
    )
    fig_ba.update_layout(height=320, margin=dict(t=10, b=10),
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_ba, use_container_width=True)

    st.markdown("---")

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        label_filter = st.multiselect(
            "Метка", ["BOT", "HUMAN"], default=["BOT", "HUMAN"],
            format_func=lambda x: {"BOT": "🤖 Бот", "HUMAN": "👤 Оператор"}[x])
    with fc2:
        keyword_ba = st.text_input("Поиск", placeholder="wifi, уборка, залог...")
    with fc3:
        min_conf_ba = st.slider("Мин. уверенность", 0.0, 1.0, 0.0, 0.1)

    view_ba = ba[ba["label"].isin(label_filter) & (ba["confidence"] >= min_conf_ba)].copy()
    if keyword_ba:
        view_ba = view_ba[
            view_ba["text"].str.contains(keyword_ba, case=False, na=False) |
            view_ba["reason"].str.contains(keyword_ba, case=False, na=False)
        ]
    view_ba["Метка"] = view_ba["label"].map({"BOT": "🤖 Бот", "HUMAN": "👤 Оператор"})

    st.caption(f"Найдено: {len(view_ba):,} тредов")
    st.dataframe(
        view_ba[["hotel_name", "ID_booking", "Метка", "thread_start",
                 "n_guest_msgs", "confidence", "reason", "text"]]
        .sort_values("thread_start", ascending=False)
        .rename(columns={
            "hotel_name": "Отель", "ID_booking": "Бронирование",
            "thread_start": "Начало", "n_guest_msgs": "Сообщений",
            "confidence": "Уверенность", "reason": "Причина", "text": "Текст",
        }),
        use_container_width=True, height=500,
        column_config={
            "Уверенность": st.column_config.ProgressColumn(
                "Уверенность", min_value=0.0, max_value=1.0, format="%.2f"),
            "Текст":   st.column_config.TextColumn("Текст",   width="large"),
            "Причина": st.column_config.TextColumn("Причина", width="medium"),
        }
    )


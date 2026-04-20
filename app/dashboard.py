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

HOTEL_NAMES = {1: "БС74", 2: "МК16", 3: "Дубай", 4: "М73", 5: "О-44"}
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
    br["first_guest_time"] = pd.to_datetime(br["first_guest_time"])
    br["year"] = br["first_guest_time"].dt.year
    daily["date"] = pd.to_datetime(daily["date"])
    th["thread_start"] = pd.to_datetime(th["thread_start"])
    th["hotel_name"] = th["hotel_id"].map(HOTEL_NAMES)

    gt, tt = None, None
    gt_path = BASE / "data/guest_topics_3m.parquet"
    tt_path = BASE / "data/thread_topics_3m.parquet"
    if gt_path.exists():
        gt = pd.read_parquet(gt_path)
    if tt_path.exists():
        tt = pd.read_parquet(tt_path)
        tt["thread_start"] = pd.to_datetime(tt["thread_start"])
    return br, hs, daily, th, ht, gt, tt

br, hs, daily, th, ht, gt, tt = load_all()
ALL_YEARS  = sorted(br["year"].dropna().unique().tolist())
ALL_HOTELS = sorted(br["hotel_id"].unique().tolist())


st.title("🏨 Аналитика гостевых коммуникаций")
tab1, tab2, tab3 = st.tabs(["📊 Статистика по отелям", "❓ Топ темы", "🗣️ Горячие темы (3 мес.)"])

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

    st.markdown("---")
    with st.expander("ℹ️ Как считается время ответа"):
        st.markdown("""
**Источник данных:** `2026_04_17_messages.xls` (3 листа, ~184 тыс. сообщений) + `2026_04_17_reviews.xls`

**Алгоритм:**
1. Переписка по каждой брони разбивается на **треды** — пауза между сообщениями > 4 ч считается началом нового треда
2. Для каждого треда берётся время от **первого гостевого сообщения** до **первого ответа администратора**
3. Ответ позже 24 ч не засчитывается (→ `NaN`, тред считается без ответа)
4. Итоговое время ответа по брони — **медиана по всем тредам** этой брони

**Период:** 2022-03-31 — 2026-04-17 · 30 370 броней · 34 688 тредов с гостевыми сообщениями
""")

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
    st.subheader("Горячие темы гостей — последние 3 месяца")

    if gt is None:
        st.info("Данные не найдены. Запустите `notebooks/bot_answerability.ipynb`.")
        st.stop()

    period_cap = ""
    if tt is not None and len(tt) > 0:
        period_cap = (f"{tt['thread_start'].min().date()} — {tt['thread_start'].max().date()} · "
                      f"{tt['ID_booking'].nunique():,} бронирований · ")
    st.caption(f"{period_cap}{gt['hotel_name'].nunique()} отелей")

    sel_hotel_t3 = st.selectbox("Отель", sorted(gt["hotel_name"].unique()), key="t3_hotel")
    view_gt = gt[gt["hotel_name"] == sel_hotel_t3].sort_values("n_threads", ascending=False)

    fig_gt = px.bar(
        view_gt, x="n_threads", y="topic", orientation="h",
        color="pct_of_hotel", color_continuous_scale="Blues",
        hover_data={"pct_of_hotel": True, "example": True, "n_threads": False},
        labels={"n_threads": "Тредов", "topic": "", "pct_of_hotel": "% тредов"},
    )
    fig_gt.update_layout(
        height=max(380, len(view_gt) * 40),
        margin=dict(t=10, b=10),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_gt, use_container_width=True)

    st.dataframe(
        view_gt[["topic", "n_threads", "pct_of_hotel", "example"]].rename(columns={
            "topic": "Тема", "n_threads": "Тредов",
            "pct_of_hotel": "% тредов", "example": "Пример",
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "Тредов": st.column_config.ProgressColumn(
                "Тредов", min_value=0, max_value=int(view_gt["n_threads"].max()), format="%d"),
            "Пример": st.column_config.TextColumn("Пример", width="large"),
        },
    )

    if tt is not None and len(tt) > 0:
        st.markdown("---")
        st.subheader("Треды")

        view_tt = tt[tt["hotel_name"] == sel_hotel_t3].copy()
        fc1, fc2 = st.columns([2, 3])
        with fc1:
            topics_in_hotel = ["Все"] + sorted(view_tt["topic"].unique())
            sel_topic = st.selectbox("Тема", topics_in_hotel, key="t3_topic")
        with fc2:
            kw_t3 = st.text_input("Поиск по тексту", placeholder="уборка, wifi...", key="t3_kw")

        if sel_topic != "Все":
            view_tt = view_tt[view_tt["topic"] == sel_topic]
        if kw_t3:
            view_tt = view_tt[view_tt["text"].str.contains(kw_t3, case=False, na=False)]

        st.caption(f"Найдено: {len(view_tt):,} тредов")
        st.dataframe(
            view_tt[["topic", "thread_start", "ID_booking", "n_guest_msgs", "text"]]
            .sort_values("thread_start", ascending=False)
            .rename(columns={
                "topic": "Тема", "thread_start": "Начало",
                "ID_booking": "Бронирование", "n_guest_msgs": "Сообщений", "text": "Текст",
            }),
            use_container_width=True, height=480,
            column_config={
                "Текст": st.column_config.TextColumn("Текст", width="large"),
            },
        )


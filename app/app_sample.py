"""
Hotel Sample — Thread Analysis Dashboard
Data: Feb–Mar 2026 · Hotel 5 · 1114 threads · 765 bookings

Run from the folder containing threads_sample_classified.parquet:
    streamlit run app_sample.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Анализ обращений — выборка",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

CAT_COLORS = {
    "PROBLEM":  "#ef4444",
    "QUESTION": "#3b82f6",
    "OTHER":    "#94a3b8",
    "ERROR":    "#fbbf24",
}
CAT_RU = {
    "PROBLEM":  "🔧 Проблема",
    "QUESTION": "❓ Вопрос",
    "OTHER":    "💬 Прочее",
    "ERROR":    "⚠️ Ошибка",
}

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_parquet(BASE / "threads_sample_classified.parquet")
    df["thread_start"] = pd.to_datetime(df["thread_start"])
    df["thread_end"]   = pd.to_datetime(df["thread_end"])
    df["date"]         = df["thread_start"].dt.date
    df["week"]         = df["thread_start"].dt.to_period("W").astype(str)
    df["cat_ru"]       = df["category"].map(CAT_RU)
    return df

df = load()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📋 Фильтры")

min_conf = st.sidebar.slider(
    "Минимальная уверенность модели", 0.0, 1.0, 0.0, 0.05,
    help="Исключить потоки с низкой уверенностью классификации"
)

selected_cats = st.sidebar.multiselect(
    "Категория",
    options=["PROBLEM", "QUESTION", "OTHER"],
    default=["PROBLEM", "QUESTION", "OTHER"],
    format_func=lambda x: CAT_RU[x],
)

df_f = df[
    (df["category"].isin(selected_cats)) &
    (df["confidence"] >= min_conf)
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Анализ обращений — выборка за месяц")
st.caption(
    f"Отель 5 · "
    f"{df['thread_start'].min().date()} — {df['thread_start'].max().date()} · "
    f"{df['ID_booking'].nunique():,} бронирований · "
    f"{len(df):,} тредов"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)

n_threads  = len(df_f)
n_problem  = (df_f["category"] == "PROBLEM").sum()
n_question = (df_f["category"] == "QUESTION").sum()
n_other    = (df_f["category"] == "OTHER").sum()
avg_conf   = df_f["confidence"].mean()

c1.metric("Тредов (выборка)", f"{n_threads:,}")
c2.metric("🔧 Проблемы", f"{n_problem:,}", f"{n_problem/n_threads*100:.0f}%")
c3.metric("❓ Вопросы",  f"{n_question:,}", f"{n_question/n_threads*100:.0f}%")
c4.metric("💬 Прочее",  f"{n_other:,}", f"{n_other/n_threads*100:.0f}%")
c5.metric("Ср. уверенность", f"{avg_conf:.2f}")

st.markdown("---")

# ── Row 1: Pie + Weekly trend ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Распределение категорий")
    pie_data = df_f["category"].value_counts().reset_index()
    pie_data.columns = ["category", "n"]
    pie_data["label"] = pie_data["category"].map(CAT_RU)
    fig_pie = px.pie(
        pie_data,
        values="n",
        names="label",
        color="category",
        color_discrete_map=CAT_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        height=320,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Треды по неделям")
    weekly = (
        df_f.groupby(["week", "category"])
        .size()
        .reset_index(name="n")
    )
    weekly["label"] = weekly["category"].map(CAT_RU)
    weekly = weekly.sort_values("week")
    fig_bar = px.bar(
        weekly,
        x="week",
        y="n",
        color="category",
        color_discrete_map=CAT_COLORS,
        barmode="stack",
        labels={"n": "Тредов", "week": "Неделя", "category": ""},
        category_orders={"category": ["PROBLEM", "QUESTION", "OTHER"]},
    )
    fig_bar.update_layout(
        margin=dict(t=10, b=10),
        xaxis_tickangle=-30,
        legend_title="",
        height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Row 2: Threads per booking + Confidence dist ──────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Тредов на бронирование")
    tpb = (
        df_f.groupby("ID_booking")
        .size()
        .reset_index(name="n_threads")
    )
    tpb_counts = tpb["n_threads"].value_counts().sort_index().reset_index()
    tpb_counts.columns = ["threads", "bookings"]
    tpb_counts = tpb_counts[tpb_counts["threads"] <= 10]  # cap display
    fig_tpb = px.bar(
        tpb_counts,
        x="threads",
        y="bookings",
        labels={"threads": "Кол-во тредов", "bookings": "Бронирований"},
        color_discrete_sequence=["#3b82f6"],
    )
    fig_tpb.update_layout(margin=dict(t=10, b=10), height=300)
    st.plotly_chart(fig_tpb, use_container_width=True)

with col4:
    st.subheader("Уверенность классификации")
    fig_conf = px.histogram(
        df_f,
        x="confidence",
        color="category",
        color_discrete_map=CAT_COLORS,
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        labels={"confidence": "Уверенность", "count": "Тредов"},
    )
    fig_conf.update_layout(
        margin=dict(t=10, b=10),
        legend_title="",
        height=300,
    )
    st.plotly_chart(fig_conf, use_container_width=True)

# ── Thread browser ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Просмотр тредов")

tab_problem, tab_question, tab_other = st.tabs(["🔧 Проблемы", "❓ Вопросы", "💬 Прочее"])

def render_thread_table(cat):
    sub = df[df["category"] == cat].copy()
    sub = sub.sort_values("thread_start", ascending=False)

    # Sort controls
    col_a, col_b = st.columns([3, 1])
    with col_a:
        search = st.text_input(
            "Поиск по тексту",
            key=f"search_{cat}",
            placeholder="Введите ключевое слово...",
        )
    with col_b:
        min_c = st.slider(
            "Мин. уверенность",
            0.0, 1.0, 0.0, 0.1,
            key=f"conf_{cat}",
        )

    if search:
        sub = sub[sub["text"].str.contains(search, case=False, na=False)]
    sub = sub[sub["confidence"] >= min_c]

    st.caption(f"Показано: {len(sub):,} тредов")

    display = sub[[
        "ID_booking", "thread_id", "thread_start",
        "n_guest_msgs", "confidence", "reason", "text"
    ]].rename(columns={
        "ID_booking":   "Бронирование",
        "thread_id":    "Тред",
        "thread_start": "Начало",
        "n_guest_msgs": "Сообщений",
        "confidence":   "Уверенность",
        "reason":       "Причина",
        "text":         "Текст",
    })

    st.dataframe(
        display,
        use_container_width=True,
        height=420,
        column_config={
            "Уверенность": st.column_config.ProgressColumn(
                "Уверенность", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Текст": st.column_config.TextColumn("Текст", width="large"),
            "Причина": st.column_config.TextColumn("Причина", width="medium"),
        },
    )

with tab_problem:
    render_thread_table("PROBLEM")

with tab_question:
    render_thread_table("QUESTION")

with tab_other:
    render_thread_table("OTHER")

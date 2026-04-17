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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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

STOP_RU = [
    "и","в","не","на","что","я","с","он","а","как","это","но","по","к","у","из",
    "за","то","до","мне","вы","мы","же","для","бы","все","так","уже","если","или",
    "да","нет","при","от","об","со","во","его","её","их","ее","был","была","были",
    "будет","есть","вот","там","здесь","чтобы","когда","который","можно","можете",
    "можем","пожалуйста","спасибо","добрый","доброго","здравствуйте","привет",
    "день","вечер","утро","дня","вечера","утра","хорошо","ладно","понял","поняла",
    "окей","ок","пока","сегодня","завтра","вчера","номер","номера","номере",
    "отель","отеля","отеле","нас","вас","них","нам","вам","им","ему","ей",
    "меня","тебя","тебе","себя","себе","этот","эта","эти","того","тех","том",
    "один","два","три","раз","очень","также","тоже","чтоб","потому","поэтому",
    "большое","благодарю","заранее","просьба","прошу","подскажите","скажите",
    "скиньте","пришлите","отправьте","напишите",
]

@st.cache_data
def load():
    br = pd.read_parquet(BASE / "booking_risk.parquet")
    hs = pd.read_parquet(BASE / "hotel_summary.parquet")
    th = pd.read_parquet(BASE / "threads_sample_classified.parquet")
    br["first_guest_time"] = pd.to_datetime(br["first_guest_time"])
    br.loc[br["reply_time_min"] < 0, "reply_time_min"] = np.nan
    br["year"]       = br["first_guest_time"].dt.year
    br["hotel_name"] = br["hotel_id"].map(HOTEL_NAMES)
    hs["hotel_name"] = hs["hotel_id"].map(HOTEL_NAMES)
    th["thread_start"] = pd.to_datetime(th["thread_start"])
    th["hotel_name"] = th["hotel_id"].map(HOTEL_NAMES)
    return br, hs, th

br, hs, th = load()
ALL_YEARS  = sorted(br["year"].dropna().unique().tolist())
ALL_HOTELS = sorted(br["hotel_id"].unique().tolist())

@st.cache_data
def cluster_questions(n_topics: int):
    q_texts = th[th["category"] == "QUESTION"]["text"].tolist()
    n_topics = min(n_topics, max(5, len(q_texts) // 5))
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3,
                          sublinear_tf=True, stop_words=STOP_RU)
    X = vec.fit_transform(q_texts)
    km = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    rows = []
    for i, center in enumerate(km.cluster_centers_):
        top_idx   = center.argsort()[-5:][::-1]
        top_terms = [terms[j] for j in top_idx]
        size      = int((labels == i).sum())
        cluster_texts = [t for t, l in zip(q_texts, labels) if l == i]
        example = cluster_texts[0][:150].replace("\n---\n", " / ")
        rows.append({"Тема": ", ".join(top_terms), "Кол-во тредов": size, "Пример": example})
    return pd.DataFrame(rows).sort_values("Кол-во тредов", ascending=False).reset_index(drop=True)

st.title("🏨 Аналитика гостевых коммуникаций")
tab1, tab2, tab3 = st.tabs(["📊 Статистика по отелям", "❓ Топ вопросы", "🔍 Браузер тредов"])

# ── Вкладка 1 ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Сводка по отелям (все годы)")
    reply_stats = (br.groupby("hotel_id")["reply_time_min"]
        .agg(median_reply="median", mean_reply="mean",
             p90_reply=lambda x: x.quantile(0.9))
        .round(1).reset_index())
    no_reply = (br.groupby("hotel_id")
        .apply(lambda x: x["reply_time_min"].isna().sum())
        .reset_index(name="no_reply"))
    resp_rate = (br.groupby("hotel_id")
        .apply(lambda x: (x["reply_time_min"].notna()).mean() * 100)
        .round(1).reset_index(name="response_rate_%"))
    summary = (hs[["hotel_id","hotel_name","n_bookings","avg_neg_share"]]
               .merge(reply_stats, on="hotel_id")
               .merge(no_reply, on="hotel_id")
               .merge(resp_rate, on="hotel_id"))
    st.dataframe(
        summary.rename(columns={
            "hotel_name": "Отель", "n_bookings": "Бронирований",
            "response_rate_%": "% с ответом", "median_reply": "Медиана (мин)",
            "mean_reply": "Среднее (мин)", "p90_reply": "P90 (мин)",
            "no_reply": "Без ответа", "avg_neg_share": "% негатива",
        }).drop(columns=["hotel_id"]).set_index("Отель"),
        use_container_width=True)

    st.markdown("---")
    st.subheader("Время ответа по годам")

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        sel_years = st.multiselect("Годы", ALL_YEARS, default=ALL_YEARS, format_func=str)
    with fc2:
        metric = st.selectbox("Метрика", ["Медиана", "Среднее", "P90"])

    br_y = br[br["year"].isin(sel_years)].copy()
    yearly = (br_y.groupby(["hotel_id","hotel_name","year"])["reply_time_min"]
              .agg(median="median", mean="mean", p90=lambda x: x.quantile(0.9))
              .round(1).reset_index())
    yearly["year"] = yearly["year"].astype(str)
    col_name = {"Медиана": "median", "Среднее": "mean", "P90": "p90"}[metric]

    fig_yr = px.line(yearly, x="year", y=col_name, color="hotel_name", markers=True,
                     labels={col_name: f"{metric} (мин)", "year": "Год", "hotel_name": "Отель"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_yr.update_layout(height=320, margin=dict(t=10, b=10), legend_title="Отель")
    st.plotly_chart(fig_yr, use_container_width=True)

    st.markdown("---")
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
    st.subheader("Топ вопросы гостей")
    st.caption(f"О-44 · последний месяц · {(th['category']=='QUESTION').sum()} тредов · TF-IDF без LLM")
    n_topics = st.slider("Количество тем", 5, 25, 15, 1)
    with st.spinner("Кластеризация..."):
        topics_df = cluster_questions(n_topics)
    fig_topics = px.bar(topics_df, x="Кол-во тредов", y="Тема", orientation="h",
                        color="Кол-во тредов", color_continuous_scale="Blues",
                        hover_data=["Пример"])
    fig_topics.update_layout(height=max(380, n_topics * 30), margin=dict(t=10, b=10),
                              yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig_topics, use_container_width=True)
    st.dataframe(topics_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Кол-во тредов": st.column_config.ProgressColumn(
                         "Кол-во тредов", min_value=0,
                         max_value=int(topics_df["Кол-во тредов"].max()), format="%d"),
                     "Пример": st.column_config.TextColumn("Пример треда", width="large"),
                 })

# ── Вкладка 3 ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Браузер тредов — последний месяц (О-44)")
    st.caption(f"{th['thread_start'].min().date()} — {th['thread_start'].max().date()} · "
               f"{len(th)} тредов · {th['ID_booking'].nunique()} бронирований")

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        cat_filter = st.multiselect("Категория",
            ["PROBLEM","QUESTION","OTHER"], default=["PROBLEM","QUESTION"],
            format_func=lambda x: {"PROBLEM":"🔧 Проблема","QUESTION":"❓ Вопрос","OTHER":"💬 Прочее"}[x])
    with fc2:
        keyword = st.text_input("Поиск", placeholder="уборка, wifi, залог...")
    with fc3:
        min_conf = st.slider("Мин. уверенность", 0.0, 1.0, 0.0, 0.1)

    view = th[th["category"].isin(cat_filter) & (th["confidence"] >= min_conf)].copy()
    if keyword:
        view = view[view["text"].str.contains(keyword, case=False, na=False) |
                    view["reason"].str.contains(keyword, case=False, na=False)]

    view["Категория"] = view["category"].map(
        {"PROBLEM":"🔧 Проблема","QUESTION":"❓ Вопрос","OTHER":"💬 Прочее"})
    st.caption(f"Найдено: {len(view):,} тредов")
    st.dataframe(
        view[["ID_booking","thread_id","Категория","thread_start",
              "n_guest_msgs","confidence","reason","text"]]
        .sort_values("thread_start", ascending=False)
        .rename(columns={"ID_booking":"Бронирование","thread_id":"Тред №",
                         "thread_start":"Начало","n_guest_msgs":"Сообщений",
                         "confidence":"Уверенность","reason":"Причина","text":"Текст"}),
        use_container_width=True, height=520,
        column_config={
            "Уверенность": st.column_config.ProgressColumn(
                "Уверенность", min_value=0.0, max_value=1.0, format="%.2f"),
            "Текст":   st.column_config.TextColumn("Текст",   width="large"),
            "Причина": st.column_config.TextColumn("Причина", width="medium"),
        })

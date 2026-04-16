"""
Streamlit dashboard — Hotel Guest Communication Analysis
Russian UI, Russian charts, English code comments.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Анализ обращений гостей",
    page_icon="🏨",
    layout="wide",
)

HOTEL_NAMES = {1: "БС74", 2: "МК16", 3: "You&Co", 4: "М73", 5: "О-44"}

SENTIMENT_COLORS = {"NEG": "#e63946", "NEU": "#adb5bd", "POS": "#2dc653"}
SENTIMENT_LABELS = {"NEG": "Негатив", "NEU": "Нейтрал", "POS": "Позитив"}

CATEGORY_COLORS = {
    "PROBLEM": "#e63946", "QUESTION": "#4a90d9", "OTHER": "#adb5bd", "ERROR": "#888888"
}
CATEGORY_LABELS = {
    "PROBLEM": "Проблема", "QUESTION": "Вопрос", "OTHER": "Прочее", "ERROR": "Ошибка"
}

BASE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    guest = pd.read_parquet(BASE / "guest_enriched.parquet")
    booking = pd.read_parquet(BASE / "booking_risk.parquet")
    hotel_summary = pd.read_parquet(BASE / "hotel_summary.parquet")
    threads = pd.read_parquet(BASE / "threads_classified.parquet")
    topics = pd.read_parquet(BASE / "hotel_topics.parquet")

    guest["hotel_name"] = guest["hotel_id"].map(HOTEL_NAMES)
    booking["hotel_name"] = booking["hotel_id"].map(HOTEL_NAMES)
    hotel_summary["hotel_name"] = hotel_summary["hotel_id"].map(HOTEL_NAMES)
    threads["hotel_name"] = threads["hotel_id"].map(HOTEL_NAMES)

    guest["date_add"] = pd.to_datetime(guest["date_add"])
    guest["year_month"] = guest["date_add"].dt.to_period("M").astype(str)

    booking = booking.drop(
        columns=["risk_score", "risk_level", "reply_bucket"], errors="ignore"
    )
    return guest, booking, hotel_summary, threads, topics


guest, booking, hotel_summary, threads, topics = load_data()

hotel_order = [HOTEL_NAMES[i] for i in sorted(HOTEL_NAMES)]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🏨 Анализ обращений гостей")
st.caption(
    f"5 отелей · {len(guest):,} сообщений · {len(booking):,} бронирований · "
    f"{guest['date_add'].dt.year.min()}–{guest['date_add'].dt.year.max()}"
)

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏨 Отели и время ответа",
    "🎭 Тональность",
    "📝 Темы",
    "🧵 Типы тредов",
    "📋 Бронирования",
])


# ===========================================================================
# TAB 0 — ОТЕЛИ И ВРЕМЯ ОТВЕТА  (самый интерактивный)
# ===========================================================================

with tab0:
    with st.expander("📖 Как это считалось?", expanded=False):
        st.markdown("""
**Время ответа:**
```
время_ответа = первое_сообщение_администратора − первое_сообщение_гостя
               в рамках одного бронирования  (в минутах)
```
- Если администратор написал раньше гостя → 0 мин (граничный случай).
- Если администратор не ответил вообще → бронирование помечается как «без ответа»
  и исключается из медианы.

**Показатели в карточках:**
- **Бронирований** — уникальных `ID_booking` с ≥1 гостевым сообщением.
- **Ответили** — бронирования, в которых есть хотя бы одно сообщение администратора.
- **Медиана ответа** — 50-й перцентиль по всем бронированиям с ответом.
- **Доля негатива** — среднее `neg_share` по бронированиям отеля.

**Почему медиана, а не среднее?**
Распределение времени ответа сильно скошено: есть бронирования с ответом через несколько дней,
которые вздёрнут среднее. Медиана устойчива к таким выбросам.
""")

    # ── Interactive hotel selector ──────────────────────────────────────────
    all_label = "Все отели"
    hotel_choice = st.radio(
        "Выберите отель для детализации",
        options=[all_label] + hotel_order,
        horizontal=True,
        key="overview_hotel",
    )

    # Filter booking dataframe
    if hotel_choice == all_label:
        bk = booking.copy()
        hs = hotel_summary.copy()
    else:
        bk = booking[booking["hotel_name"] == hotel_choice].copy()
        hs = hotel_summary[hotel_summary["hotel_name"] == hotel_choice].copy()

    bk_with_reply = bk[bk["reply_time_min"] >= 0].copy()

    st.divider()

    # ── Metric cards per hotel (or single hotel) ────────────────────────────
    if hotel_choice == all_label:
        # Show one card per hotel
        cols = st.columns(len(hotel_order))
        for col, h_name in zip(cols, hotel_order):
            h_row = hotel_summary[hotel_summary["hotel_name"] == h_name].iloc[0]
            h_bk = booking[booking["hotel_name"] == h_name]
            h_reply = h_bk[h_bk["reply_time_min"] >= 0]

            resp_rate = (len(h_reply) / len(h_bk) * 100) if len(h_bk) > 0 else 0
            med_reply = h_reply["reply_time_min"].median() if len(h_reply) > 0 else float("nan")
            avg_neg = h_bk["neg_share"].mean()

            with col:
                st.markdown(f"### {h_name}")
                st.metric("Бронирований", f"{int(h_row['n_bookings']):,}")
                st.metric("Ответили", f"{resp_rate:.0f}%")
                st.metric("Медиана ответа", f"{med_reply:.0f} мин" if not np.isnan(med_reply) else "—")
                st.metric("Ср. негатив", f"{avg_neg:.1f}%")
    else:
        # Single hotel — big metrics in a row
        h_row = hs.iloc[0] if len(hs) > 0 else None
        resp_rate = (len(bk_with_reply) / len(bk) * 100) if len(bk) > 0 else 0
        med_reply = bk_with_reply["reply_time_min"].median() if len(bk_with_reply) > 0 else float("nan")
        avg_neg = bk["neg_share"].mean() if len(bk) > 0 else 0
        no_reply_pct = (bk["reply_time_min"] == -1).mean() * 100 if len(bk) > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Бронирований", f"{len(bk):,}")
        c2.metric("Ответ. %", f"{resp_rate:.0f}%")
        c3.metric("Медиана ответа", f"{med_reply:.0f} мин" if not np.isnan(med_reply) else "—")
        c4.metric("Без ответа", f"{no_reply_pct:.1f}%")
        c5.metric("Ср. негатив", f"{avg_neg:.1f}%")

    st.divider()

    # ── Response time distribution ──────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        # Histogram of reply times (cap at 240 min for readability)
        cap_min = st.slider(
            "Обрезать ось X (мин)", min_value=30, max_value=1440, value=120, step=30,
            key="reply_cap",
        )
        capped = bk_with_reply[bk_with_reply["reply_time_min"] <= cap_min].copy()

        if hotel_choice == all_label:
            fig_hist = px.histogram(
                capped, x="reply_time_min", color="hotel_name",
                nbins=40, barmode="overlay", opacity=0.7,
                labels={"reply_time_min": "Минуты до первого ответа", "hotel_name": "Отель"},
                title="Распределение времени ответа",
                category_orders={"hotel_name": hotel_order},
            )
        else:
            fig_hist = px.histogram(
                capped, x="reply_time_min", nbins=40,
                color_discrete_sequence=["#4a90d9"],
                labels={"reply_time_min": "Минуты до первого ответа"},
                title=f"Распределение времени ответа — {hotel_choice}",
            )
        fig_hist.add_vline(x=15, line_dash="dash", line_color="orange",
                           annotation_text="15 мин")
        fig_hist.add_vline(x=60, line_dash="dash", line_color="red",
                           annotation_text="60 мин")
        fig_hist.update_layout(height=380)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        # Median reply per hotel (bar)
        if hotel_choice == all_label:
            hotel_reply_med = (
                booking[booking["reply_time_min"] >= 0]
                .groupby("hotel_name")["reply_time_min"]
                .median()
                .reset_index()
                .rename(columns={"reply_time_min": "медиана_мин"})
            )
            hotel_reply_med["hotel_name"] = pd.Categorical(
                hotel_reply_med["hotel_name"], categories=hotel_order, ordered=True
            )
            fig_med = px.bar(
                hotel_reply_med.sort_values("hotel_name"),
                x="hotel_name", y="медиана_мин",
                color="медиана_мин", color_continuous_scale="RdYlGn_r",
                text="медиана_мин",
                labels={"hotel_name": "Отель", "медиана_мин": "Медиана (мин)"},
                title="Медиана времени первого ответа по отелям",
                category_orders={"hotel_name": hotel_order},
            )
            fig_med.update_traces(texttemplate="%{text:.0f} мин", textposition="outside")
            fig_med.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_med, use_container_width=True)
        else:
            # Percentile breakdown for single hotel
            if len(bk_with_reply) > 0:
                pcts = [10, 25, 50, 75, 90, 95]
                pct_vals = np.percentile(bk_with_reply["reply_time_min"], pcts)
                pct_df = pd.DataFrame({"Перцентиль": [f"P{p}" for p in pcts], "Минуты": pct_vals.round(1)})
                fig_pct = px.bar(
                    pct_df, x="Перцентиль", y="Минуты",
                    color="Минуты", color_continuous_scale="RdYlGn_r",
                    text="Минуты",
                    title=f"Перцентили времени ответа — {hotel_choice}",
                )
                fig_pct.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                fig_pct.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_pct, use_container_width=True)

    # ── Reply bucket analysis ───────────────────────────────────────────────
    st.subheader("Группировка по скорости ответа")

    bucket_bins = [-0.1, 5, 15, 60, 240, float("inf")]
    bucket_labels = ["≤5 мин", "5–15 мин", "15–60 мин", "1–4 часа", ">4 часов"]

    bk_buckets = bk_with_reply.copy()
    bk_buckets["bucket"] = pd.cut(
        bk_buckets["reply_time_min"], bins=bucket_bins, labels=bucket_labels
    )

    bucket_stats = (
        bk_buckets.groupby("bucket", observed=True)
        .agg(
            n_бронирований=("ID_booking", "count"),
            ср_негатив=("neg_share", "mean"),
            ср_серия_neg=("neg_streak", "mean"),
        )
        .reset_index()
        .rename(columns={"bucket": "Скорость ответа"})
    )
    bucket_stats["ср_негатив"] = bucket_stats["ср_негатив"].round(1)
    bucket_stats["ср_серия_neg"] = bucket_stats["ср_серия_neg"].round(2)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_bkt_n = px.bar(
            bucket_stats, x="Скорость ответа", y="n_бронирований",
            color="n_бронирований", color_continuous_scale="Blues",
            text="n_бронирований",
            title="Кол-во бронирований по скорости ответа",
        )
        fig_bkt_n.update_traces(textposition="outside")
        fig_bkt_n.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_bkt_n, use_container_width=True)

    with col_b2:
        fig_bkt_neg = px.bar(
            bucket_stats, x="Скорость ответа", y="ср_негатив",
            color="ср_негатив", color_continuous_scale="RdYlGn_r",
            text="ср_негатив",
            title="Средняя доля негатива по группам ответа (%)",
            labels={"ср_негатив": "% негатива"},
        )
        fig_bkt_neg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bkt_neg.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_bkt_neg, use_container_width=True)

    # ── Scatter: reply time vs neg_share ───────────────────────────────────
    st.subheader("Взаимосвязь времени ответа и негатива")

    scatter_cap = st.slider(
        "Показывать бронирования с ответом до (мин)", 10, 1440, 300, step=10,
        key="scatter_cap",
    )
    scatter_data = bk_with_reply[bk_with_reply["reply_time_min"] <= scatter_cap].copy()

    if hotel_choice == all_label:
        scatter_data["hotel_name"] = pd.Categorical(
            scatter_data["hotel_name"], categories=hotel_order, ordered=True
        )
        fig_sc = px.scatter(
            scatter_data,
            x="reply_time_min", y="neg_share",
            color="hotel_name",
            opacity=0.4, trendline="lowess",
            trendline_options=dict(frac=0.4),
            labels={
                "reply_time_min": "Время ответа (мин)",
                "neg_share": "Доля негатива (%)",
                "hotel_name": "Отель",
            },
            title="Время ответа vs доля негатива (каждая точка = бронирование)",
            hover_data=["ID_booking", "n_guest_msgs"],
            category_orders={"hotel_name": hotel_order},
        )
    else:
        fig_sc = px.scatter(
            scatter_data,
            x="reply_time_min", y="neg_share",
            opacity=0.5, trendline="lowess",
            trendline_options=dict(frac=0.4),
            color_discrete_sequence=["#4a90d9"],
            labels={
                "reply_time_min": "Время ответа (мин)",
                "neg_share": "Доля негатива (%)",
            },
            title=f"{hotel_choice} — время ответа vs доля негатива",
            hover_data=["ID_booking", "n_guest_msgs"],
        )
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── No-reply bookings ───────────────────────────────────────────────────
    st.subheader("Бронирования без ответа администратора")
    no_reply = bk[bk["reply_time_min"] == -1].copy()
    st.caption(
        f"Таких бронирований: **{len(no_reply):,}** "
        f"({len(no_reply)/len(bk)*100:.1f}% от выборки)"
    )
    if len(no_reply) > 0:
        nr_by_hotel = (
            no_reply.groupby("hotel_name")
            .agg(n=("ID_booking", "count"), avg_neg=("neg_share", "mean"))
            .reset_index()
        )
        nr_by_hotel["avg_neg"] = nr_by_hotel["avg_neg"].round(1)
        fig_nr = px.bar(
            nr_by_hotel, x="hotel_name", y="n",
            color="avg_neg", color_continuous_scale="Reds",
            text="n",
            labels={"hotel_name": "Отель", "n": "Без ответа", "avg_neg": "Ср. негатив %"},
            title="Бронирования без ответа (цвет = средний % негатива гостя)",
            category_orders={"hotel_name": hotel_order},
        )
        fig_nr.update_traces(textposition="outside")
        fig_nr.update_layout(height=340)
        st.plotly_chart(fig_nr, use_container_width=True)


# ===========================================================================
# TAB 1 — ТОНАЛЬНОСТЬ
# ===========================================================================

with tab1:
    with st.expander("📖 Как это считалось?", expanded=False):
        st.markdown("""
**Модель:** [`cointegrated/rubert-tiny-sentiment-balanced`](https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced) —
компактная русскоязычная BERT-модель, дообученная на сбалансированном корпусе отзывов и переписок.

**Процесс:**
1. Каждое сообщение гостя подаётся в модель отдельно.
2. Модель возвращает три вероятности: NEG / NEU / POS.
3. Класс с максимальной вероятностью становится меткой сообщения.
4. Уверенность (`hf_confidence`) — максимальная из трёх вероятностей.

**Агрегация по бронированию:**
- `neg_share` = доля NEG-сообщений среди всех сообщений гостя в этом бронировании (%).

**Ограничения:**
- Сарказм модель может пропустить.
- Короткие нейтральные запросы («Добрый день, уборка?») часто получают метку NEU — это корректно.
""")

    # Sentiment % per hotel
    sent_pct = (
        guest.groupby(["hotel_name", "sentiment"])
        .size().reset_index(name="n")
    )
    sent_pct["total"] = sent_pct.groupby("hotel_name")["n"].transform("sum")
    sent_pct["pct"] = (sent_pct["n"] / sent_pct["total"] * 100).round(1)
    sent_pct["sentiment_ru"] = sent_pct["sentiment"].map(SENTIMENT_LABELS)
    sent_pct["hotel_name"] = pd.Categorical(sent_pct["hotel_name"], categories=hotel_order, ordered=True)

    fig_sent = px.bar(
        sent_pct.sort_values("hotel_name"),
        x="hotel_name", y="pct", color="sentiment_ru",
        barmode="stack",
        color_discrete_map={v: SENTIMENT_COLORS[k] for k, v in SENTIMENT_LABELS.items()},
        labels={"hotel_name": "Отель", "pct": "% сообщений", "sentiment_ru": "Тональность"},
        title="Распределение тональности по отелям",
        text="pct",
        category_orders={"hotel_name": hotel_order},
    )
    fig_sent.update_traces(texttemplate="%{text:.0f}%", textposition="inside", textfont_size=12)
    fig_sent.update_layout(height=420, legend_title="Тональность")
    st.plotly_chart(fig_sent, use_container_width=True)

    # Monthly NEG trend
    monthly = (
        guest.groupby("year_month")
        .agg(total=("sentiment", "size"), neg=("sentiment", lambda x: (x == "NEG").sum()))
        .reset_index()
    )
    monthly["neg_pct"] = (monthly["neg"] / monthly["total"] * 100).round(1)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly["year_month"], y=monthly["total"],
        name="Всего сообщений", marker_color="#4a90d9", opacity=0.3, yaxis="y2",
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["neg_pct"],
        name="% негатива", mode="lines+markers",
        line=dict(color="#e63946", width=3), marker=dict(size=7),
    ))
    fig_trend.update_layout(
        title="Динамика негатива по месяцам",
        xaxis_title="Месяц",
        yaxis=dict(title="% негатива", ticksuffix="%"),
        yaxis2=dict(title="Кол-во сообщений", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Year-over-year by hotel
    guest_v = guest[guest["date_add"].notna()].copy()
    guest_v["year"] = guest_v["date_add"].dt.year
    yearly = (
        guest_v.groupby(["hotel_name", "year"])
        .agg(total=("sentiment", "size"), neg=("sentiment", lambda x: (x == "NEG").sum()))
        .reset_index()
    )
    yearly["neg_pct"] = (yearly["neg"] / yearly["total"] * 100).round(1)
    yearly["hotel_name"] = pd.Categorical(yearly["hotel_name"], categories=hotel_order, ordered=True)

    fig_yearly = px.line(
        yearly, x="year", y="neg_pct", color="hotel_name", markers=True,
        labels={"year": "Год", "neg_pct": "% негативных сообщений", "hotel_name": "Отель"},
        title="% негативных сообщений по годам",
        category_orders={"hotel_name": hotel_order},
    )
    fig_yearly.update_layout(height=360, yaxis_ticksuffix="%")
    st.plotly_chart(fig_yearly, use_container_width=True)


# ===========================================================================
# TAB 2 — ТЕМЫ
# ===========================================================================

with tab2:
    with st.expander("📖 Как это считалось?", expanded=False):
        st.markdown("""
**Лемматизация** (`pymorphy3`):
Каждое слово приводится к начальной форме: «уборку» → «уборка», «работает» → «работать».
Оставляются только существительные и глаголы. Стоп-слова (приветствия, местоимения) убираются.

**TF-IDF** (Term Frequency–Inverse Document Frequency):
Каждое сообщение → вектор весов слов.
- TF: как часто слово встречается в этом сообщении.
- IDF: насколько редко слово встречается по всем сообщениям (редкие = информативнее).

Слова вроде «пожалуйста» встречаются везде → низкий IDF → низкий вес.
«Шлагбаум» встречается только в О-44 → высокий IDF → высокий вес.

**KMeans** (10 кластеров на отель):
Группирует сообщения по близости TF-IDF-векторов. Каждый кластер = тема.
Показываются 6 топ-слов с наибольшим весом в центре кластера.

Кластеры строятся **отдельно для каждого отеля**.
""")

    selected_hotel_topics = st.selectbox("Отель", options=hotel_order, key="topics_hotel")
    hotel_id_sel = {v: k for k, v in HOTEL_NAMES.items()}[selected_hotel_topics]
    hotel_topics_df = topics[topics["hotel_id"] == hotel_id_sel].copy()
    hotel_topics_df = hotel_topics_df.sort_values("Сообщений", ascending=True)

    fig_topics = px.bar(
        hotel_topics_df, x="Сообщений", y="Ключевые слова",
        orientation="h",
        title=f"{selected_hotel_topics} — темы обращений",
        color="Сообщений", color_continuous_scale="Blues",
        hover_data={"Пример": True, "% от отеля": True},
        text="% от отеля",
    )
    fig_topics.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_topics.update_layout(
        height=500, coloraxis_showscale=False,
        yaxis_title="", xaxis_title="Количество сообщений",
    )
    st.plotly_chart(fig_topics, use_container_width=True)

    # Heatmap across hotels (keyword-based topics from guest_enriched)
    st.subheader("Сравнение тем по всем отелям")
    topic_hotel = (
        guest.groupby(["hotel_name", "topic"]).size().reset_index(name="n")
    )
    topic_hotel["total"] = topic_hotel.groupby("hotel_name")["n"].transform("sum")
    topic_hotel["pct"] = (topic_hotel["n"] / topic_hotel["total"] * 100).round(1)
    pivot = topic_hotel.pivot(index="topic", columns="hotel_name", values="pct").fillna(0)
    cols = [h for h in hotel_order if h in pivot.columns]
    pivot = pivot[cols]

    fig_heat = px.imshow(
        pivot,
        title="% сообщений по теме от всех сообщений отеля",
        labels=dict(x="Отель", y="Тема", color="%"),
        color_continuous_scale="Blues", aspect="auto", text_auto=".1f",
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================================
# TAB 3 — ТИПЫ ТРЕДОВ
# ===========================================================================

with tab3:
    with st.expander("📖 Как это считалось?", expanded=False):
        st.markdown("""
**Что такое тред?**
Тред = группа последовательных сообщений в рамках одного бронирования,
между которыми прошло **не более 4 часов**.
Если пауза больше — начинается новый тред.

Итого: **21,707 тредов** в 7,982 бронированиях (в среднем 2.8 треда на бронь).

**Классификация GPT-4o-mini:**
Каждый тред (все гостевые сообщения внутри него) отправлялся в OpenAI:

- **PROBLEM** — нужно физическое действие персонала: уборка, починка, доставка.
- **QUESTION** — достаточно текстового ответа: пароль WiFi, время завтрака, правила.
- **OTHER** — не вписывается: жалоба без запроса, тесты, угрозы.

Модель также возвращала `reason` (объяснение) и `confidence` (уверенность 0–1).
""")

    valid_threads = threads[threads["category"] != "ERROR"].copy()
    valid_threads["category_ru"] = valid_threads["category"].map(CATEGORY_LABELS)

    col_a, col_b = st.columns(2)

    with col_a:
        cat_counts = valid_threads["category_ru"].value_counts().reset_index()
        cat_counts.columns = ["Тип", "Количество"]
        fig_pie = px.pie(
            cat_counts, names="Тип", values="Количество",
            title="Типы тредов (все отели)",
            color="Тип",
            color_discrete_map={v: CATEGORY_COLORS[k] for k, v in CATEGORY_LABELS.items()},
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+value")
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        hotel_cat = (
            valid_threads.groupby(["hotel_name", "category_ru"]).size().reset_index(name="n")
        )
        hotel_cat["total"] = hotel_cat.groupby("hotel_name")["n"].transform("sum")
        hotel_cat["pct"] = (hotel_cat["n"] / hotel_cat["total"] * 100).round(1)
        hotel_cat["hotel_name"] = pd.Categorical(hotel_cat["hotel_name"], categories=hotel_order, ordered=True)

        fig_cat_bar = px.bar(
            hotel_cat.sort_values("hotel_name"),
            x="hotel_name", y="pct", color="category_ru",
            barmode="stack",
            color_discrete_map={v: CATEGORY_COLORS[k] for k, v in CATEGORY_LABELS.items()},
            labels={"hotel_name": "Отель", "pct": "%", "category_ru": "Тип"},
            title="Типы тредов по отелям (%)",
            category_orders={"hotel_name": hotel_order},
        )
        fig_cat_bar.update_layout(height=380, legend_title="Тип")
        st.plotly_chart(fig_cat_bar, use_container_width=True)

    # Top PROBLEM reasons
    st.subheader("Топ проблем по отелям")
    selected_hotel_threads = st.selectbox("Отель", options=hotel_order, key="threads_hotel")

    problems = valid_threads[
        (valid_threads["category"] == "PROBLEM") &
        (valid_threads["hotel_name"] == selected_hotel_threads)
    ].copy()

    top_prob = (
        problems.groupby("reason").size().reset_index(name="n")
        .sort_values("n", ascending=False).head(12)
    )
    top_prob["reason_short"] = top_prob["reason"].str[:80]

    fig_prob = px.bar(
        top_prob, x="n", y="reason_short", orientation="h",
        title=f"{selected_hotel_threads} — частые проблемы (по описанию GPT)",
        labels={"n": "Количество тредов", "reason_short": ""},
        color="n", color_continuous_scale="Reds",
    )
    fig_prob.update_layout(height=480, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_prob, use_container_width=True)

    with st.expander("Примеры тредов"):
        sample = (
            valid_threads[valid_threads["hotel_name"] == selected_hotel_threads]
            .sort_values("confidence", ascending=False)
            .head(20)[["category_ru", "confidence", "reason", "text"]]
            .rename(columns={
                "category_ru": "Тип", "confidence": "Уверенность",
                "reason": "Причина", "text": "Текст (гость)",
            })
        )
        sample["Текст (гость)"] = sample["Текст (гость)"].str[:120]
        st.dataframe(sample, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4 — БРОНИРОВАНИЯ
# ===========================================================================

with tab4:
    with st.expander("📖 Как это считалось?", expanded=False):
        st.markdown("""
**Метрики бронирования** — агрегаты по всем сообщениям гостя в рамках одной брони:

| Метрика | Что это |
|---|---|
| `neg_share` | % сообщений с меткой NEG от всех гостевых сообщений |
| `neg_streak` | Максимальная длина последовательных NEG-сообщений подряд |
| `last_sentiment` | Тональность последнего сообщения гостя |
| `reply_time_min` | Минуты до первого ответа администратора |
| `n_guest_msgs` | Количество сообщений от гостя |
| `admin_guest_ratio` | Сколько сообщений написал admin на 1 сообщение гостя |

**neg_streak** — индикатор нарастающего недовольства: если гость написал 3 NEG подряд,
ситуация скорее всего не разрешалась.

**last_sentiment** — как завершился диалог с точки зрения гостя.
NEG в конце = гость ушёл неудовлетворённым.
""")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_hotels = st.multiselect("Отель", options=hotel_order, default=hotel_order)
    with col_f2:
        min_neg = st.slider("Мин. доля негатива (%)", 0, 100, 30, step=5)
    with col_f3:
        min_msgs = st.slider("Мин. сообщений гостя", 1, 20, 3)

    filtered = booking[
        (booking["hotel_name"].isin(sel_hotels)) &
        (booking["neg_share"] >= min_neg) &
        (booking["n_guest_msgs"] >= min_msgs)
    ].copy()
    filtered["hotel_name"] = pd.Categorical(filtered["hotel_name"], categories=hotel_order, ordered=True)
    filtered = filtered.sort_values(["neg_share", "neg_streak"], ascending=[False, False])

    st.caption(f"Найдено бронирований: **{len(filtered):,}**")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_b_scatter = px.scatter(
            filtered, x="neg_share", y="neg_streak",
            color="hotel_name", size="n_guest_msgs", opacity=0.6,
            labels={
                "neg_share": "Доля негатива (%)",
                "neg_streak": "Макс. серия NEG подряд",
                "hotel_name": "Отель", "n_guest_msgs": "Сообщений",
            },
            title="Доля негатива vs серия негатива",
            hover_data=["ID_booking", "last_sentiment", "reply_time_min"],
            category_orders={"hotel_name": hotel_order},
        )
        fig_b_scatter.update_layout(height=420)
        st.plotly_chart(fig_b_scatter, use_container_width=True)

    with col_s2:
        last_sent = (
            filtered.groupby(["hotel_name", "last_sentiment"]).size().reset_index(name="n")
        )
        last_sent["last_ru"] = last_sent["last_sentiment"].map(SENTIMENT_LABELS)
        last_sent["hotel_name"] = pd.Categorical(last_sent["hotel_name"], categories=hotel_order, ordered=True)
        fig_last = px.bar(
            last_sent.sort_values("hotel_name"),
            x="hotel_name", y="n", color="last_ru",
            barmode="stack",
            color_discrete_map={v: SENTIMENT_COLORS[k] for k, v in SENTIMENT_LABELS.items()},
            labels={"hotel_name": "Отель", "n": "Бронирований", "last_ru": "Последнее"},
            title="Тональность последнего сообщения гостя",
            category_orders={"hotel_name": hotel_order},
        )
        fig_last.update_layout(height=420, legend_title="Последнее")
        st.plotly_chart(fig_last, use_container_width=True)

    st.subheader("Таблица бронирований")
    display_cols = {
        "hotel_name": "Отель", "ID_booking": "Бронь",
        "neg_share": "Негатив %", "neg_streak": "Серия NEG",
        "last_sentiment": "Последнее", "reply_time_min": "Ответ (мин)",
        "n_guest_msgs": "Сообщ. гостя", "admin_guest_ratio": "Admin/Guest",
        "top_topic": "Тема",
    }
    table = filtered[list(display_cols.keys())].rename(columns=display_cols).head(300)
    table["Последнее"] = table["Последнее"].map(SENTIMENT_LABELS).fillna(table["Последнее"])

    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "Негатив %": st.column_config.ProgressColumn(
                "Негатив %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Серия NEG": st.column_config.NumberColumn("Серия NEG"),
        },
    )

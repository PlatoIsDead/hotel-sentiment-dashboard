# =============================================================================
# 🏨 АНАЛИЗ КОММУНИКАЦИИ С ГОСТЯМИ - STREAMLIT DASHBOARD
# Для менеджеров отелей (на русском языке)
# Запуск: streamlit run hotel_dashboard_ru.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# НАСТРОЙКИ СТРАНИЦЫ
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Анализ гостей",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# СЛОВАРЬ ОТЕЛЕЙ
# -----------------------------------------------------------------------------
HOTEL_NAMES = {
    1: "БС74",
    2: "МК16",
    3: "Дубай",
    4: "М73",
    5: "О44"
}

# Обратный словарь для фильтров
HOTEL_IDS = {v: k for k, v in HOTEL_NAMES.items()}

# -----------------------------------------------------------------------------
# ЗАГРУЗКА ДАННЫХ
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Загрузка обработанных данных"""
    try:
        guest = pd.read_parquet("guest_enriched.parquet")
        booking_risk = pd.read_parquet("booking_risk.parquet")
        hotel_summary = pd.read_parquet("hotel_summary.parquet")
        top_topics = pd.read_csv("top_topics_per_hotel.csv")
        top_risky = pd.read_csv("top_risky_bookings.csv")
        sla_view = pd.read_csv("sla_view.csv")

        # Добавляем названия отелей
        guest["hotel_name"] = guest["hotel_id"].map(HOTEL_NAMES)
        booking_risk["hotel_name"] = booking_risk["hotel_id"].map(HOTEL_NAMES)
        hotel_summary["hotel_name"] = hotel_summary["hotel_id"].map(HOTEL_NAMES)
        top_topics["hotel_name"] = top_topics["hotel_id"].map(HOTEL_NAMES)
        top_risky["hotel_name"] = top_risky["hotel_id"].map(HOTEL_NAMES)

        # Парсим даты
        if "date_add" in guest.columns:
            guest["date_add"] = pd.to_datetime(guest["date_add"])
            guest["year"] = guest["date_add"].dt.year
            guest["month"] = guest["date_add"].dt.month
            guest["year_month"] = guest["date_add"].dt.to_period("M").astype(str)

        return guest, booking_risk, hotel_summary, top_topics, top_risky, sla_view
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        st.info("Сначала запустите скрипт анализа для создания файлов данных.")
        return None, None, None, None, None, None

guest, booking_risk, hotel_summary, top_topics, top_risky, sla_view = load_data()

if guest is None:
    st.stop()

# -----------------------------------------------------------------------------
# ЗАГОЛОВОК
# -----------------------------------------------------------------------------
st.title("🏨 Анализ коммуникации с гостями")
st.markdown("**Мониторинг тональности, рисков и качества обслуживания**")

# -----------------------------------------------------------------------------
# БОКОВАЯ ПАНЕЛЬ - ФИЛЬТРЫ
# -----------------------------------------------------------------------------
st.sidebar.header("🔧 Фильтры")

# Выбор отеля
hotel_options = ["Все отели"] + [HOTEL_NAMES[i] for i in sorted(HOTEL_NAMES.keys())]
selected_hotel_name = st.sidebar.selectbox("Выберите отель", hotel_options)

# Выбор года (если есть данные по датам)
if "year" in guest.columns:
    years = sorted(guest["year"].dropna().unique())
    year_options = ["Все годы"] + [str(int(y)) for y in years]
    selected_year = st.sidebar.selectbox("Выберите год", year_options)
else:
    selected_year = "Все годы"

# Выбор уровня риска
risk_options = ["Все уровни", "ВЫСОКИЙ", "СРЕДНИЙ", "НИЗКИЙ"]
selected_risk = st.sidebar.selectbox("Уровень риска", risk_options)

# Маппинг уровней риска
RISK_MAP_RU = {"HIGH": "ВЫСОКИЙ", "MEDIUM": "СРЕДНИЙ", "LOW": "НИЗКИЙ"}
RISK_MAP_EN = {"ВЫСОКИЙ": "HIGH", "СРЕДНИЙ": "MEDIUM", "НИЗКИЙ": "LOW"}

# -----------------------------------------------------------------------------
# ПРИМЕНЕНИЕ ФИЛЬТРОВ
# -----------------------------------------------------------------------------
guest_filtered = guest.copy()
booking_filtered = booking_risk.copy()

# Фильтр по отелю
if selected_hotel_name != "Все отели":
    hotel_id = HOTEL_IDS[selected_hotel_name]
    guest_filtered = guest_filtered[guest_filtered["hotel_id"] == hotel_id]
    booking_filtered = booking_filtered[booking_filtered["hotel_id"] == hotel_id]

# Фильтр по году
if selected_year != "Все годы" and "year" in guest_filtered.columns:
    year_int = int(selected_year)
    guest_filtered = guest_filtered[guest_filtered["year"] == year_int]
    # Для бронирований нужно джойнить с guest чтобы получить год
    booking_ids_in_year = guest_filtered[["hotel_id", "ID_booking"]].drop_duplicates()
    booking_filtered = booking_filtered.merge(
        booking_ids_in_year, on=["hotel_id", "ID_booking"], how="inner"
    )

# Фильтр по риску
if selected_risk != "Все уровни":
    risk_en = RISK_MAP_EN[selected_risk]
    booking_filtered = booking_filtered[booking_filtered["risk_level"] == risk_en]

# -----------------------------------------------------------------------------
# КЛЮЧЕВЫЕ МЕТРИКИ
# -----------------------------------------------------------------------------
st.header("📊 Ключевые показатели")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Всего бронирований",
        f"{len(booking_filtered):,}",
        help="Количество уникальных диалогов с гостями"
    )

with col2:
    high_risk = (booking_filtered["risk_level"] == "HIGH").sum()
    high_risk_pct = high_risk / len(booking_filtered) * 100 if len(booking_filtered) > 0 else 0
    st.metric(
        "🚨 Высокий риск",
        f"{high_risk}",
        delta=f"{high_risk_pct:.1f}%",
        delta_color="inverse"
    )

with col3:
    avg_risk = booking_filtered["risk_score"].mean() if len(booking_filtered) > 0 else 0
    st.metric(
        "Средний балл риска",
        f"{avg_risk:.1f}",
        help="Шкала 0-100, выше = опаснее"
    )

with col4:
    neg_pct = (guest_filtered["sentiment"] == "NEG").mean() * 100 if len(guest_filtered) > 0 else 0
    st.metric(
        "😠 Негатив",
        f"{neg_pct:.1f}%",
        help="Доля негативных сообщений"
    )

with col5:
    no_reply = (booking_filtered["reply_time_min"] == -1).mean() * 100 if len(booking_filtered) > 0 else 0
    st.metric(
        "⏰ Без ответа",
        f"{no_reply:.1f}%",
        delta_color="inverse"
    )

# -----------------------------------------------------------------------------
# ОБЪЯСНЕНИЕ РАСЧЕТА РИСКА
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Как рассчитывается балл риска?"):
    st.markdown("""
    ### 📐 Формула расчета риска (0-100 баллов)

    Балл риска складывается из нескольких компонентов:

    | Компонент | Вес | Описание |
    |-----------|-----|----------|
    | **Доля негатива** | 40% | Процент негативных сообщений в диалоге |
    | **Серия негатива** | +10 баллов | За каждое подряд негативное сообщение |
    | **Последнее сообщение** | +15 баллов | Если диалог закончился на негативе |
    | **Время ответа** | до +15 баллов | Чем дольше ждал гость, тем хуже |
    | **Угрозы** | +15 баллов | Если гость угрожает отзывом/судом |
    | **Серьезность** | до +20 баллов | На основе типа жалобы (1-5) |

    ### 🚦 Уровни риска:
    - **НИЗКИЙ (0-30)**: Обычный диалог, всё в порядке
    - **СРЕДНИЙ (30-60)**: Есть проблемы, требует внимания
    - **ВЫСОКИЙ (60-100)**: Критично! Возможен негативный отзыв

    ### 💡 Пример:
    > Гость написал 5 сообщений, из них 3 негативных (60% негатива = 24 балла).
    > Последние 2 сообщения негативные подряд (+20 баллов).
    > Администратор ответил через 2 часа (+15 баллов).
    > Гость угрожает написать отзыв (+15 баллов).
    > **Итого: 74 балла = ВЫСОКИЙ РИСК** 🚨
    """)

# -----------------------------------------------------------------------------
# ГРАФИКИ - ПЕРВЫЙ РЯД
# -----------------------------------------------------------------------------
st.header("📈 Аналитика")

col1, col2 = st.columns(2)

with col1:
    # Распределение риска
    if len(booking_filtered) > 0:
        risk_counts = booking_filtered["risk_level"].map(RISK_MAP_RU).value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Распределение по уровню риска",
            color=risk_counts.index,
            color_discrete_map={
                "ВЫСОКИЙ": "#ff4444",
                "СРЕДНИЙ": "#ffaa00",
                "НИЗКИЙ": "#44bb44"
            },
            hole=0.4
        )
        fig_risk.update_traces(textinfo='percent+value')
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with col2:
    # Тональность сообщений
    if len(guest_filtered) > 0:
        sent_map = {"NEG": "Негатив", "NEU": "Нейтрал", "POS": "Позитив"}
        sent_counts = guest_filtered["sentiment"].map(sent_map).value_counts()
        fig_sent = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            title="Тональность сообщений гостей",
            color=sent_counts.index,
            color_discrete_map={
                "Негатив": "#ff4444",
                "Нейтрал": "#888888",
                "Позитив": "#44bb44"
            },
            hole=0.4
        )
        fig_sent.update_traces(textinfo='percent+value')
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

# -----------------------------------------------------------------------------
# ГРАФИКИ - ВТОРОЙ РЯД
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    # Типы сообщений
    if len(guest_filtered) > 0:
        type_map = {
            "THREAT": "Угрозы",
            "COMPLAINT": "Жалобы",
            "PROBLEM": "Проблемы",
            "QUESTION": "Вопросы",
            "PRAISE": "Похвала",
            "OTHER": "Прочее"
        }
        type_counts = guest_filtered["msg_type"].map(type_map).fillna("Прочее").value_counts()
        fig_type = px.bar(
            x=type_counts.values,
            y=type_counts.index,
            orientation='h',
            title="Типы сообщений",
            labels={"x": "Количество", "y": "Тип"},
            color=type_counts.index,
            color_discrete_map={
                "Угрозы": "#ff0000",
                "Жалобы": "#ff6600",
                "Проблемы": "#ffcc00",
                "Вопросы": "#6699ff",
                "Похвала": "#44bb44",
                "Прочее": "#cccccc"
            }
        )
        fig_type.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with col2:
    # Темы обращений
    if len(guest_filtered) > 0 and "topic" in guest_filtered.columns:
        topic_counts = guest_filtered["topic"].value_counts().head(8)
        fig_topic = px.bar(
            x=topic_counts.values,
            y=topic_counts.index,
            orientation='h',
            title="Топ тем обращений",
            labels={"x": "Количество", "y": "Тема"},
            color=topic_counts.values,
            color_continuous_scale="Blues"
        )
        fig_topic.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_topic, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

# -----------------------------------------------------------------------------
# ДИНАМИКА ПО ВРЕМЕНИ
# -----------------------------------------------------------------------------
if "year_month" in guest_filtered.columns and len(guest_filtered) > 0:
    st.header("📅 Динамика по месяцам")

    # Агрегация по месяцам
    monthly = guest_filtered.groupby("year_month").agg(
        total=("message", "size"),
        neg_count=("sentiment", lambda x: (x == "NEG").sum())
    ).reset_index()
    monthly["neg_pct"] = (monthly["neg_count"] / monthly["total"] * 100).round(1)

    # График
    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x=monthly["year_month"],
        y=monthly["neg_pct"],
        mode='lines+markers',
        name='% негатива',
        line=dict(color='#ff4444', width=3),
        marker=dict(size=8)
    ))

    fig_trend.add_trace(go.Bar(
        x=monthly["year_month"],
        y=monthly["total"],
        name='Всего сообщений',
        marker_color='#4a90d9',
        opacity=0.3,
        yaxis='y2'
    ))

    fig_trend.update_layout(
        title="Динамика негатива по месяцам",
        xaxis_title="Месяц",
        yaxis=dict(title="% негатива", side="left", ticksuffix="%"),
        yaxis2=dict(title="Кол-во сообщений", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------------------------------------------
# SLA: ВРЕМЯ ОТВЕТА vs РИСК
# -----------------------------------------------------------------------------
st.header("⏱️ Время ответа и риск")

col1, col2 = st.columns(2)

with col1:
    if len(sla_view) > 0:
        sla_view_ru = sla_view.copy()
        sla_view_ru["reply_bucket"] = sla_view_ru["reply_bucket"].replace({
            "<=5m": "≤5 мин",
            "5-15m": "5-15 мин",
            "15-60m": "15-60 мин",
            ">60m": ">60 мин"
        })

        fig_sla = px.bar(
            sla_view_ru,
            x="reply_bucket",
            y="avg_risk",
            color="avg_risk",
            color_continuous_scale="RdYlGn_r",
            title="Связь времени ответа и риска",
            labels={"reply_bucket": "Время ответа", "avg_risk": "Средний риск"},
            text="n_bookings"
        )
        fig_sla.update_traces(texttemplate='%{text} брон.', textposition='outside')
        fig_sla.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sla, use_container_width=True)
    else:
        st.info("Нет данных SLA")

with col2:
    st.markdown("""
    ### 💡 Выводы по SLA

    **Чем быстрее ответ — тем ниже риск!**

    Рекомендации:
    - ⚡ **≤5 минут** — идеально для срочных проблем
    - ✅ **5-15 минут** — приемлемо для большинства запросов
    - ⚠️ **15-60 минут** — риск эскалации растет
    - 🚨 **>60 минут** — высокая вероятность негативного отзыва

    **Цель:** Ответить на 80% обращений в течение 15 минут.
    """)

# -----------------------------------------------------------------------------
# СРАВНЕНИЕ ОТЕЛЕЙ
# -----------------------------------------------------------------------------
if selected_hotel_name == "Все отели":
    st.header("🏨 Сравнение отелей")

    hotel_summary_display = hotel_summary.copy()
    hotel_summary_display = hotel_summary_display.rename(columns={
        "hotel_name": "Отель",
        "n_bookings": "Бронирований",
        "avg_risk": "Ср. риск",
        "high_risk_share_%": "Выс. риск %",
        "median_reply_min": "Медиана ответа (мин)",
        "avg_neg_share": "Ср. негатив %",
        "threats": "Угрозы",
        "complaints": "Жалобы",
        "problems": "Проблемы"
    })

    # Таблица
    st.dataframe(
        hotel_summary_display[["Отель", "Бронирований", "Ср. риск", "Выс. риск %",
                               "Медиана ответа (мин)", "Угрозы", "Жалобы"]],
        use_container_width=True,
        hide_index=True
    )

    # График сравнения
    fig_compare = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Средний риск", "% высокого риска", "Медиана ответа (мин)")
    )

    colors = px.colors.qualitative.Set2

    fig_compare.add_trace(
        go.Bar(x=hotel_summary["hotel_name"], y=hotel_summary["avg_risk"],
               marker_color=colors[0], name="Риск"),
        row=1, col=1
    )

    fig_compare.add_trace(
        go.Bar(x=hotel_summary["hotel_name"], y=hotel_summary["high_risk_share_%"],
               marker_color=colors[1], name="Выс. риск %"),
        row=1, col=2
    )

    fig_compare.add_trace(
        go.Bar(x=hotel_summary["hotel_name"], y=hotel_summary["median_reply_min"],
               marker_color=colors[2], name="Ответ"),
        row=1, col=3
    )

    fig_compare.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

# -----------------------------------------------------------------------------
# ОЧЕРЕДЬ РИСКОВЫХ БРОНИРОВАНИЙ
# -----------------------------------------------------------------------------
st.header("🚨 Бронирования, требующие внимания")

# Фильтруем топ рисковые
if len(booking_filtered) > 0:
    risky_bookings = booking_filtered.nlargest(15, "risk_score")

    for _, row in risky_bookings.iterrows():
        risk_level_ru = RISK_MAP_RU.get(row["risk_level"], "?")
        risk_color = {"ВЫСОКИЙ": "🔴", "СРЕДНИЙ": "🟡", "НИЗКИЙ": "🟢"}.get(risk_level_ru, "⚪")

        hotel_name = HOTEL_NAMES.get(row["hotel_id"], f"Отель {row['hotel_id']}")

        with st.expander(f"{risk_color} {hotel_name} | Бронь #{int(row['ID_booking'])} | Риск: {row['risk_score']:.0f}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Уровень риска:** {risk_level_ru}")
                st.write(f"**Негатив:** {row['neg_share']:.1f}%")
                st.write(f"**Серия NEG:** {int(row['neg_streak'])}")

            with col2:
                reply = row['reply_time_min']
                if reply == -1:
                    st.write("**Время ответа:** ❌ Нет ответа")
                else:
                    st.write(f"**Время ответа:** {reply:.0f} мин")

                threat_count = row.get('THREAT', 0)
                complaint_count = row.get('COMPLAINT', 0)
                st.write(f"**Угрозы:** {int(threat_count)}")
                st.write(f"**Жалобы:** {int(complaint_count)}")

            with col3:
                top_topic = row.get('top_topic', 'неизвестно')
                st.write(f"**Тема:** {top_topic}")
                st.write(f"**Последн. тональность:** {row['last_sentiment']}")

            # Показываем сообщения из этого бронирования
            booking_msgs = guest_filtered[
                (guest_filtered["hotel_id"] == row["hotel_id"]) &
                (guest_filtered["ID_booking"] == row["ID_booking"])
            ].tail(5)

            if len(booking_msgs) > 0:
                st.write("**Последние сообщения:**")
                for _, msg in booking_msgs.iterrows():
                    sent_emoji = {"NEG": "😠", "NEU": "😐", "POS": "😊"}.get(msg["sentiment"], "❓")
                    msg_text = str(msg["message"])[:150]
                    st.write(f"{sent_emoji} {msg_text}...")
else:
    st.success("✅ Нет бронирований с высоким риском!")

# -----------------------------------------------------------------------------
# СНИЖЕНИЕ ЛОЖНЫХ СРАБАТЫВАНИЙ
# -----------------------------------------------------------------------------
st.header("🎯 Эффективность системы")

col1, col2 = st.columns(2)

with col1:
    # Сравнение подходов
    if len(guest_filtered) > 0:
        neg_only = (guest_filtered["sentiment"] == "NEG").sum()
        high_risk = (
            (guest_filtered["msg_type"].isin(["THREAT", "COMPLAINT"])) &
            (guest_filtered["sentiment"] == "NEG")
        ).sum()

        fp_data = pd.DataFrame({
            "Подход": ["Только тональность\n(старый метод)", "Тональность + тип\n(новый метод)"],
            "Отмечено сообщений": [neg_only, high_risk]
        })

        fig_fp = px.bar(
            fp_data,
            x="Подход",
            y="Отмечено сообщений",
            color="Подход",
            color_discrete_sequence=["#ff4444", "#44bb44"],
            title="Снижение ложных срабатываний",
            text="Отмечено сообщений"
        )
        fig_fp.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_fp.update_layout(showlegend=False)

        if neg_only > 0:
            reduction_pct = (neg_only - high_risk) / neg_only * 100
            fig_fp.add_annotation(
                x=0.5, y=max(fp_data["Отмечено сообщений"]) * 0.6,
                text=f"📉 На {reduction_pct:.0f}% меньше\nложных тревог!",
                showarrow=False,
                font=dict(size=16, color="green")
            )

        st.plotly_chart(fig_fp, use_container_width=True)

with col2:
    st.markdown("""
    ### 💡 Почему это важно?

    **Старый подход** (только тональность):
    - "Сушилка не работает" → 😠 Негатив → ⚠️ Тревога!
    - Но это просто **сообщение о проблеме**, не жалоба

    **Новый подход** (тональность + тип сообщения):
    - "Сушилка не работает" → 😠 Негатив + 🔧 Проблема → ✅ Норма
    - "Напишу негативный отзыв!" → 😠 Негатив + ⚠️ Угроза → 🚨 Тревога!

    **Результат:**
    - Меньше ложных тревог
    - Фокус на реальных проблемах
    - Экономия времени менеджера
    """)

# -----------------------------------------------------------------------------
# ФУТЕР
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏨 Система анализа коммуникации с гостями | v1.0</p>
    <p>Данные обрабатываются локально | HuggingFace BERT + Ключевые слова</p>
</div>
""", unsafe_allow_html=True)

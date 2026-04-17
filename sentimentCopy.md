```python
import pandas as pd
import numpy as np

path = "/home/nikita/code/PlatoIsDead/notebook_prototype/sentiment_analysis/Сообщения Vertical (5).xlsx"
df = pd.read_excel(path)

df["date_add"] = pd.to_datetime(df["date_add"])
df["is_admin"] = df["from_admin"].fillna(0).astype(int)

# Focus MVP on guest messages only
guest = df[df["is_admin"] == 0].copy()

# Basic hygiene
guest["message"] = guest["message"].astype(str).str.strip()
guest = guest[guest["message"].str.len() > 0]

# Booking-level aggregation scaffold (you’ll join sentiment labels later)
booking = guest.groupby(["hotel_id", "ID_booking"]).agg(
    n_guest_msgs=("message", "size"),
    first_msg=("date_add", "min"),
    last_msg=("date_add", "max"),
).reset_index()

booking["duration_hr"] = (booking["last_msg"] - booking["first_msg"]).dt.total_seconds() / 3600
booking.head()

```

**Output:**
```
   hotel_id  ID_booking  n_guest_msgs           first_msg            last_msg  \
0         1       29204            37 2023-10-05 14:23:16 2024-09-11 14:41:12   
1         1       55489             4 2023-09-21 15:11:36 2023-09-24 15:03:24   
2         1       68575             7 2023-12-21 00:08:05 2024-01-15 00:44:24   
3         1       68934             2 2023-10-17 15:30:15 2023-10-17 16:11:34   
4         1       69417             8 2023-09-19 10:25:43 2023-09-30 20:30:47   

   duration_hr  
0  8208.298889  
1    71.863333  
2   600.605278  
3     0.688611  
4   274.084444  
```

```python
guest = guest.sort_values(
    ["hotel_id", "ID_booking", "date_add"]
).reset_index(drop=True)
```

```python
import os, json
from openai import OpenAI
from tqdm.auto import tqdm

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = """
You label hotel GUEST messages with sentiment.
Return ONLY JSON.
Sentiment: NEG, NEU, POS.
Confidence: 0..1.
Output must be a JSON array with same length/order as inputs.
""".strip()

def classify_batch(texts):
    # Keep prompts small-ish
    items = [{"i": i, "text": t} for i, t in enumerate(texts)]
    user = json.dumps(items, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Input messages as JSON list:\n{user}\n\n"
                "Return JSON object with key 'out' = array of "
                "{sentiment, confidence} in same order."
            },
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return data["out"]

```

```python
BATCH_SIZE = 25

texts = guest["message"].astype(str).tolist()
sent, conf = [], []

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    out = classify_batch(batch)
    sent.extend([x["sentiment"] for x in out])
    conf.extend([x.get("confidence") for x in out])
    if i % (BATCH_SIZE*50) == 0 and i > 0:
        pd.DataFrame({"sentiment": sent, "sentiment_score": conf}).to_parquet(
        "openai_partial.parquet", index=False
    )


guest["sentiment"] = sent
guest["sentiment_score"] = conf

```

```python
#
# STEP 1: SENTIMENT ANALYSIS (you already have this)
# Based on: Le Wagon Deep Learning → Transformers
#

# Assuming guest["sentiment"] and guest["sentiment_score"] already exist
# If not, run your HuggingFace model code here

# Quick check
print("=== Sentiment Distribution (Guest Messages) ===")
print(guest["sentiment"].value_counts())
print()
print((guest["sentiment"].value_counts(normalize=True) * 100).round(1))
```

```
=== Sentiment Distribution (Guest Messages) ===
sentiment
NEU    36529
NEG    10909
POS     8977
Name: count, dtype: int64

sentiment
NEU    64.8
NEG    19.3
POS    15.9
Name: proportion, dtype: float64

```

```python
# STEP 1: SENTIMENT ANALYSIS (you already have this)
# Based on: Le Wagon Deep Learning → Transformers
#

# Assuming guest["sentiment"] and guest["sentiment_score"] already exist
# If not, run your HuggingFace model code here

# Quick check
print("=== Sentiment Distribution (Guest Messages) ===")
print(guest["sentiment"].value_counts())
print()
print((guest["sentiment"].value_counts(normalize=True) * 100).round(1))
```

```
=== Sentiment Distribution (Guest Messages) ===
sentiment
NEU    36529
NEG    10909
POS     8977
Name: count, dtype: int64

sentiment
NEU    64.8
NEG    19.3
POS    15.9
Name: proportion, dtype: float64

```

```python
# STEP 2: MESSAGE TYPE CLASSIFICATION
# Based on: Le Wagon NLP → Keyword extraction


# Keywords for classification
COMPLAINT_KEYWORDS = [
    "возмущен", "безобразие", "хамство", "хамят",
    "кошмар", "ужас", "отвратительн", "недопустим",
    "обман", "наглость", "хватит", "достали", "надоело"
]

THREAT_KEYWORDS = [
    r"\суд\bb", "роспотребнадзор", "прокуратур", "адвокат",
    "напишу отзыв", "оставлю отзыв", "верну деньги"
]

PROBLEM_KEYWORDS = [
    "не работает", "сломан", "не включается", "течет",
    "воняет", "грязн", "холодно", "жарко", "шум"
]

import re

def classify_message(row) -> str:
    """Classify message using keywords + sentiment."""
    text = str(row["message"]).lower()
    sentiment = row["sentiment"]

    # THREAT - highest priority
    for pattern in THREAT_KEYWORDS:
        if re.search(pattern, text):
            return "THREAT"

    # COMPLAINT - strong negative language
    for kw in COMPLAINT_KEYWORDS:
        if kw in text:
            return "COMPLAINT"

    # COMPLAINT - NEG sentiment + frustration words
    if sentiment == "NEG" and any(w in text for w in ["снова", "опять", "уже", "сколько раз"]):
        return "COMPLAINT"

    # PROBLEM_REPORT
    for kw in PROBLEM_KEYWORDS:
        if kw in text:
            return "PROBLEM"

    # POSITIVE
    if sentiment == "POS" or any(w in text for w in ["спасибо", "благодар", "отлично"]):
        return "POSITIVE"

    return "NEUTRAL"

guest["msg_type"] = guest.apply(classify_message, axis=1)

print("=== Message Type Distribution ===")
print(guest["msg_type"].value_counts())
```

```
=== Message Type Distribution ===
msg_type
NEUTRAL      43381
POSITIVE      9884
PROBLEM       1732
COMPLAINT     1410
THREAT           8
Name: count, dtype: int64

```

```python
# STEP 3: BOOKING-LEVEL AGGREGATION
# Based on: Le Wagon Data Toolkit → GroupBy

# Sort by time
guest_sorted = guest.sort_values(["hotel_id", "ID_booking", "date_add"])

# Aggregate per booking
booking_stats = guest_sorted.groupby(["hotel_id", "ID_booking"]).agg(
    n_guest_msgs=("message", "size"),
    first_msg=("date_add", "min"),
    last_msg=("date_add", "max"),

    # Sentiment counts
    neg_count=("sentiment", lambda x: (x == "NEG").sum()),
    pos_count=("sentiment", lambda x: (x == "POS").sum()),

    # Message type counts
    complaint_count=("msg_type", lambda x: (x == "COMPLAINT").sum()),
    threat_count=("msg_type", lambda x: (x == "THREAT").sum()),
    problem_count=("msg_type", lambda x: (x == "PROBLEM").sum()),

    # Last message info
    last_sentiment=("sentiment", "last"),
    last_msg_type=("msg_type", "last")
).reset_index()

# Calculate percentages
booking_stats["neg_share"] = (booking_stats["neg_count"] / booking_stats["n_guest_msgs"] * 100).round(1)

# Conversation duration
booking_stats["duration_hours"] = (
    (booking_stats["last_msg"] - booking_stats["first_msg"]).dt.total_seconds() / 3600
).round(1)

booking_stats.head()
```

**Output:**
```
   hotel_id  ID_booking  n_guest_msgs           first_msg            last_msg  \
0         1       29204            37 2023-10-05 14:23:16 2024-09-11 14:41:12   
1         1       55489             4 2023-09-21 15:11:36 2023-09-24 15:03:24   
2         1       68575             7 2023-12-21 00:08:05 2024-01-15 00:44:24   
3         1       68934             2 2023-10-17 15:30:15 2023-10-17 16:11:34   
4         1       69417             8 2023-09-19 10:25:43 2023-09-30 20:30:47   

   neg_count  pos_count  complaint_count  threat_count  problem_count  \
0          1          1                0             0              0   
1          2          0                1             0              0   
2          1          3                0             0              1   
3          0          1                0             0              0   
4          1          1                1             0              0   

  last_sentiment last_msg_type  neg_share  duration_hours  
0            NEU       NEUTRAL        2.7          8208.3  
1            NEU       NEUTRAL       50.0            71.9  
2            NEU       NEUTRAL       14.3           600.6  
3            POS      POSITIVE        0.0             0.7  
4            NEU       NEUTRAL       12.5           274.1  
```

```python
# Based on: Le Wagon Data Toolkit → Merge operations


# First guest message per booking
first_guest = (
    guest_sorted.groupby(["hotel_id", "ID_booking"])["date_add"]
    .min()
    .reset_index()
    .rename(columns={"date_add": "first_guest_time"})
)

# First admin message per booking
first_admin = (
    admin.groupby(["hotel_id", "ID_booking"])["date_add"]
    .min()
    .reset_index()
    .rename(columns={"date_add": "first_admin_time"})
)

# Admin message count per booking
admin_counts = (
    admin.groupby(["hotel_id", "ID_booking"])
    .size()
    .reset_index(name="n_admin_msgs")
)

# Merge all
booking_stats = booking_stats.merge(first_guest, on=["hotel_id", "ID_booking"], how="left")
booking_stats = booking_stats.merge(first_admin, on=["hotel_id", "ID_booking"], how="left")
booking_stats = booking_stats.merge(admin_counts, on=["hotel_id", "ID_booking"], how="left")

# Calculate response time (minutes)
booking_stats["reply_time_min"] = (
    (booking_stats["first_admin_time"] - booking_stats["first_guest_time"])
    .dt.total_seconds() / 60
).round(1)

# Handle edge cases
booking_stats["n_admin_msgs"] = booking_stats["n_admin_msgs"].fillna(0).astype(int)
booking_stats.loc[booking_stats["reply_time_min"] < 0, "reply_time_min"] = np.nan  # admin wrote first

# Admin responded? (boolean)
booking_stats["admin_responded"] = booking_stats["n_admin_msgs"] > 0

# Admin-to-guest ratio
booking_stats["admin_guest_ratio"] = (
    booking_stats["n_admin_msgs"] / booking_stats["n_guest_msgs"]
).round(2)

print("=== Admin Response Stats ===")
print(f"Bookings with admin response: {booking_stats['admin_responded'].sum()} / {len(booking_stats)}")
print(f"Average response time: {booking_stats['reply_time_min'].mean():.1f} min")
print(f"Median response time: {booking_stats['reply_time_min'].median():.1f} min")
```

```
=== Admin Response Stats ===
Bookings with admin response: 6695 / 7982
Average response time: 878.2 min
Median response time: 7.8 min

```

```python
# STEP 5: RISK SCORING
# Based on: Le Wagon ML → Feature Engineering


def calculate_risk_score(row) -> float:
    """
    Calculate booking risk score (0-100).
    Higher = more attention needed.
    """
    score = 0

    # THREAT = immediate high risk
    score += row["threat_count"] * 50

    # COMPLAINT = significant risk
    score += row["complaint_count"] * 15

    # High NEG share (only if enough messages to be meaningful)
    if row["n_guest_msgs"] >= 5 and row["neg_share"] > 50:
        score += 20

    # Conversation ended badly
    if row["last_msg_type"] in ["COMPLAINT", "THREAT"]:
        score += 15
    if row["last_sentiment"] == "NEG" and row["n_guest_msgs"] >= 3:
        score += 10

    # No admin response = bad
    if not row["admin_responded"]:
        score += 15

    # Slow response (>60 min) = minor penalty
    elif pd.notna(row["reply_time_min"]) and row["reply_time_min"] > 60:
        score += 10

    # REDUCE risk if resolved (ended positive)
    if row["last_sentiment"] == "POS":
        score -= 20

    return min(max(score, 0), 100)  # Clip to 0-100

booking_stats["risk_score"] = booking_stats.apply(calculate_risk_score, axis=1)

# Risk levels
booking_stats["risk_level"] = pd.cut(
    booking_stats["risk_score"],
    bins=[-1, 20, 50, 100],
    labels=["LOW", "MEDIUM", "HIGH"]
)

print("=== Risk Distribution ===")
print(booking_stats["risk_level"].value_counts())
```

```
=== Risk Distribution ===
risk_level
LOW       7464
MEDIUM     419
HIGH        99
Name: count, dtype: int64

```

```python

# STEP 6: HOTEL-LEVEL KPIs
# Based on: Le Wagon Data Toolkit → Aggregation


hotel_kpis = booking_stats.groupby("hotel_id").agg(
    total_bookings=("ID_booking", "nunique"),
    total_guest_msgs=("n_guest_msgs", "sum"),
    total_admin_msgs=("n_admin_msgs", "sum"),

    # Response metrics
    response_rate=("admin_responded", "mean"),
    avg_reply_time_min=("reply_time_min", "mean"),
    median_reply_time_min=("reply_time_min", "median"),

    # Risk metrics
    high_risk_bookings=("risk_level", lambda x: (x == "HIGH").sum()),
    medium_risk_bookings=("risk_level", lambda x: (x == "MEDIUM").sum()),

    # Sentiment metrics
    avg_neg_share=("neg_share", "mean"),
    total_complaints=("complaint_count", "sum"),
    total_threats=("threat_count", "sum")
).reset_index()

# Calculate percentages
hotel_kpis["response_rate"] = (hotel_kpis["response_rate"] * 100).round(1)
hotel_kpis["high_risk_pct"] = (hotel_kpis["high_risk_bookings"] / hotel_kpis["total_bookings"] * 100).round(1)

# Format
hotel_kpis = hotel_kpis.round(1)

print("=== HOTEL KPIs ===")
hotel_kpis
```

```
=== HOTEL KPIs ===

```

**Output:**
```
   hotel_id  total_bookings  total_guest_msgs  total_admin_msgs  \
0         1            2488             12205              6917   
1         2            1009              3056              1878   
2         3             813              2953              2005   
3         4             227               593               298   
4         5            3445             37608             22247   

   response_rate  avg_reply_time_min  median_reply_time_min  \
0           81.9              1201.9                   19.6   
1           80.2               186.1                    5.4   
2           69.6              1604.2                   15.3   
3           69.2               391.0                   21.4   
4           90.7               734.8                    4.9   

   high_risk_bookings  medium_risk_bookings  avg_neg_share  total_complaints  \
0                  28                   135           19.5               376   
1                   1                    30           19.5                49   
2                   0                    16           29.2                 2   
3                   0                     2           14.3                 3   
4                  70                   236           17.2               980   

   total_threats  high_risk_pct  
0              2            1.1  
1              1            0.1  
2              0            0.0  
3              0            0.0  
4              5            2.0  
```

```python
# STEP 7: TASK QUEUE (Actionable output)
# What bookings need attention TODAY?

# Filter to HIGH and MEDIUM risk, with recent activity
today = pd.Timestamp.now()
recent_cutoff = today - pd.Timedelta(days=7)

task_queue = booking_stats[
    (booking_stats["risk_level"].isin(["HIGH", "MEDIUM"])) &
    (booking_stats["last_msg"] >= recent_cutoff)
].sort_values("risk_score", ascending=False)

print(f"=== TASK QUEUE: {len(task_queue)} bookings need review ===")
task_queue[["hotel_id", "ID_booking", "risk_score", "risk_level",
            "complaint_count", "threat_count", "admin_responded",
            "last_sentiment", "n_guest_msgs"]].head(20)
```

```
=== TASK QUEUE: 0 bookings need review ===

```

**Output:**
```
Empty DataFrame
Columns: [hotel_id, ID_booking, risk_score, risk_level, complaint_count, threat_count, admin_responded, last_sentiment, n_guest_msgs]
Index: []
```

```python

# STEP 8: YEAR-OVER-YEAR TRENDS
# Based on: Le Wagon Data Toolkit → Time series grouping

guest["year"] = guest["date_add"].dt.year
guest["month"] = guest["date_add"].dt.to_period("M")

# Sentiment by hotel + year
sentiment_trend = (
    guest.groupby(["hotel_id", "year"])["sentiment"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure columns exist
for col in ["NEG", "NEU", "POS"]:
    if col not in sentiment_trend.columns:
        sentiment_trend[col] = 0.0

# Add message counts
msg_counts = guest.groupby(["hotel_id", "year"]).size().reset_index(name="n_messages")
sentiment_trend = sentiment_trend.merge(msg_counts, on=["hotel_id", "year"])

# Format as percentages
sentiment_trend[["NEG", "NEU", "POS"]] = (sentiment_trend[["NEG", "NEU", "POS"]] * 100).round(1)

print("=== Sentiment % by Hotel + Year ===")
sentiment_trend[["hotel_id", "year", "n_messages", "NEG", "NEU", "POS"]]
```

```
=== Sentiment % by Hotel + Year ===

```

**Output:**
```
    hotel_id  year  n_messages   NEG   NEU   POS
0          1  2023        3509  20.5  62.5  17.0
1          1  2024        6172  21.6  64.6  13.7
2          1  2025        2524  20.6  64.1  15.3
3          2  2023         555  15.5  71.7  12.8
4          2  2024        1671  17.0  67.6  15.4
5          2  2025         830  18.7  64.6  16.7
6          3  2023         413  25.4  54.0  20.6
7          3  2024        1449  29.5  46.9  23.6
8          3  2025        1091  23.3  50.9  25.8
9          4  2023         117  14.5  70.9  14.5
10         4  2024         331  14.2  73.4  12.4
11         4  2025         145  17.2  69.7  13.1
12         5  2023        3710  17.0  65.0  18.0
13         5  2024       10938  18.4  66.5  15.1
14         5  2025       22960  18.7  65.8  15.6
```

```python

# STEP 8: YEAR-OVER-YEAR TRENDS
# Based on: Le Wagon Data Toolkit → Time series grouping


guest["year"] = guest["date_add"].dt.year
guest["month"] = guest["date_add"].dt.to_period("M")

# Sentiment by hotel + year
sentiment_trend = (
    guest.groupby(["hotel_id", "year"])["sentiment"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure columns exist
for col in ["NEG", "NEU", "POS"]:
    if col not in sentiment_trend.columns:
        sentiment_trend[col] = 0.0

# Add message counts
msg_counts = guest.groupby(["hotel_id", "year"]).size().reset_index(name="n_messages")
sentiment_trend = sentiment_trend.merge(msg_counts, on=["hotel_id", "year"])

# Format as percentages
sentiment_trend[["NEG", "NEU", "POS"]] = (sentiment_trend[["NEG", "NEU", "POS"]] * 100).round(1)

print("=== Sentiment % by Hotel + Year ===")
sentiment_trend[["hotel_id", "year", "n_messages", "NEG", "NEU", "POS"]]
```

```
=== Sentiment % by Hotel + Year ===

```

**Output:**
```
    hotel_id  year  n_messages   NEG   NEU   POS
0          1  2023        3509  20.5  62.5  17.0
1          1  2024        6172  21.6  64.6  13.7
2          1  2025        2524  20.6  64.1  15.3
3          2  2023         555  15.5  71.7  12.8
4          2  2024        1671  17.0  67.6  15.4
5          2  2025         830  18.7  64.6  16.7
6          3  2023         413  25.4  54.0  20.6
7          3  2024        1449  29.5  46.9  23.6
8          3  2025        1091  23.3  50.9  25.8
9          4  2023         117  14.5  70.9  14.5
10         4  2024         331  14.2  73.4  12.4
11         4  2025         145  17.2  69.7  13.1
12         5  2023        3710  17.0  65.0  18.0
13         5  2024       10938  18.4  66.5  15.1
14         5  2025       22960  18.7  65.8  15.6
```


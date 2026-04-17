```python
import pandas as pd
import numpy as np

path = "Сообщения Vertical (5).xlsx"
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
guest = df[df["is_admin"] == 0].copy()

```

```python
guest["sentiment"] = None      # NEG / NEU / POS
guest["sentiment_score"] = None  # optional: -1..1 or 0..1
```

```python
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"  # recommended for RU :contentReference[oaicite:2]{index=2}

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# label mapping depends on model config; we'll read it safely:
id2label = model.config.id2label  # e.g. {0:'negative',1:'neutral',2:'positive'}

def predict_sentiment_batch(texts, max_len=256):
    # tokenize batch
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    pred_ids = probs.argmax(axis=1)
    pred_labels = [id2label[i].lower() for i in pred_ids]
    conf = probs.max(axis=1)

    # map to your NEG/NEU/POS
    map3 = {
        "negative": "NEG",
        "neutral": "NEU",
        "positive": "POS",
    }
    sentiments = [map3.get(lbl, "NEU") for lbl in pred_labels]
    return sentiments, conf
from tqdm.auto import tqdm

guest["sentiment"] = None
guest["sentiment_score"] = None

texts = guest["message"].astype(str).tolist()

batch_size = 64 if device == "cuda" else 16

all_sent = []
all_conf = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    s, c = predict_sentiment_batch(batch)
    all_sent.extend(s)
    all_conf.extend(c)

guest["sentiment"] = all_sent
guest["sentiment_score"] = all_conf

guest[["message","sentiment","sentiment_score"]].sample(10, random_state=42)


hotel_sentiment = (
    guest.groupby(["hotel_id", "sentiment"])
        .size()
        .unstack(fill_value=0)
)
hotel_sentiment

booking_sentiment = (
    guest.groupby(["hotel_id","ID_booking"])["sentiment"]
         .value_counts(normalize=True)
         .unstack(fill_value=0)
         .reset_index()
)

# top bookings by NEG share (and min number of guest messages)
top_bad = booking_sentiment.merge(
    guest.groupby(["hotel_id","ID_booking"]).size().rename("n_msgs").reset_index(),
    on=["hotel_id","ID_booking"],
    how="left"
)

top_bad = top_bad[top_bad["n_msgs"] >= 3].sort_values("NEG", ascending=False)
top_bad.head(20)


guest["year"] = guest["date_add"].dt.year

guest[["date_add", "year"]].sample(5, random_state=42)

guest.columns[guest.columns.duplicated()].tolist(), guest.index.names

hotel_sentiment_pct = (
    guest.groupby("hotel_id")["sentiment"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

hotel_sentiment_pct
```

```
2026-01-23 17:33:27.704164: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-23 17:33:28.418758: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-01-23 17:33:28.689526: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-01-23 17:33:28.691707: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-01-23 17:33:29.183509: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-01-23 17:33:37.780063: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

```

**Output:**
```
sentiment  hotel_id       NEG       NEU       POS
0                 1  0.211061  0.638918  0.150020
1                 2  0.171793  0.675393  0.152814
2                 3  0.266170  0.493735  0.240095
3                 4  0.150084  0.720067  0.129848
4                 5  0.184349  0.658956  0.156695
```

```python
# -----------------------------------------------------------------------------
# FEATURE 1: neg_share — % of NEG guest messages per booking
# -----------------------------------------------------------------------------

booking_sentiment = (
    guest.groupby(["hotel_id", "ID_booking"])["sentiment"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure columns exist
for col in ["NEG", "NEU", "POS"]:
    if col not in booking_sentiment.columns:
        booking_sentiment[col] = 0.0

booking_sentiment["neg_share"] = (booking_sentiment["NEG"] * 100).round(1)

booking_sentiment.head()
```

**Output:**
```
sentiment  hotel_id  ID_booking       NEG       NEU       POS  neg_share
0                 1       29204  0.027027  0.945946  0.027027        2.7
1                 1       55489  0.500000  0.500000  0.000000       50.0
2                 1       68575  0.142857  0.428571  0.428571       14.3
3                 1       68934  0.000000  0.500000  0.500000        0.0
4                 1       69417  0.125000  0.750000  0.125000       12.5
```

```python
# -----------------------------------------------------------------------------
# FEATURE 2: neg_streak — longest consecutive NEG run
# -----------------------------------------------------------------------------

def calc_neg_streak(sentiments: list) -> int:
    """Calculate longest consecutive NEG streak."""
    max_streak = 0
    current = 0
    for s in sentiments:
        if s == "NEG":
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

# Get sentiment sequence per booking (ordered by time)
guest_sorted = guest.sort_values(["hotel_id", "ID_booking", "date_add"])

streak_df = (
    guest_sorted.groupby(["hotel_id", "ID_booking"])["sentiment"]
    .apply(list)
    .reset_index()
)
streak_df["neg_streak"] = streak_df["sentiment"].apply(calc_neg_streak)
streak_df = streak_df.drop(columns=["sentiment"])

streak_df.head()
```

**Output:**
```
   hotel_id  ID_booking  neg_streak
0         1       29204           1
1         1       55489           1
2         1       68575           1
3         1       68934           0
4         1       69417           1
```

```python
# -----------------------------------------------------------------------------
# FEATURE 3: n_msgs — number of guest messages
# FEATURE 4: last_sentiment — how conversation ended
# -----------------------------------------------------------------------------

msg_features = (
    guest_sorted.groupby(["hotel_id", "ID_booking"])
    .agg(
        n_guest_msgs=("message", "size"),
        last_sentiment=("sentiment", "last"),
        first_guest_time=("date_add", "min"),
        last_guest_time=("date_add", "max")
    )
    .reset_index()
)

msg_features.head()
```

**Output:**
```
   hotel_id  ID_booking  n_guest_msgs last_sentiment    first_guest_time  \
0         1       29204            37            NEU 2023-10-05 14:23:16   
1         1       55489             4            NEU 2023-09-21 15:11:36   
2         1       68575             7            NEU 2023-12-21 00:08:05   
3         1       68934             2            POS 2023-10-17 15:30:15   
4         1       69417             8            NEU 2023-09-19 10:25:43   

      last_guest_time  
0 2024-09-11 14:41:12  
1 2023-09-24 15:03:24  
2 2024-01-15 00:44:24  
3 2023-10-17 16:11:34  
4 2023-09-30 20:30:47  
```

```python
# -----------------------------------------------------------------------------
# FEATURE 5: time_to_first_admin_reply (minutes)
# -----------------------------------------------------------------------------

# First admin message per booking
first_admin = (
    admin.groupby(["hotel_id", "ID_booking"])["date_add"]
    .min()
    .reset_index()
    .rename(columns={"date_add": "first_admin_time"})
)

# First guest message per booking (already in msg_features)

# Merge and calculate reply time
reply_time = msg_features[["hotel_id", "ID_booking", "first_guest_time"]].merge(
    first_admin,
    on=["hotel_id", "ID_booking"],
    how="left"
)

reply_time["reply_time_min"] = (
    (reply_time["first_admin_time"] - reply_time["first_guest_time"])
    .dt.total_seconds() / 60
).round(1)

# Handle edge cases (negative = admin wrote first, treat as 0)
reply_time.loc[reply_time["reply_time_min"] < 0, "reply_time_min"] = 0

reply_time = reply_time[["hotel_id", "ID_booking", "reply_time_min"]]

reply_time.head()
```

**Error:**
```
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
Cell [0;32mIn[11], line 7[0m
[1;32m      1[0m [38;5;66;03m# -----------------------------------------------------------------------------[39;00m
[1;32m      2[0m [38;5;66;03m# FEATURE 5: time_to_first_admin_reply (minutes)[39;00m
[1;32m      3[0m [38;5;66;03m# -----------------------------------------------------------------------------[39;00m
[1;32m      4[0m 
[1;32m      5[0m [38;5;66;03m# First admin message per booking[39;00m
[1;32m      6[0m first_admin [38;5;241m=[39m (
[0;32m----> 7[0m     [43madmin[49m[38;5;241m.[39mgroupby([[38;5;124m"[39m[38;5;124mhotel_id[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mID_booking[39m[38;5;124m"[39m])[[38;5;124m"[39m[38;5;124mdate_add[39m[38;5;124m"[39m]
[1;32m      8[0m     [38;5;241m.[39mmin()
[1;32m      9[0m     [38;5;241m.[39mreset_index()
[1;32m     10[0m     [38;5;241m.[39mrename(columns[38;5;241m=[39m{[38;5;124m"[39m[38;5;124mdate_add[39m[38;5;124m"[39m: [38;5;124m"[39m[38;5;124mfirst_admin_time[39m[38;5;124m"[39m})
[1;32m     11[0m )
[1;32m     13[0m [38;5;66;03m# First guest message per booking (already in msg_features)[39;00m
[1;32m     14[0m 
[1;32m     15[0m [38;5;66;03m# Merge and calculate reply time[39;00m
[1;32m     16[0m reply_time [38;5;241m=[39m msg_features[[[38;5;124m"[39m[38;5;124mhotel_id[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mID_booking[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mfirst_guest_time[39m[38;5;124m"[39m]][38;5;241m.[39mmerge(
[1;32m     17[0m     first_admin,
[1;32m     18[0m     on[38;5;241m=[39m[[38;5;124m"[39m[38;5;124mhotel_id[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mID_booking[39m[38;5;124m"[39m],
[1;32m     19[0m     how[38;5;241m=[39m[38;5;124m"[39m[38;5;124mleft[39m[38;5;124m"[39m
[1;32m     20[0m )

[0;31mNameError[0m: name 'admin' is not defined
```

```python
# -----------------------------------------------------------------------------
# FEATURE 6: admin_to_guest_ratio
# -----------------------------------------------------------------------------

# Count admin messages per booking
admin_counts = (
    admin.groupby(["hotel_id", "ID_booking"])
    .size()
    .reset_index(name="n_admin_msgs")
)

admin_counts.head()
```

**Error:**
```
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
Cell [0;32mIn[10], line 7[0m
[1;32m      1[0m [38;5;66;03m# -----------------------------------------------------------------------------[39;00m
[1;32m      2[0m [38;5;66;03m# FEATURE 6: admin_to_guest_ratio[39;00m
[1;32m      3[0m [38;5;66;03m# -----------------------------------------------------------------------------[39;00m
[1;32m      4[0m 
[1;32m      5[0m [38;5;66;03m# Count admin messages per booking[39;00m
[1;32m      6[0m admin_counts [38;5;241m=[39m (
[0;32m----> 7[0m     [43madmin[49m[38;5;241m.[39mgroupby([[38;5;124m"[39m[38;5;124mhotel_id[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mID_booking[39m[38;5;124m"[39m])
[1;32m      8[0m     [38;5;241m.[39msize()
[1;32m      9[0m     [38;5;241m.[39mreset_index(name[38;5;241m=[39m[38;5;124m"[39m[38;5;124mn_admin_msgs[39m[38;5;124m"[39m)
[1;32m     10[0m )
[1;32m     12[0m admin_counts[38;5;241m.[39mhead()

[0;31mNameError[0m: name 'admin' is not defined
```

```python
# -----------------------------------------------------------------------------
# COMBINE ALL FEATURES INTO booking_risk DataFrame
# -----------------------------------------------------------------------------

booking_risk = (
    msg_features
    .merge(booking_sentiment[["hotel_id", "ID_booking", "neg_share"]],
           on=["hotel_id", "ID_booking"], how="left")
    .merge(streak_df,
           on=["hotel_id", "ID_booking"], how="left")
    .merge(reply_time,
           on=["hotel_id", "ID_booking"], how="left")
    .merge(admin_counts,
           on=["hotel_id", "ID_booking"], how="left")
)

# Fill NaN for bookings with no admin response
booking_risk["n_admin_msgs"] = booking_risk["n_admin_msgs"].fillna(0).astype(int)
booking_risk["reply_time_min"] = booking_risk["reply_time_min"].fillna(-1)  # -1 = no reply

# Calculate admin_to_guest_ratio
booking_risk["admin_guest_ratio"] = (
    booking_risk["n_admin_msgs"] / booking_risk["n_guest_msgs"]
).round(2)

booking_risk.head(10)
```

```python
# -----------------------------------------------------------------------------
# RISK SCORE FORMULA (simple, explainable)
# -----------------------------------------------------------------------------
# Scale: 0-100, higher = more risky

booking_risk["risk_score"] = (
    # 40% weight: overall negativity
    booking_risk["neg_share"] * 0.4 +

    # Streak penalty: +10 per consecutive NEG
    booking_risk["neg_streak"] * 10 +

    # Ended badly: +15
    (booking_risk["last_sentiment"] == "NEG").astype(int) * 15 +

    # Slow/no reply penalty: 0-15 points
    # -1 means no reply = max penalty (15)
    # >60 min = max penalty (15)
    booking_risk["reply_time_min"].apply(
        lambda x: 15 if x == -1 else min(x / 60 * 15, 15)
    )
).clip(0, 100).round(1)

# Risk levels
booking_risk["risk_level"] = pd.cut(
    booking_risk["risk_score"],
    bins=[-1, 30, 60, 100],
    labels=["LOW", "MEDIUM", "HIGH"]
)

booking_risk[["hotel_id", "ID_booking", "risk_score", "risk_level",
              "neg_share", "neg_streak", "last_sentiment", "reply_time_min"]].head(20)
```

```python
# -----------------------------------------------------------------------------
# CHECK: Risk distribution
# -----------------------------------------------------------------------------

print("=== Risk Level Distribution ===")
print(booking_risk["risk_level"].value_counts())
print()
print("=== Risk Score Stats ===")
print(booking_risk["risk_score"].describe())
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


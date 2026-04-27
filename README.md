# NLP Prediction Markets

## Market Data - Kalshi

### Format:

```csv
timestamp,yes_price
2025-01-01 00:00:00,0.42
2025-01-01 12:00:00,0.45
2025-01-02 00:00:00,0.47
```

* `timestamp` -> when the price was recorded
* `yes_price` -> probability (0 to 1)

---

## Social Data (Tweets / News / Posts)

### Format:

```csv
timestamp,text,views
2025-01-01 03:12:00,"Fed might cut rates",1200
2025-01-01 05:47:00,"Markets overreacting",800
2025-01-01 10:21:00,"Recession fears rising",5000
```

* `text` -> what people are saying
* `views` -> how many people saw it (Can't be 0 or else useless)

---

## Training Data (for Naive Bayes)

Used only to teach the basic model what sentiment looks like.

### Format:

```csv
text,label
"Stocks going up",1
"Market crashing",-1
"Uncertainty remains",0
```

### Labels:

* `1` = positive
* `0` = neutral
* `-1` = negative

---

# What code does

1. Train Naive Bayes (once)
2. Run sentiment on all text:

   * FinBERT (smart model)
   * Naive Bayes (simple baseline)
3. Weight posts by views:

   * big posts matter more
4. Group everything into **12-hour chunks**
5. Match sentiment with market prices
6. Run analysis

---

# What the results mean

## Directional Accuracy

### Question:

> Did we correctly predict if price goes up or down?

### Example:

* Predicted: up
* Actual: up → correct

### Interpretation:

* ~33% = random (3 options: up/down/stable)
* > 50% = actually useful

> “Is sentiment helpful at all?”

---

## MAE (Mean Absolute Error)

### Question:

> How close are our predictions to the actual price?

```text
difference = |prediction - actual|
```

Then average it

### Interpretation:

* Lower = better
* Measures **how wrong you are** not just direction

> “How accurate are we numerically?”

---

## Lag Analysis

### Question:

> Does sentiment come before price moves, or after?

We shift sentiment forward and backward in time and see where it lines up best with price changes.

### How to read results:

#### If lag is positive:

```text
Lag = +2
```

Sentiment happens **before** price moves
**Predictive**

---

#### If lag is 0:

Happens at the same time
market reacts instantly

---

#### If lag is negative:

```text
Lag = -2
```

Price moves first
sentiment is just reacting

---

> “Is social media actually useful for predicting markets?”

---
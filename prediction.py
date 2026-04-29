import pandas as pd
import numpy as np
import os
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

class PredictionMarketSystem:
    def __init__(self, initial_capital=1000.0):
        self.initial_capital = initial_capital
        print("Initializing NLP Models...")

        # Use a general news/social sentiment model for better overlap on sports headlines.
        self.sentiment_model_name = "ProsusAI/finbert"
        try:
            self.finbert = pipeline("sentiment-analysis", model=self.sentiment_model_name)
        except Exception as e:
            print(
                "Warning: Could not load sentiment model "
                f"({self.sentiment_model_name}). Error: {e}"
            )
            self.finbert = None

        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.nb_model = MultinomialNB()
        self.nb_is_trained = False

    # Train the Naive Bayes model on our custom dataset
    def train_naive_bayes(self, training_df):
        print("Training Naive Bayes...")
        X = self.vectorizer.fit_transform(training_df["text"])
        self.nb_model.fit(X, training_df["label"])
        self.nb_is_trained = True
        print("Naive Bayes training complete.")

    # Process FinBERT in batches so it doesn't take forever
    def _get_finbert_scores_batch(self, texts, batch_size=32):
        if not self.finbert:
            return [0.0] * len(texts)

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            outputs = self.finbert(batch)
            for res in outputs:
                label = str(res["label"]).lower()
                confidence = float(res["score"])

                # Support common HF sentiment label schemes:
                # - positive/neutral/negative
                # - LABEL_2/LABEL_1/LABEL_0 (twitter-roberta style)
                if label in {"positive", "label_2"}:
                    polarity = 1.0
                elif label in {"negative", "label_0"}:
                    polarity = -1.0
                else:
                    polarity = 0.0

                score = polarity * confidence
                results.append(score)
        return results

    # Safely get a Naive Bayes score (returns 0 if it hasn't been trained yet)
    def _get_nb_score(self, text):
        if not self.nb_is_trained:
            return 0.0

        vec = self.vectorizer.transform([text])
        probs = self.nb_model.predict_proba(vec)[0]
        class_map = dict(zip(self.nb_model.classes_, probs))

        # 0 = neutral, 1 = negative, 2 = positive
        return class_map.get(2, 0.0) - class_map.get(1, 0.0)

    # Convert raw text into 12-hour sentiment averages
    def process_sentiment(self, text_df):
        print("Processing sentiment without view weighting...")
        
        text_df["finbert_raw"] = self._get_finbert_scores_batch(text_df["text"].tolist())
        text_df["nb_raw"] = text_df["text"].apply(self._get_nb_score)
        text_df.set_index("timestamp", inplace=True)

        return text_df.groupby(pd.Grouper(freq="12h")).agg(
            finbert_score=("finbert_raw", "mean"), nb_score=("nb_raw", "mean")
        )

    # Run the trading simulation and calculate our performance metrics
    def run_simulation(self, kalshi_df, sentiment_df, model_name="finbert", z_threshold=1.0, rolling_window=14, predict_lag=0):
        # Line up the sentiment timestamps with the market timestamps
        sentiment_df = sentiment_df.reindex(kalshi_df.index).ffill().bfill()
        merged_df = pd.merge(kalshi_df, sentiment_df, left_index=True, right_index=True, how="inner")
        target_col = f"{model_name}_score"

        # Shift the signals forward or backward in time if requested
        if predict_lag != 0:
            merged_df[target_col] = merged_df[target_col].shift(predict_lag)

        merged_df = merged_df.dropna(subset=[target_col, "yes_price"])
        if merged_df.empty:
            raise ValueError("No overlapping rows between market and sentiment data. Check your timestamps.")

        # Calculate a rolling Z-score so we only trade on unusually high/low sentiment spikes
        rolling_mean = merged_df[target_col].rolling(window=rolling_window, min_periods=1).mean()
        rolling_std = merged_df[target_col].rolling(window=rolling_window, min_periods=1).std().replace(0, 1e-5)
        merged_df["z_score"] = (merged_df[target_col] - rolling_mean) / rolling_std

        # 1 = Buy YES, -1 = Buy NO, 0 = Hold
        merged_df["signal"] = 0
        merged_df.loc[merged_df["z_score"] > z_threshold, "signal"] = 1
        merged_df.loc[merged_df["z_score"] < -z_threshold, "signal"] = -1

        capital = self.initial_capital
        contracts = 0.0
        position = "cash"
        portfolio_history = []

        print(f"\n--- Execution Log: {model_name.upper()} (Lag={predict_lag}, Z-Thresh={z_threshold}) ---")

        # Walk through time step-by-step to simulate trading
        for timestamp, row in merged_df.iterrows():
            price = row["yes_price"]
            signal = row["signal"]

            # Sentiment spiked up -> Buy YES
            if signal == 1 and position != "yes":
                if position == "no":
                    capital = contracts * (1 - price)
                    contracts = 0.0
                if capital > 0 and price > 0:
                    contracts = capital / price
                    capital = 0.0
                    position = "yes"
                    print(f"[{timestamp}] BUY YES @ {price:.2f} (Z: {row['z_score']:.2f})")
            
            # Sentiment crashed -> Buy NO
            elif signal == -1 and position != "no":
                if position == "yes":
                    capital = contracts * price
                    contracts = 0.0
                if capital > 0 and price < 1:
                    contracts = capital / (1 - price)
                    capital = 0.0
                    position = "no"
                    print(f"[{timestamp}] BUY NO @ {price:.2f} (Z: {row['z_score']:.2f})")

            # Calculate current portfolio value and save it
            portfolio_value = capital + (contracts * price if position == "yes" else 0.0) + (contracts * (1 - price) if position == "no" else 0.0)
            portfolio_history.append(portfolio_value)

        if not portfolio_history:
            raise ValueError(
                f"No rows left to simulate for model={model_name}, lag={predict_lag}. "
                "Likely caused by lag shift + sparse overlap."
            )

        # Wrap up the simulation and calculate ROI
        merged_df["portfolio_value"] = portfolio_history
        final_value = merged_df["portfolio_value"].iloc[-1]
        roi = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # Figure out if the model actually guessed the right direction
        merged_df["future_price"] = merged_df["yes_price"].shift(-1)
        merged_df["future_change"] = merged_df["future_price"] - merged_df["yes_price"]
        merged_df["actual_direction"] = np.where(merged_df["future_change"] > 0, 1, np.where(merged_df["future_change"] < 0, -1, 0))

        valid = merged_df["future_price"].notna() & (merged_df["signal"] != 0)
        directional_accuracy = (merged_df.loc[valid, "signal"] == merged_df.loc[valid, "actual_direction"]).mean() if valid.sum() > 0 else 0.0

        # Calculate Mean Absolute Error (MAE) based on standard deviation
        price_volatility = merged_df["yes_price"].std()
        merged_df["predicted_price"] = merged_df["yes_price"] + (merged_df["z_score"] * price_volatility * 0.1)
        mae_df = merged_df.dropna(subset=["future_price", "predicted_price"])
        mae = np.mean(np.abs(mae_df["predicted_price"] - mae_df["future_price"]))

        return final_value, roi, directional_accuracy, mae

    # Test different time shifts to see if sentiment predicts price or just reacts to it
    def lag_time_analysis(self, kalshi_df, sentiment_df, model_name="finbert", max_lag=24):
        sentiment_df = sentiment_df.reindex(kalshi_df.index).ffill().bfill()
        merged_df = pd.merge(kalshi_df, sentiment_df, left_index=True, right_index=True, how="inner")
        target_col = f"{model_name}_score"

        merged_df = merged_df.dropna(subset=[target_col, "yes_price"])
        if merged_df.empty:
            raise ValueError("No overlapping rows for lag analysis.")

        merged_df["price_change"] = merged_df["yes_price"].diff()
        lags = range(-max_lag, max_lag + 1)
        correlations = []

        # Shift the sentiment forward and backward to find the strongest correlation
        for lag in lags:
            shifted_sentiment = merged_df[target_col].shift(lag)
            valid = shifted_sentiment.notna() & merged_df["price_change"].notna()
            
            if valid.sum() < 10:
                correlations.append(np.nan)
                continue
                
            corr = shifted_sentiment[valid].corr(merged_df.loc[valid, "price_change"])
            correlations.append(corr)

        lag_df = pd.DataFrame({"lag": list(lags), "correlation": correlations})

        # Guard against flat/degenerate signals where all correlations are NaN.
        valid_corr = lag_df.dropna(subset=["correlation"])
        if valid_corr.empty:
            print("\n--- LAG ANALYSIS ---")
            print("Best Lag: 0 periods")
            print("Correlation at Best Lag: nan")
            print(
                "Interpretation: No valid lag correlation (signal likely constant or insufficient variance)."
            )
            return lag_df, 0, np.nan

        best_row = valid_corr.loc[valid_corr["correlation"].abs().idxmax()]
        best_lag = int(best_row["lag"])
        best_corr = best_row["correlation"]

        print("\n--- LAG ANALYSIS ---")
        print(f"Best Lag: {best_lag} periods")
        print(f"Correlation at Best Lag: {best_corr:.4f}")

        # Explain what the best lag actually means for trading
        if best_lag > 0:
            print("Interpretation: Sentiment LEADS price. (Trade near lag 0 to capture future move.)")
        elif best_lag < 0:
            print("Interpretation: Price LEADS sentiment. (Sentiment is a lagging validator, not an entry trigger.)")
        else:
            print("Interpretation: Instantaneous relationship.")

        return lag_df, best_lag, best_corr

# Helper function to load and clean up CSV/JSON files
def load_data(file_path, time_col="timestamp", required_cols=None, column_aliases=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Only CSV or JSON supported")

    # Rename columns to standard names if needed
    if column_aliases:
        for canonical_col, aliases in column_aliases.items():
            if canonical_col in df.columns:
                continue
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: canonical_col})
                    break

    # Standardize the time column
    if time_col is not None:
        if time_col not in df.columns:
            fallback_time_cols = ["timestamp", "date", "datetime", "pubDate", "time"]
            for fallback in fallback_time_cols:
                if fallback in df.columns:
                    df = df.rename(columns={fallback: time_col})
                    break
                    
        if time_col not in df.columns:
            raise KeyError(f"Missing required time column '{time_col}'. Available: {list(df.columns)}")

        df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)

    # Double check that we have all the data we need
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns {missing_cols} in {file_path}. Available: {list(df.columns)}")

    return df

# Helper to visualize the lag analysis
def plot_lag_analysis(lag_df):
    plt.figure()
    plt.plot(lag_df["lag"], lag_df["correlation"])
    plt.axvline(0)
    plt.title("Lag vs Correlation (Sentiment vs Price Change)")
    plt.xlabel("Lag (Periods)")
    plt.ylabel("Correlation")
    plt.show()

if __name__ == "__main__":
    KALSHI_DATA_FILE = "data/nba_market_data_okc.csv"
    SOCIAL_DATA_FILE = "data/okc_headlines_data.csv"
    NB_TRAIN_FILE = "data/nbtrain.csv"

    system = PredictionMarketSystem(initial_capital=1000.0)

    try:
        train_df = load_data(NB_TRAIN_FILE, time_col=None, required_cols=["text", "label"])
        system.train_naive_bayes(train_df)
    except Exception as e:
        print(f"Skipping NB training: {e}")

    kalshi_prices = load_data(
        KALSHI_DATA_FILE,
        required_cols=["timestamp", "yes_price"],
        column_aliases={"yes_price": ["price", "yesPrice", "yes", "close"]},
    )
    kalshi_prices.set_index("timestamp", inplace=True)
    kalshi_prices = kalshi_prices.groupby(pd.Grouper(freq="12h")).last()
    kalshi_prices = kalshi_prices.dropna(subset=["yes_price"])

    social_data = load_data(
        SOCIAL_DATA_FILE,
        required_cols=["timestamp", "text"],
        column_aliases={
            "timestamp": ["date", "datetime", "pubDate", "time"],
            "text": ["title", "headline", "content"],
        },
    )

    sentiment_df = system.process_sentiment(social_data)
    lag_df, best_lag, best_corr = system.lag_time_analysis(kalshi_prices, sentiment_df, model_name="finbert", max_lag=24)

    finbert_res = system.run_simulation(kalshi_prices, sentiment_df, "finbert", predict_lag=0)
    nb_res = system.run_simulation(kalshi_prices, sentiment_df, "nb", predict_lag=0)

    predictive_lag = 1
    finbert_pred_res = system.run_simulation(kalshi_prices, sentiment_df, "finbert", predict_lag=predictive_lag)
    nb_pred_res = system.run_simulation(kalshi_prices, sentiment_df, "nb", predict_lag=predictive_lag)

    finbert_aligned_res = system.run_simulation(kalshi_prices, sentiment_df, "finbert", predict_lag=best_lag)

    print("\nFINAL RESULTS")
    print(f"FinBERT -> Final: ${finbert_res[0]:.2f}, ROI: {finbert_res[1]:.2f}%")
    print(f"          Directional Acc: {finbert_res[2]:.3f}, MAE: {finbert_res[3]:.4f}")
    print(f"Naive Bayes -> Final: ${nb_res[0]:.2f}, ROI: {nb_res[1]:.2f}%")
    print(f"               Directional Acc: {nb_res[2]:.3f}, MAE: {nb_res[3]:.4f}")

    print("\nPREDICTIVE TEST (lag=1)")
    print(f"FinBERT -> Final: ${finbert_pred_res[0]:.2f}, ROI: {finbert_pred_res[1]:.2f}%")
    print(f"          Directional Acc: {finbert_pred_res[2]:.3f}, MAE: {finbert_pred_res[3]:.4f}")
    print(f"Naive Bayes -> Final: ${nb_pred_res[0]:.2f}, ROI: {nb_pred_res[1]:.2f}%")
    print(f"               Directional Acc: {nb_pred_res[2]:.3f}, MAE: {nb_pred_res[3]:.4f}")

    print(f"\nREACTIVE ALIGNMENT (lag={best_lag})")
    print(f"FinBERT -> Final: ${finbert_aligned_res[0]:.2f}, ROI: {finbert_aligned_res[1]:.2f}%")
    print(f"          Directional Acc: {finbert_aligned_res[2]:.3f}, MAE: {finbert_aligned_res[3]:.4f}")

    plot_lag_analysis(lag_df)
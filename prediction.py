import pandas as pd
import numpy as np
import os
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class PredictionMarketSystem:
    def __init__(self, initial_capital=1000.0):
        self.initial_capital = initial_capital
        
        print("Initializing NLP Models...")
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.nb_model = MultinomialNB()
        self.nb_is_trained = False


    # NAIVE BAYES TRAINING
    def train_naive_bayes(self, training_df):
        print("Training Naive Bayes...")
        X = self.vectorizer.fit_transform(training_df['text'])
        self.nb_model.fit(X, training_df['label'])
        self.nb_is_trained = True
        print("Naive Bayes training complete.")

    # FAST FINBERT
    def _get_finbert_scores_batch(self, texts, batch_size=32):
        mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            outputs = self.finbert(batch)

            for res in outputs:
                score = mapping.get(res['label'], 0.0) * res['score']
                results.append(score)

        return results

    # SAFE NB SCORE
    def _get_nb_score(self, text):
        if not self.nb_is_trained:
            return 0.0

        vec = self.vectorizer.transform([text])
        probs = self.nb_model.predict_proba(vec)[0]
        class_map = dict(zip(self.nb_model.classes_, probs))

        return class_map.get(1, 0.0) - class_map.get(-1, 0.0)

    # SENTIMENT PROCESSING (WITH VIEW WEIGHT)
    def process_sentiment(self, text_df):
        print("Processing sentiment with view weighting...")

        # FinBERT
        text_df['finbert_raw'] = self._get_finbert_scores_batch(
            text_df['text'].tolist()
        )

        # Naive Bayes
        text_df['nb_raw'] = text_df['text'].apply(self._get_nb_score)

        # View weighting (log scaled)
        text_df['log_views'] = np.log10(text_df['views'] + 1)

        text_df.set_index('timestamp', inplace=True)

        # Weighted aggregation per 12H window
        def weighted_avg(group):
            weights = group['log_views']

            if weights.sum() == 0:
                return pd.Series({'finbert_score': 0.0, 'nb_score': 0.0})

            weights = weights / weights.sum()

            return pd.Series({
                'finbert_score': (group['finbert_raw'] * weights).sum(),
                'nb_score': (group['nb_raw'] * weights).sum()
            })

        return text_df.groupby(pd.Grouper(freq='12H')).apply(weighted_avg)

    # MAIN SIMULATION + METRICS
    def run_simulation(self, kalshi_df, sentiment_df, model_name='finbert', threshold=0.20):
        merged_df = pd.merge(
            kalshi_df, sentiment_df,
            left_index=True, right_index=True, how='left'
        )

        target_col = f'{model_name}_score'

        # Prevent stale sentiment
        merged_df[target_col] = merged_df[target_col].ffill(limit=2)
        merged_df = merged_df.dropna(subset=[target_col, 'yes_price'])

        # 3-CLASS SIGNAL (WITH HOLD ZONE)
        neutral_band = threshold / 2

        merged_df['signal'] = np.select(
            [
                merged_df[target_col] > threshold,
                merged_df[target_col] < -threshold,
                merged_df[target_col].abs() <= neutral_band
            ],
            [1, -1, 0],
            default=0
        )

        # HOLDING LOGIC (NO OVERTRADING)
        capital = self.initial_capital
        position_contracts = 0.0
        current_position = 0

        portfolio_history = []

        print(f"\n--- Execution Log: {model_name.upper()} ---")

        for timestamp, row in merged_df.iterrows():
            price = row['yes_price']
            signal = row['signal']

            # BUY
            if signal == 1 and current_position != 1:
                if position_contracts > 0:
                    capital = position_contracts * price
                    position_contracts = 0.0

                if capital > 0:
                    position_contracts = capital / price
                    capital = 0.0
                    current_position = 1
                    print(f"[{timestamp}] BUY @ {price:.2f}")

            # SELL / EXIT
            elif signal == -1 and current_position != -1:
                if position_contracts > 0:
                    capital = position_contracts * price
                    position_contracts = 0.0

                current_position = -1
                print(f"[{timestamp}] EXIT @ {price:.2f}")

            # HOLD -> do nothing

            portfolio_history.append(capital + position_contracts * price)

        merged_df['portfolio_value'] = portfolio_history

        final_value = merged_df['portfolio_value'].iloc[-1]
        roi = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # METRICS

        # Price movement
        merged_df['price_change'] = merged_df['yes_price'].diff()

        merged_df['actual_direction'] = np.where(
            merged_df['price_change'] > 0, 1,
            np.where(merged_df['price_change'] < 0, -1, 0)
        )

        # Directional Accuracy
        directional_accuracy = (
            (merged_df['signal'] == merged_df['actual_direction'])
            .mean()
        )

        # MAE
        mae = np.mean(np.abs(merged_df[target_col] - merged_df['yes_price']))

        return final_value, roi, directional_accuracy, mae


# FILE LOADING
def load_data(file_path, time_col='timestamp'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Only CSV or JSON supported")

    df[time_col] = pd.to_datetime(df[time_col])
    return df


# MAIN EXECUTION
if __name__ == "__main__":

    KALSHI_DATA_FILE = "data/kalshi_prices.csv"
    SOCIAL_DATA_FILE = "data/social_feed.csv"
    KAGGLE_TRAIN_FILE = "data/kaggle_train.csv"

    system = PredictionMarketSystem(initial_capital=1000.0)

    # Train NB
    try:
        train_df = load_data(KAGGLE_TRAIN_FILE)
        system.train_naive_bayes(train_df)
    except Exception as e:
        print(f"Skipping NB training: {e}")

    # Load data
    kalshi_prices = load_data(KALSHI_DATA_FILE)
    kalshi_prices.set_index('timestamp', inplace=True)

    social_data = load_data(SOCIAL_DATA_FILE)

    # Run pipeline
    sentiment_df = system.process_sentiment(social_data)

    finbert_res = system.run_simulation(kalshi_prices, sentiment_df, 'finbert')
    nb_res = system.run_simulation(kalshi_prices, sentiment_df, 'nb')

    # Results
    print("\nFINAL RESULTS")

    print(f"FinBERT -> Final: ${finbert_res[0]:.2f}, ROI: {finbert_res[1]:.2f}%")
    print(f"          Directional Acc: {finbert_res[2]:.3f}, MAE: {finbert_res[3]:.4f}")

    print(f"Naive Bayes -> Final: ${nb_res[0]:.2f}, ROI: {nb_res[1]:.2f}%")
    print(f"               Directional Acc: {nb_res[2]:.3f}, MAE: {nb_res[3]:.4f}")
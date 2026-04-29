import os
import importlib
import io
from contextlib import redirect_stdout
import pandas as pd
import matplotlib.pyplot as plt
sns = importlib.import_module("seaborn") if importlib.util.find_spec("seaborn") else None

from prediction import PredictionMarketSystem, load_data

NB_TRAIN_FILE = "data/nbtrain.csv"
OUTPUT_DIR = "results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MARKETS = [
    {
        "name": "Apple CEO",
        "market_data": "data/kalshi/appleceo_market.csv",
        "social_data": "data/news/appleCEO_headlines_data.csv",
    },
    {
        "name": "Celtics",
        "market_data": "data/kalshi/nba_market_data_celtics.csv",
        "social_data": "data/news/celtics_headlines_data.csv",
    },
    {
        "name": "Spurs",
        "market_data": "data/kalshi/nba_market_data_spurs.csv",
        "social_data": "data/news/spurs_headlines_data.csv",
    },
    {
        "name": "OKC",
        "market_data": "data/kalshi/nba_market_data_okc.csv",
        "social_data": "data/news/okc_headlines_data.csv",
    },
    {
        "name": "England",
        "market_data": "data/kalshi/worldcup_market_data_england.csv",
        "social_data": "data/news/england_headlines_data.csv",
    },
    {
        "name": "France",
        "market_data": "data/kalshi/worldcup_market_data_france.csv",
        "social_data": "data/news/france_headlines_data.csv",
    },
    {
        "name": "Spain",
        "market_data": "data/kalshi/worldcup_market_data_spain.csv",
        "social_data": "data/news/spain_headlines_data.csv",
    },
    {
        "name": "Gas Price",
        "market_data": "data/kalshi/gas_kalshi_12h.csv",
        "social_data": "data/news/gas_price_headlines_data.csv",
    },
    {
        "name": "Jobless Claims",
        "market_data": "data/kalshi/jobless_market.csv",
        "social_data": "data/news/unemployment_headlines_data.csv",
    },
]

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def _run_quiet(func, *args, **kwargs):
    """Run noisy functions without flooding terminal output."""
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)

def _build_markdown_report(results_df: pd.DataFrame) -> str:
    total = len(results_df)
    failed = int(results_df["Error"].notna().sum()) if "Error" in results_df.columns else 0
    succeeded = total - failed

    lines = [
        "# Batch Run Report",
        "",
        "## Summary",
        f"- Total markets: {total}",
        f"- Succeeded: {succeeded}",
        f"- Failed: {failed}",
        "",
    ]

    if failed > 0:
        fail_df = results_df[results_df["Error"].notna()][["Market Name", "Error"]]
        lines.extend(["## Failures", ""])
        try:
            lines.append(fail_df.to_markdown(index=False))
        except ImportError:
            lines.append(fail_df.to_string(index=False))
        lines.append("")

    lines.extend(["## Detailed Results", ""])
    try:
        lines.append(results_df.to_markdown(index=False))
    except ImportError:
        lines.append(results_df.to_string(index=False))
    lines.append("")
    return "\n".join(lines)

def save_lag_plot(lag_df, market_name):
    """Saves the lag plot instead of using plt.show() which blocks execution."""
    plt.figure(figsize=(10, 6))
    plt.plot(lag_df["lag"], lag_df["correlation"], marker='o', linestyle='-')
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f"Lag vs Correlation - {market_name}")
    plt.xlabel("Lag (Periods)")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"{market_name.replace(' ', '_')}_lag_analysis.png"))
    plt.close() # Close figure to free memory

def plot_aggregate_results(results_df):
    """Generates a master bar chart comparing ROI across all markets."""
    plt.figure(figsize=(14, 7))
    
    # Reshape data for seaborn plotting
    plot_data = results_df.melt(
        id_vars=['Market Name'], 
        value_vars=['FinBERT ROI (Lag 0)', 'FinBERT ROI (Lag 1)', 'FinBERT ROI (Best Lag)'],
        var_name='Strategy', 
        value_name='ROI (%)'
    )
    
    if sns is not None:
        sns.barplot(data=plot_data, x='Market Name', y='ROI (%)', hue='Strategy')
    else:
        pivot = plot_data.pivot(
            index="Market Name", columns="Strategy", values="ROI (%)"
        ).fillna(0)
        ax = pivot.plot(kind="bar", ax=plt.gca())
        ax.legend(title="Strategy")
    plt.title("FinBERT ROI Comparison Across All Markets")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Master_ROI_Comparison.png"))
    plt.close()

def run_batch():
    setup_directories()
    
    # Initialize the system
    print("Initializing batch system...")
    system = _run_quiet(PredictionMarketSystem, initial_capital=1000.0)
    
    # Train Naive Bayes
    try:
        train_df = load_data(NB_TRAIN_FILE, time_col=None, required_cols=["text", "label"])
        _run_quiet(system.train_naive_bayes, train_df)
    except Exception as e:
        print(f"Skipping NB training: {e}")

    all_results = []

    # Process each market
    for i, market in enumerate(MARKETS):
        market_name = market["name"]
        print(f"[{i+1}/{len(MARKETS)}] Processing: {market_name}")
        
        try:
            # Load Data
            kalshi_prices = load_data(
                market["market_data"],
                required_cols=["timestamp", "yes_price"],
                column_aliases={"yes_price": ["price", "yesPrice", "yes", "close"]},
            )
            kalshi_prices.set_index("timestamp", inplace=True)
            kalshi_prices = kalshi_prices.groupby(pd.Grouper(freq="12h")).last().dropna(subset=["yes_price"])

            social_data = load_data(
                market["social_data"],
                required_cols=["timestamp", "text"],
                column_aliases={"timestamp": ["date", "datetime", "pubDate", "time"], "text": ["title", "headline", "content"]},
            )

            # Process Sentiment & Lag
            sentiment_df = _run_quiet(system.process_sentiment, social_data)
            lag_df, best_lag, best_corr = _run_quiet(
                system.lag_time_analysis,
                kalshi_prices,
                sentiment_df,
                model_name="finbert",
                max_lag=24
            )
            save_lag_plot(lag_df, market_name)

            # Run Simulations
            # Baseline (Lag 0)
            fb_0 = _run_quiet(
                system.run_simulation, kalshi_prices, sentiment_df, "finbert", predict_lag=0
            )
            nb_0 = _run_quiet(
                system.run_simulation, kalshi_prices, sentiment_df, "nb", predict_lag=0
            )
            
            # Predictive (Lag 1)
            fb_1 = _run_quiet(
                system.run_simulation, kalshi_prices, sentiment_df, "finbert", predict_lag=1
            )
            nb_1 = _run_quiet(
                system.run_simulation, kalshi_prices, sentiment_df, "nb", predict_lag=1
            )
            
            # Reactive Alignment (Best Lag)
            fb_best = _run_quiet(
                system.run_simulation, kalshi_prices, sentiment_df, "finbert", predict_lag=best_lag
            )

            # Store Results
            all_results.append({
                "Market Name": market_name,
                "Best Lag": best_lag,
                "Best Lag Corr": round(best_corr, 4),
                "FinBERT ROI (Lag 0)": round(fb_0[1], 2),
                "FinBERT Acc (Lag 0)": round(fb_0[2], 3),
                "NB ROI (Lag 0)": round(nb_0[1], 2),
                "NB Acc (Lag 0)": round(nb_0[2], 3),
                "FinBERT ROI (Lag 1)": round(fb_1[1], 2),
                "FinBERT Acc (Lag 1)": round(fb_1[2], 3),
                "NB ROI (Lag 1)": round(nb_1[1], 2),
                "NB Acc (Lag 1)": round(nb_1[2], 3),
                "FinBERT ROI (Best Lag)": round(fb_best[1], 2),
                "FinBERT Acc (Best Lag)": round(fb_best[2], 3),
                "Error": None,
            })
            print(f"  -> done")

        except Exception as e:
            print(f"ERROR processing {market_name}: {e}")
            all_results.append({"Market Name": market_name, "Error": str(e)})


    print("Compiling reports...")
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "Master_Market_Results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Master results saved to: {csv_path}")
    
   
    valid_results_df = results_df.dropna(subset=['FinBERT ROI (Lag 0)'])
    if not valid_results_df.empty:
        plot_aggregate_results(valid_results_df)
        print("Plots saved.")

    report_path = os.path.join(OUTPUT_DIR, "Batch_Run_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown_report(results_df))
    print(f"Markdown report saved to: {report_path}")

if __name__ == "__main__":
    run_batch()
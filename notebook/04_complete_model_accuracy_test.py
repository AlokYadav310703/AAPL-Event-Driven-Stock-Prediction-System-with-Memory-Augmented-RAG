"""
AAPL Stock Price Prediction Model - Accuracy Testing on 2025 Data

This script tests the complete 2-stage ensemble model on 6 months of 2025 AAPL data:
  - Stage 1: Ensemble (LSTM + CNN) predictions
  - Stage 2: News-aware correction using Random Forest model

Outputs:
  - Comprehensive accuracy metrics (MSE, RMSE, MAE, R², MAPE, Directional Accuracy)
  - Prediction vs Actual plots
  - Error analysis and statistics
  - Results CSV file
  - Performance summary
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import logging

# SETUP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TEXT_CHAR_CAP = 2000

STAGE2_FEATURE_ORDER = [
    "sentiment_score",
    "impact_score",
    "event_weight",
    "return_1d",
    "return_5d",
]

# MODEL LOADING FUNCTIONS
def load_models(model_path: str, correction_model_filename: str = "Best_Correction_Model.pkl") -> dict:
    """Load all trained models and scalers"""
    logger.info(f"Loading models from {model_path}...")
    
    try:
        models = {}
        
        # Load base learners
        models['lstm'] = tf.keras.models.load_model(f"{model_path}/lstm_model.keras")
        models['cnn'] = tf.keras.models.load_model(f"{model_path}/cnn_model.keras")
        
        # Load meta-learner
        try:
            models['meta_learner'] = joblib.load(f"{model_path}/meta_model.pkl")
        except FileNotFoundError:
            models['meta_learner'] = joblib.load(f"{model_path}/meta_learner.pkl")
        
        # Load correction model
        models['correction_model'] = joblib.load(f"{model_path}/{correction_model_filename}")
        
        # Load scalers
        models['feature_scaler'] = joblib.load(f"{model_path}/feature_scaler.pkl")
        models['target_scaler'] = joblib.load(f"{model_path}/target_scaler.pkl")
        models['feature_scaler_raw'] = joblib.load(f"{model_path}/feature_scaler_raw.pkl")
        models['target_scaler_raw'] = joblib.load(f"{model_path}/target_scaler_raw.pkl")
        
        logger.info("✓ All models loaded successfully")
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def load_chromadb_collection(chroma_path: str):
    """Load ChromaDB collection with stored AAPL news events"""
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(
            name="aapl_events",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✓ ChromaDB loaded ({collection.count()} events)")
        return collection
    except Exception as e:
        logger.warning(f"ChromaDB not available: {e}. Using default news features.")
        return None


# DATA FETCHING FUNCTIONS
def fetch_aapl_data_2025(months: int = 6) -> pd.DataFrame:
    """
    Fetch AAPL data for 2025 (first N months)
    
    Returns:
        DataFrame with OHLC data
    """
    logger.info(f"Fetching AAPL data for 2025 ({months} months)...")
    
    start_date = "2025-01-01"
    end_date = f"2025-{months:02d}-30"  # Approximate end date
    
    df = yf.download("AAPL", start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = df.columns.str.lower()
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    logger.info(f"✓ Fetched {len(df)} trading days")
    return df


def fetch_historical_data(lookback_months: int = 12) -> pd.DataFrame:
    """
    Fetch historical data to use for training context (for denoising/scaling)
    """
    logger.info(f"Fetching historical data ({lookback_months} months for context)...")
    
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=lookback_months*30)
    
    df = yf.download("AAPL", start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = df.columns.str.lower()
    
    logger.info(f"✓ Fetched {len(df)} historical days")
    return df


def compute_market_returns(df: pd.DataFrame, date_idx: int) -> dict:
    """Compute 1-day and 5-day returns for a specific date"""
    try:
        closes = df["close"].values.astype(float)
        
        if date_idx < 5:
            # Not enough history, use what we have
            return_1d = (closes[date_idx] - closes[date_idx-1]) / closes[date_idx-1] if date_idx > 0 else 0.0
            return_5d = (closes[date_idx] - closes[0]) / closes[0] if date_idx > 0 else 0.0
        else:
            return_1d = (closes[date_idx] - closes[date_idx-1]) / closes[date_idx-1]
            return_5d = (closes[date_idx] - closes[date_idx-6]) / closes[date_idx-6]
        
        return {"return_1d": float(return_1d), "return_5d": float(return_5d)}
    except Exception as e:
        logger.warning(f"Error computing returns: {e}")
        return {"return_1d": 0.0, "return_5d": 0.0}


def fetch_news_for_date(api_key: str, date: pd.Timestamp) -> list[dict]:
    """
    Fetch Apple news for a specific date from NewsAPI
    """
    try:
        from_date = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        to_date = date.strftime("%Y-%m-%d")
        
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "Apple AAPL",
                "language": "en",
                "sortBy": "publishedAt",
                "from": from_date,
                "to": to_date,
                "pageSize": 5,
                "apiKey": api_key,
            },
            timeout=15,
        )
        
        data = resp.json()
        if data.get("status") != "ok":
            return []
        
        articles = []
        for article in data.get("articles", []):
            title = article.get("title", "")
            body = article.get("description") or article.get("content", "")
            text = f"{title}. {body}".strip()[:TEXT_CHAR_CAP]
            
            articles.append({
                "title": title,
                "source": (article.get("source") or {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", "n/a"),
                "url": article.get("url", ""),
                "text": text,
            })
        
        return articles
    except Exception as e:
        logger.debug(f"NewsAPI request failed for {date}: {e}")
        return []


# FEATURE EXTRACTION & NEWS ANALYSIS
def find_similar_events(collection, encoder, query_text: str, top_k: int = 5):
    """Find similar stored events in ChromaDB"""
    if collection is None:
        return None, 0
    
    try:
        count = collection.count()
        if count == 0:
            return None, 0
        
        query_embedding = encoder.encode(query_text[:TEXT_CHAR_CAP]).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count)
        )
        return results, count
    except Exception as e:
        logger.debug(f"ChromaDB query failed: {e}")
        return None, 0


def extract_news_features(results) -> dict:
    """Extract aggregated news features from ChromaDB results"""
    if results is None or not results['metadatas'] or not results['metadatas'][0]:
        return _default_news_features()
    
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    similarities = 1 - np.array(distances)
    total_sim = similarities.sum()
    
    if total_sim == 0:
        return _default_news_features()
    
    weights = similarities / total_sim
    
    aggregated = {
        'sentiment_score': np.average(
            [float(m.get('sentiment_score', 0)) for m in metadatas],
            weights=weights
        ),
        'impact_score': np.average(
            [float(m.get('impact_score', 0)) for m in metadatas],
            weights=weights
        ),
        'event_weight': np.average(
            [float(m.get('event_weight', 0)) for m in metadatas],
            weights=weights
        ),
        'news_count': len(metadatas),
        'has_supply_chain_event': int(any(m.get('event_type') == 'supply_chain' for m in metadatas)),
    }
    
    return aggregated


def _default_news_features() -> dict:
    """Default zero features when no news is available"""
    return {
        'sentiment_score': 0.0,
        'impact_score': 0.0,
        'event_weight': 0.0,
        'news_count': 0,
        'has_supply_chain_event': 0,
    }


def process_news_for_date(
    articles: list[dict],
    collection,
    encoder,
    top_k: int = 5
) -> dict:
    """Process news articles and extract features"""
    if not articles:
        return _default_news_features()
    
    all_news_features = []
    
    for article in articles:
        results, _ = find_similar_events(collection, encoder, article['text'], top_k=top_k)
        news_feats = extract_news_features(results)
        all_news_features.append(news_feats)
    
    if all_news_features:
        feature_keys = all_news_features[0].keys()
        aggregated = {}
        for key in feature_keys:
            values = [f[key] for f in all_news_features]
            aggregated[key] = np.mean(values)
        return aggregated
    else:
        return _default_news_features()


# PREDICTION FUNCTIONS
def prepare_ohlc_features(df: pd.DataFrame, scaler, lookback: int, date_idx: int) -> np.ndarray:
    """Prepare OHLC data for model input"""
    try:
        ohlc = df[['open', 'high', 'low', 'close']].values.astype(float)
        
        # Get the window ending at date_idx
        if date_idx < lookback:
            start_idx = 0
            ohlc_window = ohlc[:date_idx+1]
        else:
            start_idx = date_idx - lookback + 1
            ohlc_window = ohlc[start_idx:date_idx+1]
        
        # Pad if necessary
        if len(ohlc_window) < lookback:
            ohlc_window = np.vstack([np.zeros((lookback - len(ohlc_window), 4)), ohlc_window])
        
        ohlc_scaled = scaler.transform(ohlc_window)
        
        return ohlc_scaled.reshape(1, lookback, 4)
    except Exception as e:
        logger.error(f"Error preparing OHLC features: {e}")
        return None


def get_stage1_prediction(
    models: dict,
    ohlc_for_lstm: np.ndarray,
    ohlc_for_cnn: np.ndarray,
) -> tuple:
    """Get Stage 1 ensemble prediction"""
    try:
        lstm_pred_scaled = models['lstm'].predict(ohlc_for_lstm, verbose=0)
        cnn_pred_scaled = models['cnn'].predict(ohlc_for_cnn, verbose=0)
        
        meta_input = np.hstack([lstm_pred_scaled, cnn_pred_scaled])
        stage1_pred_scaled = models['meta_learner'].predict(meta_input)
        stage1_pred_scaled = np.array(stage1_pred_scaled).reshape(-1, 1)
        
        lstm_pred_actual = models['target_scaler_raw'].inverse_transform(lstm_pred_scaled)[0, 0]
        cnn_pred_actual = models['target_scaler'].inverse_transform(cnn_pred_scaled)[0, 0]
        stage1_pred_actual = models['target_scaler'].inverse_transform(stage1_pred_scaled)[0, 0]
        
        return (
            float(stage1_pred_actual),
            {
                'lstm': float(lstm_pred_actual),
                'cnn': float(cnn_pred_actual),
            },
            float(stage1_pred_scaled[0, 0]),
        )
    except Exception as e:
        logger.error(f"Error in Stage 1 prediction: {e}")
        return None, None, None


def get_stage2_prediction(
    models: dict,
    stage1_pred_actual: float,
    news_features_raw: dict,
    market_returns: dict,
) -> tuple:
    """Get Stage 2 correction model prediction"""
    try:
        fused_features = {**news_features_raw, **market_returns}
        
        X = np.array(
            [[fused_features.get(feat, 0.0) for feat in STAGE2_FEATURE_ORDER]]
        )
        
        predicted_error = float(models['correction_model'].predict(X)[0])
        final_pred_actual = stage1_pred_actual - predicted_error
        correction_actual = final_pred_actual - stage1_pred_actual
        
        return float(final_pred_actual), float(correction_actual)
    except Exception as e:
        logger.error(f"Error in Stage 2 prediction: {e}")
        return None, None


# ACCURACY METRICS
def compute_accuracy_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute comprehensive accuracy metrics"""
    
    # Basic metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # R² score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Directional Accuracy (% of time direction was correct)
    actual_direction = np.diff(actual)
    predicted_direction = np.diff(predicted)
    direction_correct = np.sign(actual_direction) == np.sign(predicted_direction)
    directional_accuracy = np.mean(direction_correct) * 100
    
    # Mean absolute deviation from actual
    mad = np.mean(np.abs(actual - predicted))
    
    # Hit rate (within 1% of actual)
    within_1pct = np.abs((actual - predicted) / actual) <= 0.01
    hit_rate_1pct = np.mean(within_1pct) * 100
    
    # Hit rate (within 2% of actual)
    within_2pct = np.abs((actual - predicted) / actual) <= 0.02
    hit_rate_2pct = np.mean(within_2pct) * 100
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape),
        'MAD': float(mad),
        'Directional_Accuracy': float(directional_accuracy),
        'Hit_Rate_1pct': float(hit_rate_1pct),
        'Hit_Rate_2pct': float(hit_rate_2pct),
    }


# VISUALIZATION
def plot_results(
    actual_prices: np.ndarray,
    stage1_predictions: np.ndarray,
    stage2_predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    output_path: str = "./results"
):
    """Create comprehensive prediction vs actual plots"""
    
    Path(output_path).mkdir(exist_ok=True)
    
    # Plot 1: Full time series
    plt.figure(figsize=(16, 6))
    plt.plot(dates, actual_prices, label="Actual Price", linewidth=2, color='black', zorder=3)
    plt.plot(dates, stage1_predictions, label="Stage 1 Prediction", linewidth=1.5, alpha=0.7, color='orange')
    plt.plot(dates, stage2_predictions, label="Stage 2 Prediction (Final)", linewidth=2, alpha=0.8, color='red', linestyle='--')
    
    plt.title("AAPL Stock Price Prediction (2025) - Full Time Series", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/01_full_timeseries.png", dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}/01_full_timeseries.png")
    plt.close()
    
    # Plot 2: Error distribution
    stage1_error = actual_prices - stage1_predictions
    stage2_error = actual_prices - stage2_predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(stage1_error, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0].set_title("Stage 1 Prediction Error Distribution", fontweight='bold')
    axes[0].set_xlabel("Error ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(stage2_error, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title("Stage 2 Prediction Error Distribution", fontweight='bold')
    axes[1].set_xlabel("Error ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/02_error_distribution.png", dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}/02_error_distribution.png")
    plt.close()
    
    # Plot 3: Scatter plot (actual vs predicted)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stage 1
    axes[0].scatter(actual_prices, stage1_predictions, alpha=0.6, color='orange', s=30)
    min_price = min(actual_prices.min(), stage1_predictions.min())
    max_price = max(actual_prices.max(), stage1_predictions.max())
    axes[0].plot([min_price, max_price], [min_price, max_price], 'k--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Stage 1 Predicted Price ($)")
    axes[0].set_title("Stage 1: Actual vs Predicted", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stage 2
    axes[1].scatter(actual_prices, stage2_predictions, alpha=0.6, color='red', s=30)
    min_price = min(actual_prices.min(), stage2_predictions.min())
    max_price = max(actual_prices.max(), stage2_predictions.max())
    axes[1].plot([min_price, max_price], [min_price, max_price], 'k--', lw=2, label='Perfect prediction')
    axes[1].set_xlabel("Actual Price ($)")
    axes[1].set_ylabel("Stage 2 Predicted Price ($)")
    axes[1].set_title("Stage 2: Actual vs Predicted", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/03_scatter_plot.png", dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}/03_scatter_plot.png")
    plt.close()
    
    # Plot 4: Daily error over time
    plt.figure(figsize=(16, 5))
    plt.plot(dates, stage2_error, label="Stage 2 Daily Error", linewidth=1, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.fill_between(dates, stage2_error, 0, where=(stage2_error >= 0), alpha=0.3, color='green', label='Overestimation')
    plt.fill_between(dates, stage2_error, 0, where=(stage2_error < 0), alpha=0.3, color='red', label='Underestimation')
    
    plt.title("Daily Prediction Error Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Error ($)")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/04_daily_error.png", dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}/04_daily_error.png")
    plt.close()


# MAIN TESTING FUNCTION
def main(
    model_path: str = "./models",
    chroma_path: str = "./aapl_memory_v2",
    news_api_key: str = None,
    months: int = 6,
    output_path: str = "./results",
    use_news: bool = True,
):
    """Main testing function"""
    
    logger.info("="*80)
    logger.info("AAPL Stock Price Prediction - Model Accuracy Testing on 2025 Data")
    logger.info("="*80)
    
    # Create output directory
    Path(output_path).mkdir(exist_ok=True)
    
    # Load models
    try:
        models = load_models(model_path)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return
    
    lookback_window = models['lstm'].input_shape[1]
    logger.info(f"Using lookback window: {lookback_window} days")
    
    # Load ChromaDB and encoder (optional)
    collection = None
    encoder = None
    if use_news:
        collection = load_chromadb_collection(chroma_path)
        if collection is not None:
            logger.info("Loading embedding model...")
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Fetch data
    logger.info("\n" + "="*80)
    logger.info("FETCHING DATA")
    logger.info("="*80)
    
    # Get historical data for context
    historical_df = fetch_historical_data(lookback_months=6)
    
    # Get 2025 test data
    test_df = fetch_aapl_data_2025(months=months)
    
    if len(test_df) < lookback_window:
        logger.error(f"Not enough test data. Need {lookback_window} days, got {len(test_df)}")
        return
    
    # Combine for continuous time series
    combined_df = pd.concat([historical_df, test_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['date']).reset_index(drop=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Find where test data starts
    test_start_idx = len(combined_df) - len(test_df)
    
    logger.info(f"Test period: {test_df.iloc[0]['date'].date()} to {test_df.iloc[-1]['date'].date()}")
    logger.info(f"Total rows for prediction: {len(test_df)}")
    
    # PREDICTION LOOP    
    logger.info("\n" + "="*80)
    logger.info("RUNNING PREDICTIONS")
    logger.info("="*80)
    
    results = []
    
    for idx in tqdm(range(test_start_idx, len(combined_df)), desc="Predicting"):
        date = combined_df.iloc[idx]['date']
        actual_price = float(combined_df.iloc[idx]['close'])
        
        try:
            # Prepare OHLC features
            ohlc_for_lstm = prepare_ohlc_features(
                combined_df,
                models['feature_scaler_raw'],
                lookback=lookback_window,
                date_idx=idx
            )
            ohlc_for_cnn = prepare_ohlc_features(
                combined_df,
                models['feature_scaler'],
                lookback=lookback_window,
                date_idx=idx
            )
            
            if ohlc_for_lstm is None or ohlc_for_cnn is None:
                continue
            
            # Stage 1 prediction
            stage1_pred, base_preds, _ = get_stage1_prediction(models, ohlc_for_lstm, ohlc_for_cnn)
            
            if stage1_pred is None:
                continue
            
            # Compute market returns
            market_returns = compute_market_returns(combined_df, idx)
            
            # Get news features
            news_features = _default_news_features()
            if use_news and news_api_key and collection is not None and encoder is not None:
                try:
                    articles = fetch_news_for_date(news_api_key, date)
                    if articles:
                        news_features = process_news_for_date(articles, collection, encoder, top_k=5)
                except Exception as e:
                    logger.debug(f"News processing failed for {date}: {e}")
            
            # Stage 2 prediction
            stage2_pred, correction = get_stage2_prediction(
                models,
                stage1_pred,
                news_features,
                market_returns,
            )
            
            if stage2_pred is None:
                continue
            
            # Calculate errors
            stage1_error = actual_price - stage1_pred
            stage2_error = actual_price - stage2_pred
            
            results.append({
                'date': date,
                'actual_price': actual_price,
                'stage1_prediction': stage1_pred,
                'stage2_prediction': stage2_pred,
                'stage1_error': stage1_error,
                'stage2_error': stage2_error,
                'stage1_error_pct': (stage1_error / actual_price * 100),
                'stage2_error_pct': (stage2_error / actual_price * 100),
                'correction': correction,
                'lstm_pred': base_preds['lstm'],
                'cnn_pred': base_preds['cnn'],
                'return_1d': market_returns['return_1d'],
                'return_5d': market_returns['return_5d'],
                'news_count': int(news_features['news_count']),
            })
        
        except Exception as e:
            logger.debug(f"Prediction failed for {date}: {e}")
            continue
    
    if not results:
        logger.error("No successful predictions made")
        return
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"✓ Generated {len(results_df)} predictions")
    
    # ACCURACY METRICS    
    logger.info("\n" + "="*80)
    logger.info("ACCURACY METRICS")
    logger.info("="*80)
    
    actual = results_df['actual_price'].values
    stage1_pred = results_df['stage1_prediction'].values
    stage2_pred = results_df['stage2_prediction'].values
    
    stage1_metrics = compute_accuracy_metrics(actual, stage1_pred)
    stage2_metrics = compute_accuracy_metrics(actual, stage2_pred)
    
    # Print Stage 1 metrics
    logger.info("\n--- Stage 1 (LSTM + CNN Ensemble) ---")
    for key, value in stage1_metrics.items():
        if 'Accuracy' in key or 'Rate' in key:
            logger.info(f"  {key:.<30} {value:>10.2f}%")
        elif 'R2' in key or 'MAPE' in key:
            logger.info(f"  {key:.<30} {value:>10.4f}")
        else:
            logger.info(f"  {key:.<30} ${value:>10.2f}")
    
    # Print Stage 2 metrics
    logger.info("\n--- Stage 2 (With News-Aware Correction) ---")
    for key, value in stage2_metrics.items():
        if 'Accuracy' in key or 'Rate' in key:
            logger.info(f"  {key:.<30} {value:>10.2f}%")
        elif 'R2' in key or 'MAPE' in key:
            logger.info(f"  {key:.<30} {value:>10.4f}")
        else:
            logger.info(f"  {key:.<30} ${value:>10.2f}")
    
    # Improvement metrics
    logger.info("\n--- Improvement (Stage 2 vs Stage 1) ---")
    for key in stage1_metrics.keys():
        if 'Accuracy' in key or 'Rate' in key:
            improvement = stage2_metrics[key] - stage1_metrics[key]
            logger.info(f"  {key:.<30} {improvement:>+10.2f}%")
        else:
            improvement = stage1_metrics[key] - stage2_metrics[key]  # For errors, lower is better
            pct_improvement = (improvement / stage1_metrics[key] * 100) if stage1_metrics[key] != 0 else 0
            logger.info(f"  {key:.<30} {improvement:>+10.2f} ({pct_improvement:+.1f}%)")
    
    # SAVE RESULTS    
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    # Save predictions CSV
    results_df.to_csv(f"{output_path}/predictions.csv", index=False)
    logger.info(f"✓ Saved: {output_path}/predictions.csv")
    
    # Save metrics JSON
    metrics_summary = {
        'test_period': {
            'start_date': str(results_df.iloc[0]['date'].date()),
            'end_date': str(results_df.iloc[-1]['date'].date()),
            'num_predictions': len(results_df),
        },
        'stage1_metrics': stage1_metrics,
        'stage2_metrics': stage2_metrics,
        'improvement': {key: float(stage2_metrics[key] - stage1_metrics[key]) for key in stage1_metrics.keys()}
    }
    
    with open(f"{output_path}/metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"✓ Saved: {output_path}/metrics.json")
    
    # VISUALIZATION    
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    plot_results(
        actual,
        stage1_pred,
        stage2_pred,
        results_df['date'],
        output_path
    )
    
    # SUMMARY STATISTICS    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    logger.info(f"\nActual Price Statistics:")
    logger.info(f"  Min:  ${actual.min():.2f}")
    logger.info(f"  Max:  ${actual.max():.2f}")
    logger.info(f"  Mean: ${actual.mean():.2f}")
    logger.info(f"  Std:  ${actual.std():.2f}")
    
    logger.info(f"\nStage 1 Prediction Statistics:")
    logger.info(f"  Min:  ${stage1_pred.min():.2f}")
    logger.info(f"  Max:  ${stage1_pred.max():.2f}")
    logger.info(f"  Mean: ${stage1_pred.mean():.2f}")
    logger.info(f"  Std:  ${stage1_pred.std():.2f}")
    
    logger.info(f"\nStage 2 Prediction Statistics:")
    logger.info(f"  Min:  ${stage2_pred.min():.2f}")
    logger.info(f"  Max:  ${stage2_pred.max():.2f}")
    logger.info(f"  Mean: ${stage2_pred.mean():.2f}")
    logger.info(f"  Std:  ${stage2_pred.std():.2f}")
    
    logger.info(f"\nError Statistics (Stage 2):")
    stage2_errors = results_df['stage2_error'].values
    logger.info(f"  Mean Error:     ${stage2_errors.mean():+.2f}")
    logger.info(f"  Median Error:   ${np.median(stage2_errors):+.2f}")
    logger.info(f"  Std Error:      ${stage2_errors.std():.2f}")
    logger.info(f"  Max Error:      ${stage2_errors.max():+.2f}")
    logger.info(f"  Min Error:      ${stage2_errors.min():+.2f}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TESTING COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test AAPL stock prediction model on 2025 data"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models",
        help="Path to directory with saved models"
    )
    
    parser.add_argument(
        "--chroma_path",
        type=str,
        default="./aapl_memory_v2",
        help="Path to ChromaDB collection"
    )
    
    parser.add_argument(
        "--news_api_key",
        type=str,
        default=None,
        help="NewsAPI.org API key for fetching news (optional)"
    )
    
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months of 2025 data to test on (default: 6)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results",
        help="Path to save results and visualizations"
    )
    
    parser.add_argument(
        "--no_news",
        action="store_true",
        help="Disable news-aware correction (Stage 2 will only use market returns)"
    )
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        chroma_path=args.chroma_path,
        news_api_key=args.news_api_key,
        months=args.months,
        output_path=args.output_path,
        use_news=not args.no_news,
    )

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timezone
import numpy as np
import ast
import warnings

# --- ML LIBRARIES ---
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import pickle
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. DATA PROCESSING ---

def safe_parse_list(data):
    """Safely parses a string that looks like a list or JSON."""
    if pd.isna(data):
        return []
    try:
        # Try JSON first
        parsed = ast.literal_eval(str(data))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        # Try JSON parsing
        try:
            import json
            parsed = json.loads(str(data))
            return parsed if isinstance(parsed, list) else []
        except:
            return []

def process_and_engineer_features(csv_path, topic_phrase):
    """Loads the single labeled CSV and engineers all features."""
    print(f"Loading labeled data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Check for label column (try both 'label' and 'Label')
    label_col = None
    for col in ['label', 'Label', 'ai_label', 'ai_label_label']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("Error: No label column found. Looking for: 'label', 'Label', 'ai_label'")
        print(f"Available columns: {list(df.columns)[:10]}...")
        return None

    print(f"Data loaded. Found {len(df)} labeled rows.")
    print(f"Using label column: '{label_col}'")
    
    # Rename label column to 'Label' for consistency
    df['Label'] = df[label_col]
    
    # Filter out unlabeled rows
    df = df[df['Label'].notna()]
    print(f"After filtering unlabeled rows: {len(df)} rows")
    
    # Check label distribution
    print(f"\nLabel distribution:")
    print(df['Label'].value_counts().sort_index())

    # --- 1. Clean and Convert ---
    print("\nCleaning and converting data...")
    df['tweet_date'] = pd.to_datetime(df['tweet_date'], errors='coerce')
    df['user_created'] = pd.to_datetime(df['user_created'], errors='coerce')
    df['tweet_rawContent'] = df['tweet_rawContent'].fillna('')
    
    num_cols = ['user_followersCount', 'user_friendsCount', 'tweet_retweetCount', 
                'tweet_likeCount', 'tweet_quoteCount', 'tweet_replyCount', 
                'user_statusesCount', 'tweet_viewCount']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Drop rows with missing essential data
    initial_len = len(df)
    df = df.dropna(subset=['tweet_rawContent', 'tweet_id'])
    if 'user_created' in df.columns:
        df = df.dropna(subset=['user_created'])
    print(f"Dropped {initial_len - len(df)} rows with missing essential data")
    print(f"Remaining rows: {len(df)}")

    # --- 2. Engineer User Features ---
    print("\nEngineering user features...")
    now = datetime.now(timezone.utc)
    if 'user_created' in df.columns:
        df['account_age_days'] = (now - df['user_created']).dt.days
    else:
        df['account_age_days'] = 0
    
    df['follower_ratio'] = df['user_followersCount'] / (df['user_friendsCount'] + 1)
    
    if 'user_verified' in df.columns:
        df['is_verified'] = df['user_verified'].astype(int)
    else:
        df['is_verified'] = 0
    
    df['user_total_tweets'] = df['user_statusesCount'].fillna(0)

    # --- 3. Engineer NLP Features ---
    print("\nEngineering NLP features... This may take a few minutes.")
    
    # Load Model
    print("Loading SentenceTransformer model...")
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Cosine Similarity
    print("Calculating cosine similarity with topic phrase...")
    print(f"Topic phrase: '{topic_phrase}'")
    topic_embedding = model.encode(topic_phrase, convert_to_tensor=True)
    tweet_embeddings = model.encode(df['tweet_rawContent'].tolist(), 
                                      convert_to_tensor=True, 
                                      show_progress_bar=True)
    cosine_scores = util.cos_sim(topic_embedding, tweet_embeddings)
    df['cosine_similarity'] = cosine_scores.cpu().numpy()[0]

    # Other NLP & Tweet Features
    print("Calculating other NLP/Tweet features...")
    
    # Text-based features
    df['tweet_length'] = df['tweet_rawContent'].str.len()
    df['tweet_word_count'] = df['tweet_rawContent'].str.split().str.len()
    df['has_emoji'] = df['tweet_rawContent'].str.contains(r'[😀-🙏🌀-🗿]', regex=True, na=False).astype(int)
    df['has_question_mark'] = df['tweet_rawContent'].str.contains('?', regex=False, na=False).astype(int)
    df['has_exclamation'] = df['tweet_rawContent'].str.contains('!', regex=False, na=False).astype(int)
    df['uppercase_ratio'] = df['tweet_rawContent'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    
    # Hashtags
    if 'tweet_hashtags' in df.columns:
        df['hashtag_count'] = df['tweet_hashtags'].apply(safe_parse_list).apply(len)
    else:
        df['hashtag_count'] = 0
    
    # Mentions
    if 'tweet_mentionedUsers' in df.columns:
        df['mention_count'] = df['tweet_mentionedUsers'].apply(safe_parse_list).apply(len)
    else:
        df['mention_count'] = 0
    
    # Links
    if 'tweet_links' in df.columns:
        df['has_link'] = df['tweet_links'].apply(safe_parse_list).apply(len).apply(lambda x: 1 if x > 0 else 0)
    else:
        df['has_link'] = 0

    # Engagement features
    df['retweet_count'] = df['tweet_retweetCount'].fillna(0)
    df['like_count'] = df['tweet_likeCount'].fillna(0)
    df['reply_count'] = df['tweet_replyCount'].fillna(0)
    df['quote_count'] = df['tweet_quoteCount'].fillna(0)
    df['view_count'] = df['tweet_viewCount'].fillna(0)
    
    # Engagement ratios
    df['engagement_total'] = df['retweet_count'] + df['like_count'] + df['reply_count'] + df['quote_count']
    df['engagement_per_follower'] = df['engagement_total'] / (df['user_followersCount'] + 1)
    
    # Time-based features
    if 'tweet_date' in df.columns:
        df['hour_of_day'] = df['tweet_date'].dt.hour
        df['day_of_week'] = df['tweet_date'].dt.dayofweek
    else:
        df['hour_of_day'] = 12
        df['day_of_week'] = 3

    print("Feature engineering complete.")
    return df

# --- 2. TRAIN AND COMPARE MODELS ---

def run_ml_comparison(df, feature_list, use_smote=False, save_best_model=True):
    """Trains and compares 3 models, then prints a report."""
    
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    # Define our X (features) and y (target)
    X = df[feature_list].copy()
    y = df['Label'].copy()
    
    # Ensure labels are 0/1
    y = y.astype(int)

    # Handle any potential remaining NaN values
    X = X.fillna(0)
    
    # Check for missing features
    missing_features = [f for f in feature_list if f not in X.columns]
    if missing_features:
        print(f"⚠ Warning: Missing features: {missing_features}")
        feature_list = [f for f in feature_list if f in X.columns]
        X = X[feature_list]

    # Check if we have enough data
    if len(df) < 20:
        print(f"⚠ Warning: Only {len(df)} rows available. Need at least 20 for meaningful training.")
        return None

    # Check class imbalance
    class_counts = y.value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    print(f"\nClass distribution:")
    print(f"  Genuine (0): {class_counts.get(0, 0)}")
    print(f"  Inauthentic (1): {class_counts.get(1, 0)}")
    print(f"  Imbalance ratio: {imbalance_ratio:.3f}")
    
    if imbalance_ratio < 0.3:
        print("⚠ Warning: Significant class imbalance detected!")
        if use_smote:
            print("  Using SMOTE to balance classes...")
        else:
            print("  Consider using SMOTE (set use_smote=True)")

    # Split the data: 80% for training, 20% for testing
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If stratification fails (not enough samples per class), use regular split
        print("⚠ Warning: Stratified split failed, using regular split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} rows")
    print(f"  Testing: {len(X_test)} rows")
    print(f"  Training labels - Genuine: {(y_train == 0).sum()}, Inauthentic: {(y_train == 1).sum()}")
    print(f"  Testing labels - Genuine: {(y_test == 0).sum()}, Inauthentic: {(y_test == 1).sum()}")
    
    # Apply SMOTE if requested
    if use_smote and imbalance_ratio < 0.3:
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"\nAfter SMOTE:")
            print(f"  Training: {len(X_train)} rows")
            print(f"  Training labels - Genuine: {(y_train == 0).sum()}, Inauthentic: {(y_train == 1).sum()}")
        except Exception as e:
            print(f"⚠ Warning: SMOTE failed: {e}")
            print("  Continuing without SMOTE...")

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Define Models ---
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        
        try:
            if name == "Logistic Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Show Feature Importance
                if hasattr(model, 'feature_importances_'):
                    print("\nTop 10 Most Important Features:")
                    imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
                    for feat, score in imp.head(10).items():
                        print(f"  {feat}: {score:.4f}")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate ROC AUC if we have probabilities
            try:
                if name == "Logistic Regression":
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = None
            
            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'model': model,
                'scaler': scaler if name == "Logistic Regression" else None,
                'predictions': y_pred
            }
            
            # --- Print Results ---
            print(f"\nAccuracy: {accuracy:.4f}")
            if roc_auc:
                print(f"ROC AUC: {roc_auc:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Genuine (0)', 'Inauthentic (1)']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Genuine  Inauthentic")
            print(f"Actual Genuine    {cm[0][0]:4d}      {cm[0][1]:4d}")
            print(f"       Inauthentic {cm[1][0]:4d}      {cm[1][1]:4d}")
            
            # Cross-validation score
            print(f"\nCross-validation (5-fold):")
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                if name == "Logistic Regression":
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"  CV failed: {e}")
            
        except Exception as e:
            print(f"⚠ Error training {name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    best_model_name = None
    best_accuracy = 0
    
    for name, result in results.items():
        acc = result['accuracy']
        roc = result.get('roc_auc', 'N/A')
        roc_str = f", ROC AUC: {roc:.4f}" if roc else ""
        print(f"{name}: {acc:.4f} accuracy{roc_str}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
    
    if best_model_name and save_best_model:
        print(f"\n{'='*60}")
        print(f"Best model: {best_model_name} ({best_accuracy:.4f} accuracy)")
        print("Saving best model...")
        
        best_result = results[best_model_name]
        model = best_result['model']
        scaler = best_result.get('scaler')
        
        # Save model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'features': feature_list,
                'accuracy': best_accuracy,
                'model_name': best_model_name
            }, f)
        print(f"✓ Model saved to: {model_file}")
        
        # Save scaler if exists
        if scaler:
            scaler_file = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✓ Scaler saved to: {scaler_file}")
    
    return results

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Configuration
    LABELED_CSV_FILE = "combined_output/all_data_labeled.csv"
    
    # Topic phrases for your hashtags (you can modify these)
    # These represent what each hashtag is about
    TOPIC_PHRASES = {
        "default": "Riyadh Season events entertainment concerts and activities",
        "#موسم_الرياض": "Riyadh Season events entertainment concerts and activities",
        "#النصر_نيوم": "Al-Nassr football club Neom matches and news",
        "#دوري_روشن_السعودي": "Saudi Roshan League football matches and news",
        "#جيش_البرهان_يرفض_الهدنه": "Sudan military news and political developments"
    }
    
    # Use default topic phrase (you can change this)
    TOPIC_PHRASE = TOPIC_PHRASES["default"]
    
    # Define all the features our model will use
    ML_FEATURES = [
        'cosine_similarity',
        'hashtag_count',
        'mention_count',
        'has_link',
        'tweet_length',
        'tweet_word_count',
        'has_emoji',
        'has_question_mark',
        'has_exclamation',
        'uppercase_ratio',
        'account_age_days',
        'follower_ratio',
        'is_verified',
        'user_followersCount',
        'user_friendsCount',
        'user_total_tweets',
        'retweet_count',
        'like_count',
        'reply_count',
        'quote_count',
        'view_count',
        'engagement_total',
        'engagement_per_follower',
        'hour_of_day',
        'day_of_week'
    ]
    
    # --- Run the full pipeline ---
    print("="*60)
    print("TWITTER BOT DETECTION - ML TRAINING")
    print("="*60)
    
    # 1. Process all raw tweets
    processed_df = process_and_engineer_features(LABELED_CSV_FILE, TOPIC_PHRASE)
    
    if processed_df is not None:
        # Save processed data
        output_file = "combined_output/all_data_processed.csv"
        processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ Processed data saved to: {output_file}")
        
        # 2. Run the ML comparison
        # Set use_smote=True if you have class imbalance issues
        results = run_ml_comparison(
            processed_df, 
            ML_FEATURES,
            use_smote=False,  # Set to True if you have class imbalance
            save_best_model=True
        )
        
        if results:
            print("\n✓ Model training complete!")
            print("\nNext steps:")
            print("  1. Review the model performance metrics")
            print("  2. Check feature importance to understand what signals bots")
            print("  3. Use the saved model to predict on new tweets")
    else:
        print("Training Canceled: Could not process data.")


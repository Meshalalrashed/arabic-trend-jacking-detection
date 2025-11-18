import pandas as pd
import numpy as np
import ast
import json
import warnings
from datetime import datetime, timezone

# --- CLUSTERING LIBRARIES ---
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ UMAP not available. Install with: pip install umap-learn")
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')

# --- DATA PROCESSING ---

def safe_parse_list(data):
    """Safely parses a string that looks like a list or JSON."""
    if pd.isna(data):
        return []
    try:
        parsed = ast.literal_eval(str(data))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(str(data))
            return parsed if isinstance(parsed, list) else []
        except:
            return []

def filter_riyadh_season_tweets(csv_path):
    """Filter tweets that contain Riyadh Season hashtag."""
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    print(f"Total tweets loaded: {len(df)}")
    
    # Filter for Riyadh Season hashtag
    # Check if tweet_hashtags contains "موسم_الرياض" or "RiyadhSeason" or similar
    riyadh_keywords = ['موسم_الرياض', 'RiyadhSeason', 'riyadhseason', 'RIYADHSEASON']
    
    def has_riyadh_hashtag(hashtags_str):
        if pd.isna(hashtags_str):
            return False
        hashtags_list = safe_parse_list(hashtags_str)
        hashtags_lower = [str(h).lower() for h in hashtags_list]
        return any(keyword.lower() in ' '.join(hashtags_lower) for keyword in riyadh_keywords)
    
    if 'tweet_hashtags' in df.columns:
        mask = df['tweet_hashtags'].apply(has_riyadh_hashtag)
        df_filtered = df[mask].copy()
        print(f"Tweets with Riyadh Season hashtag: {len(df_filtered)}")
    else:
        print("⚠ Warning: 'tweet_hashtags' column not found. Using all data.")
        df_filtered = df.copy()
    
    return df_filtered

def preprocess_text(text):
    """Preprocess text for better clustering."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove mentions (keep hashtags)
    text = re.sub(r'@\w+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def create_text_embeddings(df, use_combined_features=False):
    """Create semantic embeddings for tweet content with optional feature combination."""
    print("\nCreating text embeddings for semantic clustering...")
    
    # Preprocess texts
    print("Preprocessing texts...")
    df['tweet_processed'] = df['tweet_rawContent'].apply(preprocess_text)
    tweet_texts = df['tweet_processed'].fillna('').tolist()
    
    # Load multilingual model for Arabic/English support
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Loaded multilingual model: paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"Error loading multilingual model: {e}")
        print("Trying alternative model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    print("Encoding tweets... This may take a few minutes...")
    embeddings = model.encode(tweet_texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    
    # Optionally combine with other features
    if use_combined_features:
        print("Combining with additional features...")
        feature_list = []
        
        # Hashtag count (normalized)
        if 'hashtag_count' in df.columns:
            hashtag_feat = df['hashtag_count'].fillna(0).values.reshape(-1, 1)
            feature_list.append(hashtag_feat)
        
        # Has link (binary)
        if 'has_link' in df.columns:
            link_feat = df['has_link'].fillna(0).values.reshape(-1, 1)
            feature_list.append(link_feat)
        
        # Engagement features (normalized)
        if 'like_count' in df.columns:
            like_feat = np.log1p(df['like_count'].fillna(0)).values.reshape(-1, 1)
            feature_list.append(like_feat)
        
        if feature_list:
            additional_features = np.hstack(feature_list)
            # Normalize additional features
            scaler = StandardScaler()
            additional_features_scaled = scaler.fit_transform(additional_features)
            # Combine with embeddings
            embeddings = np.hstack([embeddings, additional_features_scaled])
            print(f"Combined embeddings shape: {embeddings.shape}")
    
    return embeddings, model

def extract_keywords_from_cluster(texts, top_n=5):
    """Extract top keywords from a cluster of texts."""
    if not texts or len(texts) == 0:
        return []
    
    # Combine all texts
    combined_text = ' '.join([str(t) for t in texts if pd.notna(t)])
    
    # Extract words (handle both Arabic and English)
    # Arabic words: \u0600-\u06FF, English: a-zA-Z
    words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+', combined_text.lower())
    
    # Remove very short words
    words = [w for w in words if len(w) > 2]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get top keywords
    top_keywords = [word for word, count in word_counts.most_common(top_n)]
    
    return top_keywords

def auto_detect_cluster_labels(df, cluster_labels, embeddings, model, method='keywords'):
    """Automatically detect labels for clusters using keyword extraction or topic modeling."""
    print("\nAuto-detecting cluster labels...")
    
    cluster_categories = {}
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Skip noise
            cluster_categories[cluster_id] = 'Noise/Outliers'
            continue
        
        # Get tweets for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_tweets = df[cluster_mask]['tweet_rawContent'].tolist()
        
        if len(cluster_tweets) == 0:
            cluster_categories[cluster_id] = f'Cluster {cluster_id}'
            continue
        
        if method == 'keywords':
            # Method 1: Extract top keywords
            keywords = extract_keywords_from_cluster(cluster_tweets, top_n=3)
            
            if keywords:
                # Create label from keywords
                label = ' / '.join(keywords[:3])
                cluster_categories[cluster_id] = label
                print(f"  Cluster {cluster_id}: {label}")
            else:
                cluster_categories[cluster_id] = f'Cluster {cluster_id}'
        
        elif method == 'topic_modeling':
            # Method 2: Use topic modeling (LDA)
            try:
                if len(cluster_tweets) >= 5:  # Need minimum tweets for LDA
                    vectorizer = TfidfVectorizer(max_features=50, stop_words=None)
                    tfidf_matrix = vectorizer.fit_transform(cluster_tweets)
                    
                    # Apply LDA
                    n_topics = min(1, len(cluster_tweets) // 5)  # At least 5 tweets per topic
                    if n_topics > 0:
                        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
                        lda.fit(tfidf_matrix)
                        
                        # Get top words for the topic
                        feature_names = vectorizer.get_feature_names_out()
                        top_words = []
                        for topic_idx, topic in enumerate(lda.components_):
                            top_words_idx = topic.argsort()[-3:][::-1]
                            top_words = [feature_names[i] for i in top_words_idx]
                        
                        label = ' / '.join(top_words[:3])
                        cluster_categories[cluster_id] = label
                        print(f"  Cluster {cluster_id}: {label}")
                    else:
                        keywords = extract_keywords_from_cluster(cluster_tweets, top_n=3)
                        label = ' / '.join(keywords[:3]) if keywords else f'Cluster {cluster_id}'
                        cluster_categories[cluster_id] = label
                        print(f"  Cluster {cluster_id}: {label}")
                else:
                    # Fallback to keywords for small clusters
                    keywords = extract_keywords_from_cluster(cluster_tweets, top_n=3)
                    label = ' / '.join(keywords[:3]) if keywords else f'Cluster {cluster_id}'
                    cluster_categories[cluster_id] = label
                    print(f"  Cluster {cluster_id}: {label}")
            except Exception as e:
                # Fallback to keywords if LDA fails
                keywords = extract_keywords_from_cluster(cluster_tweets, top_n=3)
                label = ' / '.join(keywords[:3]) if keywords else f'Cluster {cluster_id}'
                cluster_categories[cluster_id] = label
                print(f"  Cluster {cluster_id}: {label} (fallback)")
        
        elif method == 'representative_tweet':
            # Method 3: Use most representative tweet (closest to cluster center)
            cluster_embeddings = embeddings[cluster_mask]
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # Find tweet closest to center
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_idx = np.argmin(distances)
            representative_tweet = cluster_tweets[closest_idx]
            
            # Extract keywords from representative tweet
            keywords = extract_keywords_from_cluster([representative_tweet], top_n=3)
            label = ' / '.join(keywords[:3]) if keywords else f'Cluster {cluster_id}'
            cluster_categories[cluster_id] = label
            print(f"  Cluster {cluster_id}: {label}")
    
    return cluster_categories

# --- CLUSTERING ALGORITHMS ---

def apply_kmeans(X_scaled, n_clusters=5, params=None):
    """Apply K-Means clustering.
    
    Hyperparameters:
    - n_clusters: Number of clusters to form
    - n_init: Number of times to run with different centroids (default: 10)
    - max_iter: Maximum iterations per run (default: 300)
    - random_state: Seed for reproducibility (default: 42)
    - algorithm: 'lloyd', 'elkan', 'auto', 'full' (default: 'lloyd')
    """
    if params is None:
        params = {'n_init': 10, 'max_iter': 300, 'random_state': 42, 'algorithm': 'lloyd'}
    
    print(f"\n{'='*60}")
    print("K-MeANS CLUSTERING")
    print(f"{'='*60}")
    print(f"Hyperparameters: n_clusters={n_clusters}, n_init={params['n_init']}, max_iter={params['max_iter']}")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=params['n_init'],
        max_iter=params['max_iter'],
        random_state=params['random_state'],
        algorithm=params.get('algorithm', 'lloyd')
    )
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} tweets ({count/len(labels)*100:.1f}%)")
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    
    print(f"\nSilhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f} (higher is better)")
    
    return labels, {'silhouette': silhouette, 'davies_bouldin': davies_bouldin, 'calinski_harabasz': calinski_harabasz}

def apply_dbscan(X_scaled, eps=0.5, min_samples=5, params=None):
    """Apply DBSCAN clustering.
    
    Hyperparameters:
    - eps: Maximum distance between samples in same cluster (default: 0.5)
    - min_samples: Minimum samples to form a cluster (default: 5)
    - metric: Distance metric - 'euclidean', 'cosine', 'manhattan' (default: 'euclidean')
    - algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
    """
    if params is None:
        params = {'metric': 'euclidean', 'algorithm': 'auto'}
    
    print(f"\n{'='*60}")
    print("DBSCAN CLUSTERING")
    print(f"{'='*60}")
    print(f"Hyperparameters: eps={eps}, min_samples={min_samples}, metric={params['metric']}")
    
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=params.get('metric', 'euclidean'),
        algorithm=params.get('algorithm', 'auto')
    )
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points (outliers): {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        if cluster_id == -1:
            print(f"  Noise/Outliers: {count} tweets")
        else:
            print(f"  Cluster {cluster_id}: {count} tweets ({count/len(labels)*100:.1f}%)")
    
    # Calculate metrics (only if we have clusters)
    if n_clusters > 1:
        # Filter out noise points for metrics
        mask = labels != -1
        if mask.sum() > 1:
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(X_scaled[mask], labels[mask])
            calinski_harabasz = calinski_harabasz_score(X_scaled[mask], labels[mask])
            
            print(f"\nMetrics (excluding noise points):")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
            print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
            
            metrics = {'silhouette': silhouette, 'davies_bouldin': davies_bouldin, 'calinski_harabasz': calinski_harabasz}
        else:
            metrics = {}
    else:
        print("\n⚠ Warning: Not enough clusters for metrics calculation")
        metrics = {}
    
    return labels, metrics

def apply_hierarchical(X_scaled, n_clusters=5, params=None):
    """Apply Hierarchical/Agglomerative clustering.
    
    Hyperparameters:
    - n_clusters: Number of clusters to form
    - linkage: Linkage criterion - 'ward', 'complete', 'average', 'single' (default: 'ward')
    - metric: Distance metric - 'euclidean', 'l1', 'l2', 'manhattan', 'cosine' (default: 'euclidean')
    """
    if params is None:
        params = {'linkage': 'ward', 'metric': 'euclidean'}
    
    print(f"\n{'='*60}")
    print("HIERARCHICAL CLUSTERING")
    print(f"{'='*60}")
    print(f"Hyperparameters: n_clusters={n_clusters}, linkage={params['linkage']}, metric={params.get('metric', 'euclidean')}")
    
    # Build kwargs - 'ward' linkage only works with euclidean metric
    clustering_kwargs = {
        'n_clusters': n_clusters,
        'linkage': params.get('linkage', 'ward')
    }
    
    # Only add metric if linkage is not 'ward' (ward requires euclidean)
    if params.get('linkage', 'ward') != 'ward':
        clustering_kwargs['metric'] = params.get('metric', 'euclidean')
    
    hierarchical = AgglomerativeClustering(**clustering_kwargs)
    labels = hierarchical.fit_predict(X_scaled)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} tweets ({count/len(labels)*100:.1f}%)")
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    
    print(f"\nSilhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    
    return labels, {'silhouette': silhouette, 'davies_bouldin': davies_bouldin, 'calinski_harabasz': calinski_harabasz}

def apply_gaussian_mixture(X_scaled, n_components=5, params=None):
    """Apply Gaussian Mixture Model clustering.
    
    Hyperparameters:
    - n_components: Number of mixture components
    - covariance_type: 'full', 'tied', 'diag', 'spherical' (default: 'full')
    - max_iter: Maximum iterations (default: 100)
    - random_state: Seed for reproducibility (default: 42)
    - tol: Convergence threshold (default: 0.001)
    """
    if params is None:
        params = {'covariance_type': 'full', 'max_iter': 100, 'random_state': 42, 'tol': 0.001}
    
    print(f"\n{'='*60}")
    print("GAUSSIAN MIXTURE MODEL CLUSTERING")
    print(f"{'='*60}")
    print(f"Hyperparameters: n_components={n_components}, covariance_type={params['covariance_type']}, max_iter={params['max_iter']}")
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=params.get('covariance_type', 'full'),
        max_iter=params.get('max_iter', 100),
        random_state=params.get('random_state', 42),
        tol=params.get('tol', 0.001)
    )
    labels = gmm.fit_predict(X_scaled)
    
    print(f"Number of components: {n_components}")
    print(f"Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} tweets ({count/len(labels)*100:.1f}%)")
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    
    print(f"\nSilhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    
    return labels, {'silhouette': silhouette, 'davies_bouldin': davies_bouldin, 'calinski_harabasz': calinski_harabasz}

def analyze_clusters(df, labels, cluster_categories, algorithm_name):
    """Analyze cluster characteristics with semantic categories."""
    print(f"\n{'='*60}")
    print(f"CLUSTER ANALYSIS - {algorithm_name}")
    print(f"{'='*60}")
    
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    df_clustered['cluster_category'] = df_clustered['cluster'].map(cluster_categories)
    
    # Analyze each cluster
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        category = cluster_categories.get(cluster_id, 'Unknown')
        
        if cluster_id == -1:  # DBSCAN noise
            print(f"\n--- Noise/Outliers (Cluster -1) - {category} ---")
        else:
            print(f"\n--- Cluster {cluster_id}: {category} ({len(cluster_data)} tweets) ---")
        
        # Key statistics
        if 'user_followersCount' in cluster_data.columns:
            print(f"Average followers: {cluster_data['user_followersCount'].mean():.0f}")
        if 'like_count' in cluster_data.columns:
            print(f"Average likes: {cluster_data['like_count'].mean():.0f}")
        if 'retweet_count' in cluster_data.columns:
            print(f"Average retweets: {cluster_data['retweet_count'].mean():.0f}")
        if 'hashtag_count' in cluster_data.columns:
            print(f"Average hashtags: {cluster_data['hashtag_count'].mean():.1f}")
        if 'has_link' in cluster_data.columns:
            print(f"Tweets with links: {cluster_data['has_link'].sum()} ({cluster_data['has_link'].mean()*100:.1f}%)")
        
        # Sample tweets
        if len(cluster_data) > 0:
            sample_tweets = cluster_data['tweet_rawContent'].head(5).tolist()
            print(f"\nSample tweets:")
            for i, tweet in enumerate(sample_tweets, 1):
                tweet_preview = str(tweet)[:150] + "..." if len(str(tweet)) > 150 else str(tweet)
                print(f"  {i}. {tweet_preview}")

def visualize_clusters(X_scaled, labels_dict, feature_names, output_dir="clustering_results"):
    """Visualize clusters using PCA."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\n{'='*60}")
    print("VISUALIZING CLUSTERS")
    print(f"{'='*60}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Create subplots for each algorithm
    n_algorithms = len(labels_dict)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (algo_name, labels) in enumerate(labels_dict.items()):
        ax = axes[idx]
        
        # Plot clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', s=50, alpha=0.5, label='Noise')
            else:
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], s=50, alpha=0.6, label=f'Cluster {label}')
        
        ax.set_title(f'{algo_name} Clustering', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'clustering_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - HYPERPARAMETERS
    # ========================================================================
    INPUT_CSV_FILE = "combined_output/all_data_labeled.csv"
    OUTPUT_DIR = "clustering_results"
    
    # --- CLUSTERING HYPERPARAMETERS ---
    
    # K-Means Hyperparameters
    KMEANS_PARAMS = {
        'n_clusters': None,  # Auto-determined if None (based on data size)
        'n_init': 10,        # Number of times to run with different centroids
        'max_iter': 300,     # Maximum iterations per run
        'random_state': 42,  # For reproducibility
        'algorithm': 'lloyd'  # 'lloyd', 'elkan', 'auto', 'full'
    }
    
    # DBSCAN Hyperparameters
    DBSCAN_PARAMS = {
        'eps': 0.8,          # Maximum distance between samples in same cluster
        'min_samples': 5,    # Minimum samples to form a cluster
        'metric': 'euclidean',  # Distance metric: 'euclidean', 'cosine', 'manhattan'
        'algorithm': 'auto'  # 'auto', 'ball_tree', 'kd_tree', 'brute'
    }
    
    # Hierarchical/Agglomerative Hyperparameters
    HIERARCHICAL_PARAMS = {
        'n_clusters': None,  # Auto-determined if None
        'linkage': 'ward',   # Linkage criterion: 'ward', 'complete', 'average', 'single'
        'metric': 'euclidean'  # Distance metric: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine' (use 'metric' for newer sklearn)
    }
    
    # Gaussian Mixture Model Hyperparameters
    GMM_PARAMS = {
        'n_components': None,  # Auto-determined if None
        'covariance_type': 'full',  # 'full', 'tied', 'diag', 'spherical'
        'max_iter': 100,      # Maximum iterations
        'random_state': 42,   # For reproducibility
        'tol': 0.001          # Convergence threshold
    }
    
    # --- DIMENSIONALITY REDUCTION ---
    USE_DIM_REDUCTION = True  # Use UMAP/t-SNE/PCA before clustering (improves separation)
    DIM_REDUCTION_METHOD = 'tsne'  # 'umap', 'tsne', 'pca', or None
    N_COMPONENTS = 50  # Number of dimensions after reduction (None = auto)
    
    # --- FEATURE ENGINEERING ---
    USE_COMBINED_FEATURES = False  # Combine text embeddings with hashtag/engagement features
    USE_COSINE_DISTANCE = True  # Use cosine distance (better for semantic similarity)
    
    # --- AUTO-LABEL DETECTION ---
    DETECTION_METHOD = 'topic_modeling'  # 'keywords', 'topic_modeling', 'representative_tweet'
    
    # Topic Modeling Hyperparameters (if using topic_modeling method)
    TOPIC_MODELING_PARAMS = {
        'max_features': 50,   # Max features for TF-IDF
        'n_topics': 1,        # Number of topics per cluster
        'max_iter': 10        # LDA iterations
    }
    
    # --- ADAPTIVE CLUSTERING ---
    AUTO_N_CLUSTERS = True    # If True, automatically determines n_clusters
    MIN_CLUSTERS = 2          # Minimum number of clusters
    MAX_CLUSTERS = 10         # Maximum number of clusters
    CLUSTERS_PER_50_TWEETS = 1  # How many clusters per 50 tweets
    
    print("="*60)
    print("RIYADH SEASON TWEETS - CLUSTERING ANALYSIS")
    print("="*60)
    print("\nHYPERPARAMETERS:")
    print(f"  K-Means: n_init={KMEANS_PARAMS['n_init']}, max_iter={KMEANS_PARAMS['max_iter']}")
    print(f"  DBSCAN: eps={DBSCAN_PARAMS['eps']}, min_samples={DBSCAN_PARAMS['min_samples']}")
    print(f"  Hierarchical: linkage={HIERARCHICAL_PARAMS['linkage']}")
    print(f"  GMM: covariance_type={GMM_PARAMS['covariance_type']}, max_iter={GMM_PARAMS['max_iter']}")
    print(f"  Label Detection: {DETECTION_METHOD}")
    print("="*60)
    
    # 1. Filter for Riyadh Season tweets
    df = filter_riyadh_season_tweets(INPUT_CSV_FILE)
    
    if df is None or len(df) == 0:
        print("Error: No data found or filtering failed.")
        exit(1)
    
    # 2. Clean data
    df['tweet_rawContent'] = df['tweet_rawContent'].fillna('')
    df = df.dropna(subset=['tweet_rawContent', 'tweet_id'])
    
    # Add some basic features for analysis
    if 'tweet_retweetCount' in df.columns:
        df['retweet_count'] = pd.to_numeric(df['tweet_retweetCount'], errors='coerce').fillna(0)
    else:
        df['retweet_count'] = 0
    
    if 'tweet_likeCount' in df.columns:
        df['like_count'] = pd.to_numeric(df['tweet_likeCount'], errors='coerce').fillna(0)
    else:
        df['like_count'] = 0
    
    if 'tweet_hashtags' in df.columns:
        df['hashtag_count'] = df['tweet_hashtags'].apply(safe_parse_list).apply(len)
    else:
        df['hashtag_count'] = 0
    
    if 'tweet_links' in df.columns:
        df['has_link'] = df['tweet_links'].apply(safe_parse_list).apply(len).apply(lambda x: 1 if x > 0 else 0)
    else:
        df['has_link'] = 0
    
    if 'user_followersCount' in df.columns:
        df['user_followersCount'] = pd.to_numeric(df['user_followersCount'], errors='coerce').fillna(0)
    else:
        df['user_followersCount'] = 0
    
    if len(df) < 10:
        print(f"⚠ Warning: Only {len(df)} tweets available. Need at least 10 for clustering.")
        exit(1)
    
    # 3. Create text embeddings for semantic clustering
    embeddings, model = create_text_embeddings(df, use_combined_features=USE_COMBINED_FEATURES)
    
    # Apply dimensionality reduction if requested
    if USE_DIM_REDUCTION and DIM_REDUCTION_METHOD:
        print(f"\nApplying {DIM_REDUCTION_METHOD.upper()} dimensionality reduction...")
        
        if DIM_REDUCTION_METHOD.lower() == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=N_COMPONENTS if N_COMPONENTS else min(50, embeddings.shape[1] // 2),
                random_state=42,
                metric='cosine' if USE_COSINE_DISTANCE else 'euclidean',
                n_neighbors=15,
                min_dist=0.1
            )
            X_scaled = reducer.fit_transform(embeddings)
            print(f"Reduced to {X_scaled.shape[1]} dimensions using UMAP")
        
        elif DIM_REDUCTION_METHOD.lower() == 'tsne':
            # t-SNE Barnes-Hut only supports n_components <= 3
            # For more components, use exact method (slower)
            n_components = N_COMPONENTS if N_COMPONENTS else 2
            if n_components > 3:
                print(f"⚠ Warning: t-SNE with n_components > 3 uses exact method (slower). Limiting to 3.")
                n_components = 3
            
            # Use exact method if n_components > 3, otherwise use barnes_hut (faster)
            method = 'exact' if n_components > 3 else 'barnes_hut'
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(df) // 4),
                n_iter=1000,
                method=method
            )
            X_scaled = reducer.fit_transform(embeddings)
            print(f"Reduced to {X_scaled.shape[1]} dimensions using t-SNE ({method} method)")
        
        elif DIM_REDUCTION_METHOD.lower() == 'pca':
            n_components = N_COMPONENTS if N_COMPONENTS else min(50, embeddings.shape[1] // 2)
            reducer = PCA(n_components=n_components, random_state=42)
            X_scaled = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            print(f"Reduced to {X_scaled.shape[1]} dimensions using PCA (explained variance: {explained_var:.2%})")
        
        else:
            X_scaled = embeddings
            print("⚠ Invalid reduction method, using original embeddings")
    else:
        # Use embeddings directly (normalize if using cosine distance)
        if USE_COSINE_DISTANCE:
            # Normalize embeddings for cosine distance
            from sklearn.preprocessing import normalize
            X_scaled = normalize(embeddings, norm='l2')
            print("Using cosine-normalized embeddings")
        else:
            # Standard scaling for euclidean distance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(embeddings)
            print("Using standard-scaled embeddings")
    
    # 4. Apply clustering algorithms
    labels_dict = {}
    metrics_dict = {}
    
    # Determine optimal number of clusters
    if AUTO_N_CLUSTERS:
        n_clusters = min(MAX_CLUSTERS, max(MIN_CLUSTERS, len(df) // (50 // CLUSTERS_PER_50_TWEETS)))
    else:
        n_clusters = KMEANS_PARAMS.get('n_clusters', 5)
    
    if KMEANS_PARAMS.get('n_clusters') is not None:
        n_clusters = KMEANS_PARAMS['n_clusters']
    if HIERARCHICAL_PARAMS.get('n_clusters') is not None:
        n_clusters_hierarchical = HIERARCHICAL_PARAMS['n_clusters']
    else:
        n_clusters_hierarchical = n_clusters
    if GMM_PARAMS.get('n_components') is not None:
        n_components_gmm = GMM_PARAMS['n_components']
    else:
        n_components_gmm = n_clusters
    
    print(f"\nUsing {n_clusters} clusters for algorithms that require it")
    
    # K-Means
    labels_kmeans, metrics_kmeans = apply_kmeans(X_scaled, n_clusters=n_clusters, params=KMEANS_PARAMS)
    labels_dict['K-Means'] = labels_kmeans
    metrics_dict['K-Means'] = metrics_kmeans
    
    # DBSCAN
    labels_dbscan, metrics_dbscan = apply_dbscan(
        X_scaled, 
        eps=DBSCAN_PARAMS['eps'], 
        min_samples=DBSCAN_PARAMS['min_samples'],
        params=DBSCAN_PARAMS
    )
    labels_dict['DBSCAN'] = labels_dbscan
    metrics_dict['DBSCAN'] = metrics_dbscan
    
    # Hierarchical
    labels_hierarchical, metrics_hierarchical = apply_hierarchical(
        X_scaled, 
        n_clusters=n_clusters_hierarchical,
        params=HIERARCHICAL_PARAMS
    )
    labels_dict['Hierarchical'] = labels_hierarchical
    metrics_dict['Hierarchical'] = metrics_hierarchical
    
    # Gaussian Mixture
    labels_gmm, metrics_gmm = apply_gaussian_mixture(
        X_scaled, 
        n_components=n_components_gmm,
        params=GMM_PARAMS
    )
    labels_dict['Gaussian Mixture'] = labels_gmm
    metrics_dict['Gaussian Mixture'] = metrics_gmm
    
    # 5. Auto-detect cluster labels
    print("\n" + "="*60)
    print("AUTO-DETECTING CLUSTER LABELS")
    print("="*60)
    
    cluster_categories_dict = {}
    for algo_name, labels in labels_dict.items():
        print(f"\n{algo_name}:")
        categories = auto_detect_cluster_labels(df, labels, embeddings, model, method=DETECTION_METHOD)
        cluster_categories_dict[algo_name] = categories
    
    # 6. Analyze clusters
    for algo_name, labels in labels_dict.items():
        analyze_clusters(df, labels, cluster_categories_dict[algo_name], algo_name)
    
    # 7. Visualize clusters
    visualize_clusters(X_scaled, labels_dict, ['text_embedding'], OUTPUT_DIR)
    
    # 8. Save results with semantic categories
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for algo_name, labels in labels_dict.items():
        df_result = df.copy()
        df_result['cluster'] = labels
        df_result['cluster_category'] = df_result['cluster'].map(cluster_categories_dict[algo_name])
        output_file = os.path.join(OUTPUT_DIR, f'riyadh_season_clusters_{algo_name.lower().replace(" ", "_")}.csv')
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ Results saved to: {output_file}")
        
        # Also save summary by category
        category_summary = df_result.groupby('cluster_category').agg({
            'cluster': 'count',
            'like_count': 'mean',
            'retweet_count': 'mean',
            'hashtag_count': 'mean'
        }).rename(columns={'cluster': 'tweet_count'})
        summary_file = os.path.join(OUTPUT_DIR, f'category_summary_{algo_name.lower().replace(" ", "_")}.csv')
        category_summary.to_csv(summary_file, encoding='utf-8-sig')
        print(f"✓ Category summary saved to: {summary_file}")
    
    # 8. Summary
    print(f"\n{'='*60}")
    print("CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print("\nAlgorithm Comparison (Silhouette Score - higher is better):")
    for algo_name, metrics in metrics_dict.items():
        if 'silhouette' in metrics:
            print(f"  {algo_name}: {metrics['silhouette']:.4f}")
    
    print("\n✓ Clustering analysis complete!")
    print(f"✓ All results saved to: {OUTPUT_DIR}/")


import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys

# Import clustering functions from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cluster_riyadh_season import (
    filter_riyadh_season_tweets,
    create_text_embeddings,
    auto_detect_cluster_labels,
    apply_kmeans,
    apply_dbscan,
    apply_hierarchical,
    apply_gaussian_mixture,
    analyze_clusters,
    visualize_clusters
)

class ClusteringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Riyadh Season Clustering Tool")
        self.root.geometry("900x800")
        
        self.df = None
        self.embeddings = None
        self.model = None
        self.results = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Riyadh Season Tweet Clustering",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path_var = tk.StringVar(value="combined_output/all_data_labeled.csv")
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        
        # Hyperparameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Clustering Hyperparameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for different algorithms
        notebook = ttk.Notebook(params_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # K-Means tab
        kmeans_frame = ttk.Frame(notebook, padding="10")
        notebook.add(kmeans_frame, text="K-Means")
        self.setup_kmeans_params(kmeans_frame)
        
        # DBSCAN tab
        dbscan_frame = ttk.Frame(notebook, padding="10")
        notebook.add(dbscan_frame, text="DBSCAN")
        self.setup_dbscan_params(dbscan_frame)
        
        # Hierarchical tab
        hierarchical_frame = ttk.Frame(notebook, padding="10")
        notebook.add(hierarchical_frame, text="Hierarchical")
        self.setup_hierarchical_params(hierarchical_frame)
        
        # GMM tab
        gmm_frame = ttk.Frame(notebook, padding="10")
        notebook.add(gmm_frame, text="Gaussian Mixture")
        self.setup_gmm_params(gmm_frame)
        
        # General settings tab
        general_frame = ttk.Frame(notebook, padding="10")
        notebook.add(general_frame, text="General")
        self.setup_general_params(general_frame)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="▶ Run Clustering",
            command=self.run_clustering,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="📊 View Results",
            command=self.view_results,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="💾 Save Results",
            command=self.save_results,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Output/Log frame
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=10,
            wrap=tk.WORD,
            font=("Courier", 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def setup_kmeans_params(self, parent):
        """Setup K-Means parameters."""
        self.kmeans_n_clusters = tk.StringVar(value="5")
        self.kmeans_n_init = tk.StringVar(value="10")
        self.kmeans_max_iter = tk.StringVar(value="300")
        self.kmeans_algorithm = tk.StringVar(value="lloyd")
        
        ttk.Label(parent, text="Number of clusters:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.kmeans_n_clusters, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="(Leave empty for auto)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(parent, text="n_init (runs):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.kmeans_n_init, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(parent, text="max_iter:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.kmeans_max_iter, width=20).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(parent, text="Algorithm:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(parent, textvariable=self.kmeans_algorithm, values=['lloyd', 'elkan', 'auto'], width=17, state='readonly').grid(row=3, column=1, sticky=tk.W, pady=5)
    
    def setup_dbscan_params(self, parent):
        """Setup DBSCAN parameters."""
        self.dbscan_eps = tk.StringVar(value="0.8")
        self.dbscan_min_samples = tk.StringVar(value="5")
        self.dbscan_metric = tk.StringVar(value="euclidean")
        
        ttk.Label(parent, text="eps (max distance):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.dbscan_eps, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(parent, text="min_samples:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.dbscan_min_samples, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(parent, text="Metric:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(parent, textvariable=self.dbscan_metric, values=['euclidean', 'cosine', 'manhattan'], width=17, state='readonly').grid(row=2, column=1, sticky=tk.W, pady=5)
    
    def setup_hierarchical_params(self, parent):
        """Setup Hierarchical parameters."""
        self.hierarchical_n_clusters = tk.StringVar(value="5")
        self.hierarchical_linkage = tk.StringVar(value="ward")
        self.hierarchical_metric = tk.StringVar(value="euclidean")
        
        ttk.Label(parent, text="Number of clusters:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.hierarchical_n_clusters, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="(Leave empty for auto)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(parent, text="Linkage:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(parent, textvariable=self.hierarchical_linkage, values=['ward', 'complete', 'average', 'single'], width=17, state='readonly').grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(parent, text="Metric:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(parent, textvariable=self.hierarchical_metric, values=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], width=17, state='readonly').grid(row=2, column=1, sticky=tk.W, pady=5)
    
    def setup_gmm_params(self, parent):
        """Setup GMM parameters."""
        self.gmm_n_components = tk.StringVar(value="5")
        self.gmm_covariance_type = tk.StringVar(value="full")
        self.gmm_max_iter = tk.StringVar(value="100")
        
        ttk.Label(parent, text="Number of components:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.gmm_n_components, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="(Leave empty for auto)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(parent, text="Covariance type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(parent, textvariable=self.gmm_covariance_type, values=['full', 'tied', 'diag', 'spherical'], width=17, state='readonly').grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(parent, text="max_iter:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.gmm_max_iter, width=20).grid(row=2, column=1, sticky=tk.W, pady=5)
    
    def setup_general_params(self, parent):
        """Setup general parameters."""
        # Label detection
        ttk.Label(parent, text="Label Detection Method:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(5, 10))
        self.detection_method = tk.StringVar(value="keywords")
        ttk.Combobox(parent, textvariable=self.detection_method, values=['keywords', 'topic_modeling', 'representative_tweet'], width=20, state='readonly').grid(row=0, column=1, sticky=tk.W, pady=(5, 10))
        
        # Clustering settings
        ttk.Separator(parent, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(parent, text="Clustering Settings:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.auto_n_clusters = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Auto-determine number of clusters", variable=self.auto_n_clusters).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.min_clusters = tk.StringVar(value="2")
        ttk.Label(parent, text="Min clusters:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.min_clusters, width=20).grid(row=4, column=1, sticky=tk.W, pady=5)
        
        self.max_clusters = tk.StringVar(value="10")
        ttk.Label(parent, text="Max clusters:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.max_clusters, width=20).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Dimensionality reduction
        ttk.Separator(parent, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(parent, text="Dimensionality Reduction:", font=("Arial", 9, "bold")).grid(row=7, column=0, sticky=tk.W, pady=5)
        
        self.use_dim_reduction = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Use dimensionality reduction", variable=self.use_dim_reduction).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.dim_reduction_method = tk.StringVar(value="umap")
        ttk.Label(parent, text="Method:").grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(parent, textvariable=self.dim_reduction_method, values=['umap', 'tsne', 'pca', 'none'], width=17, state='readonly').grid(row=9, column=1, sticky=tk.W, pady=5)
        
        self.n_components = tk.StringVar(value="50")
        ttk.Label(parent, text="Components:").grid(row=10, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.n_components, width=20).grid(row=10, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="(Leave empty for auto)").grid(row=10, column=2, sticky=tk.W, padx=(5, 0))
        
        # Feature engineering
        ttk.Separator(parent, orient='horizontal').grid(row=11, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(parent, text="Feature Engineering:", font=("Arial", 9, "bold")).grid(row=12, column=0, sticky=tk.W, pady=5)
        
        self.use_combined_features = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Combine with hashtag/engagement features", variable=self.use_combined_features).grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.use_cosine_distance = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Use cosine distance (better for semantic similarity)", variable=self.use_cosine_distance).grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=5)
    
    def browse_file(self):
        """Browse for input file."""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def log(self, message):
        """Add message to output."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()
    
    def get_params(self):
        """Get all parameters from UI."""
        # K-Means
        kmeans_params = {
            'n_clusters': int(self.kmeans_n_clusters.get()) if self.kmeans_n_clusters.get() else None,
            'n_init': int(self.kmeans_n_init.get()),
            'max_iter': int(self.kmeans_max_iter.get()),
            'random_state': 42,
            'algorithm': self.kmeans_algorithm.get()
        }
        
        # DBSCAN
        dbscan_params = {
            'eps': float(self.dbscan_eps.get()),
            'min_samples': int(self.dbscan_min_samples.get()),
            'metric': self.dbscan_metric.get(),
            'algorithm': 'auto'
        }
        
        # Hierarchical
        hierarchical_params = {
            'n_clusters': int(self.hierarchical_n_clusters.get()) if self.hierarchical_n_clusters.get() else None,
            'linkage': self.hierarchical_linkage.get(),
            'metric': self.hierarchical_metric.get()
        }
        
        # GMM
        gmm_params = {
            'n_components': int(self.gmm_n_components.get()) if self.gmm_n_components.get() else None,
            'covariance_type': self.gmm_covariance_type.get(),
            'max_iter': int(self.gmm_max_iter.get()),
            'random_state': 42,
            'tol': 0.001
        }
        
        return kmeans_params, dbscan_params, hierarchical_params, gmm_params
    
    def run_clustering(self):
        """Run clustering in background thread."""
        if not self.file_path_var.get() or not os.path.exists(self.file_path_var.get()):
            messagebox.showerror("Error", "Please select a valid CSV file!")
            return
        
        # Disable button and start progress
        self.progress.start()
        self.output_text.delete(1.0, tk.END)
        
        # Run in background thread
        thread = threading.Thread(target=self._run_clustering_thread, daemon=True)
        thread.start()
    
    def _run_clustering_thread(self):
        """Run clustering (called in background thread)."""
        try:
            self.log("="*60)
            self.log("Starting Clustering Analysis")
            self.log("="*60)
            
            # Load data
            self.log("\nLoading data...")
            self.df = filter_riyadh_season_tweets(self.file_path_var.get())
            
            if self.df is None or len(self.df) == 0:
                self.log("Error: No data found!")
                return
            
            self.log(f"Loaded {len(self.df)} tweets")
            
            # Clean data
            self.df['tweet_rawContent'] = self.df['tweet_rawContent'].fillna('')
            self.df = self.df.dropna(subset=['tweet_rawContent', 'tweet_id'])
            
            # Add basic features for analysis
            if 'tweet_retweetCount' in self.df.columns:
                self.df['retweet_count'] = pd.to_numeric(self.df['tweet_retweetCount'], errors='coerce').fillna(0)
            else:
                self.df['retweet_count'] = 0
            
            if 'tweet_likeCount' in self.df.columns:
                self.df['like_count'] = pd.to_numeric(self.df['tweet_likeCount'], errors='coerce').fillna(0)
            else:
                self.df['like_count'] = 0
            
            if 'tweet_hashtags' in self.df.columns:
                from cluster_riyadh_season import safe_parse_list
                self.df['hashtag_count'] = self.df['tweet_hashtags'].apply(safe_parse_list).apply(len)
            else:
                self.df['hashtag_count'] = 0
            
            if 'tweet_links' in self.df.columns:
                self.df['has_link'] = self.df['tweet_links'].apply(safe_parse_list).apply(len).apply(lambda x: 1 if x > 0 else 0)
            else:
                self.df['has_link'] = 0
            
            # Create embeddings
            self.log("\nCreating text embeddings...")
            self.embeddings, self.model = create_text_embeddings(
                self.df, 
                use_combined_features=self.use_combined_features.get()
            )
            
            # Apply dimensionality reduction if requested
            if self.use_dim_reduction.get() and self.dim_reduction_method.get() != 'none':
                self.log(f"\nApplying {self.dim_reduction_method.get().upper()} dimensionality reduction...")
                
                method = self.dim_reduction_method.get().lower()
                n_comp = int(self.n_components.get()) if self.n_components.get() else None
                
                if method == 'umap':
                    try:
                        import umap
                        n_comp = n_comp if n_comp else min(50, self.embeddings.shape[1] // 2)
                        reducer = umap.UMAP(
                            n_components=n_comp,
                            random_state=42,
                            metric='cosine' if self.use_cosine_distance.get() else 'euclidean',
                            n_neighbors=15,
                            min_dist=0.1
                        )
                        X_scaled = reducer.fit_transform(self.embeddings)
                        self.log(f"Reduced to {X_scaled.shape[1]} dimensions using UMAP")
                    except ImportError:
                        self.log("⚠ UMAP not available, using PCA instead")
                        from sklearn.decomposition import PCA
                        n_comp = n_comp if n_comp else min(50, self.embeddings.shape[1] // 2)
                        reducer = PCA(n_components=n_comp, random_state=42)
                        X_scaled = reducer.fit_transform(self.embeddings)
                        self.log(f"Reduced to {X_scaled.shape[1]} dimensions using PCA")
                
                elif method == 'tsne':
                    from sklearn.manifold import TSNE
                    # t-SNE Barnes-Hut only supports n_components <= 3
                    n_comp = n_comp if n_comp else 2
                    if n_comp > 3:
                        self.log("⚠ Warning: t-SNE with n_components > 3 uses exact method (slower). Limiting to 3.")
                        n_comp = 3
                    
                    # Use exact method if n_components > 3, otherwise use barnes_hut (faster)
                    tsne_method = 'exact' if n_comp > 3 else 'barnes_hut'
                    reducer = TSNE(
                        n_components=n_comp,
                        random_state=42,
                        perplexity=min(30, len(self.df) // 4),
                        n_iter=1000,
                        method=tsne_method
                    )
                    X_scaled = reducer.fit_transform(self.embeddings)
                    self.log(f"Reduced to {X_scaled.shape[1]} dimensions using t-SNE ({tsne_method} method)")
                
                elif method == 'pca':
                    from sklearn.decomposition import PCA
                    n_comp = n_comp if n_comp else min(50, self.embeddings.shape[1] // 2)
                    reducer = PCA(n_components=n_comp, random_state=42)
                    X_scaled = reducer.fit_transform(self.embeddings)
                    explained_var = reducer.explained_variance_ratio_.sum()
                    self.log(f"Reduced to {X_scaled.shape[1]} dimensions using PCA (explained variance: {explained_var:.2%})")
                else:
                    X_scaled = self.embeddings
            else:
                # Use embeddings directly (normalize if using cosine distance)
                if self.use_cosine_distance.get():
                    from sklearn.preprocessing import normalize
                    X_scaled = normalize(self.embeddings, norm='l2')
                    self.log("Using cosine-normalized embeddings")
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(self.embeddings)
                    self.log("Using standard-scaled embeddings")
            
            # Get parameters
            kmeans_params, dbscan_params, hierarchical_params, gmm_params = self.get_params()
            
            # Update DBSCAN metric if using cosine distance
            if self.use_cosine_distance.get():
                dbscan_params['metric'] = 'cosine'
            
            # Determine n_clusters
            if self.auto_n_clusters.get():
                min_c = int(self.min_clusters.get())
                max_c = int(self.max_clusters.get())
                n_clusters = min(max_c, max(min_c, len(self.df) // 50))
            else:
                n_clusters = kmeans_params.get('n_clusters', 5)
            
            # Override if specified
            if kmeans_params.get('n_clusters'):
                n_clusters = kmeans_params['n_clusters']
            if hierarchical_params.get('n_clusters'):
                n_clusters_hierarchical = hierarchical_params['n_clusters']
            else:
                n_clusters_hierarchical = n_clusters
            if gmm_params.get('n_components'):
                n_components_gmm = gmm_params['n_components']
            else:
                n_components_gmm = n_clusters
            
            self.log(f"\nUsing {n_clusters} clusters")
            
            # Run clustering algorithms
            self.results = {}
            labels_dict = {}
            
            self.log("\nRunning K-Means...")
            labels_kmeans, metrics_kmeans = apply_kmeans(X_scaled, n_clusters=n_clusters, params=kmeans_params)
            labels_dict['K-Means'] = labels_kmeans
            self.results['K-Means'] = {'labels': labels_kmeans, 'metrics': metrics_kmeans}
            
            self.log("\nRunning DBSCAN...")
            labels_dbscan, metrics_dbscan = apply_dbscan(
                X_scaled,
                eps=dbscan_params['eps'],
                min_samples=dbscan_params['min_samples'],
                params=dbscan_params
            )
            labels_dict['DBSCAN'] = labels_dbscan
            self.results['DBSCAN'] = {'labels': labels_dbscan, 'metrics': metrics_dbscan}
            
            self.log("\nRunning Hierarchical...")
            labels_hierarchical, metrics_hierarchical = apply_hierarchical(
                X_scaled,
                n_clusters=n_clusters_hierarchical,
                params=hierarchical_params
            )
            labels_dict['Hierarchical'] = labels_hierarchical
            self.results['Hierarchical'] = {'labels': labels_hierarchical, 'metrics': metrics_hierarchical}
            
            self.log("\nRunning Gaussian Mixture...")
            labels_gmm, metrics_gmm = apply_gaussian_mixture(
                X_scaled,
                n_components=n_components_gmm,
                params=gmm_params
            )
            labels_dict['Gaussian Mixture'] = labels_gmm
            self.results['Gaussian Mixture'] = {'labels': labels_gmm, 'metrics': metrics_gmm}
            
            # Auto-detect labels
            self.log("\nAuto-detecting cluster labels...")
            cluster_categories_dict = {}
            for algo_name, labels in labels_dict.items():
                categories = auto_detect_cluster_labels(
                    self.df, labels, self.embeddings, self.model,
                    method=self.detection_method.get()
                )
                cluster_categories_dict[algo_name] = categories
            
            self.results['categories'] = cluster_categories_dict
            self.results['labels'] = labels_dict
            
            self.log("\n✓ Clustering complete!")
            self.log("\nSummary:")
            for algo_name, result in self.results.items():
                if algo_name not in ['categories', 'labels'] and 'metrics' in result:
                    metrics = result['metrics']
                    if 'silhouette' in metrics:
                        self.log(f"  {algo_name}: Silhouette = {metrics['silhouette']:.4f}")
            
        except Exception as e:
            self.log(f"\n✗ Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.progress.stop()
    
    def view_results(self):
        """View clustering results."""
        if not self.results:
            messagebox.showinfo("Info", "Please run clustering first!")
            return
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Clustering Results")
        results_window.geometry("800x600")
        
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for algo_name in ['K-Means', 'DBSCAN', 'Hierarchical', 'Gaussian Mixture']:
            if algo_name in self.results.get('labels', {}):
                frame = ttk.Frame(notebook, padding="10")
                notebook.add(frame, text=algo_name)
                
                text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True)
                
                labels = self.results['labels'][algo_name]
                categories = self.results['categories'][algo_name]
                
                text_widget.insert(tk.END, f"{algo_name} Clustering Results\n")
                text_widget.insert(tk.END, "="*60 + "\n\n")
                
                for cluster_id in sorted(np.unique(labels)):
                    if cluster_id == -1:
                        text_widget.insert(tk.END, f"Noise/Outliers: {list(labels).count(-1)} tweets\n")
                    else:
                        count = list(labels).count(cluster_id)
                        category = categories.get(cluster_id, 'Unknown')
                        text_widget.insert(tk.END, f"Cluster {cluster_id}: {category} ({count} tweets)\n")
                
                if algo_name in self.results and 'metrics' in self.results[algo_name]:
                    metrics = self.results[algo_name]['metrics']
                    text_widget.insert(tk.END, "\nMetrics:\n")
                    for metric_name, value in metrics.items():
                        text_widget.insert(tk.END, f"  {metric_name}: {value:.4f}\n")
    
    def save_results(self):
        """Save clustering results."""
        if not self.results:
            messagebox.showinfo("Info", "Please run clustering first!")
            return
        
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            for algo_name, labels in self.results.get('labels', {}).items():
                df_result = self.df.copy()
                df_result['cluster'] = labels
                df_result['cluster_category'] = df_result['cluster'].map(
                    self.results['categories'][algo_name]
                )
                
                output_file = os.path.join(output_dir, f'clusters_{algo_name.lower().replace(" ", "_")}.csv')
                df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
                self.log(f"Saved: {output_file}")
            
            messagebox.showinfo("Success", f"Results saved to {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

def main():
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()


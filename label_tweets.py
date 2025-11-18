import pandas as pd
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO
import threading
import webbrowser

# Configuration
INPUT_FILE = "combined_output/all_data_combined.csv"
OUTPUT_FILE = "combined_output/all_data_labeled.csv"
LABEL_COLUMN = "label"  # 0 = Genuine, 1 = Inauthentic

class TweetLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Twitter Tweet Labeling Tool")
        self.root.geometry("1000x700")
        
        # Load data
        self.df = None
        self.current_index = 0
        self.unlabeled_indices = []
        self.stats = {'labeled': 0, 'genuine': 0, 'inauthentic': 0}
        self.current_image = None
        self.current_image_label = None
        
        # Load data
        self.load_data()
        
        # Setup UI
        self.setup_ui()
        
        # Load first tweet
        if self.unlabeled_indices:
            self.load_tweet()
    
    def load_data(self):
        """Load data from CSV file."""
        if not os.path.exists(INPUT_FILE):
            messagebox.showerror("Error", f"Input file '{INPUT_FILE}' not found!")
            self.root.destroy()
            return
        
        try:
            self.df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
            
            # Initialize label column if it doesn't exist
            if LABEL_COLUMN not in self.df.columns:
                self.df[LABEL_COLUMN] = None
            
            # Get unlabeled tweets
            unlabeled_mask = self.df[LABEL_COLUMN].isna()
            self.unlabeled_indices = self.df[unlabeled_mask].index.tolist()
            
            # Calculate stats
            self.stats['labeled'] = len(self.df) - len(self.unlabeled_indices)
            self.stats['genuine'] = len(self.df[self.df[LABEL_COLUMN] == 0]) if LABEL_COLUMN in self.df.columns else 0
            self.stats['inauthentic'] = len(self.df[self.df[LABEL_COLUMN] == 1]) if LABEL_COLUMN in self.df.columns else 0
            
            if not self.unlabeled_indices:
                messagebox.showinfo("Info", "All tweets are already labeled!")
                self.root.destroy()
                return
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {e}")
            self.root.destroy()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Progress frame (top)
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(
            progress_frame, 
            text=f"Tweet 0 of {len(self.unlabeled_indices)}",
            font=("Arial", 12, "bold")
        )
        self.progress_label.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(
            progress_frame,
            text=f"Labeled: {self.stats['labeled']} | Genuine: {self.stats['genuine']} | Inauthentic: {self.stats['inauthentic']}",
            font=("Arial", 10)
        )
        self.stats_label.pack(side=tk.RIGHT)
        
        # Content frame (middle) - split into text and image
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        content_frame.columnconfigure(0, weight=2)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Tweet text frame (left)
        text_frame = ttk.LabelFrame(content_frame, text="Tweet Text", padding="10")
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.tweet_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            height=15,
            width=60
        )
        self.tweet_text.pack(fill=tk.BOTH, expand=True)
        
        # Metadata frame (below text)
        metadata_frame = ttk.Frame(text_frame)
        metadata_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.metadata_label = ttk.Label(
            metadata_frame,
            text="",
            font=("Arial", 9),
            foreground="gray"
        )
        self.metadata_label.pack(side=tk.LEFT)
        
        # Tweet link button
        self.tweet_link_button = ttk.Button(
            metadata_frame,
            text="🔗 Open Tweet",
            command=self.open_tweet_link,
            width=15
        )
        self.tweet_link_button.pack(side=tk.RIGHT)
        self.current_tweet_url = None
        
        # Image frame (right)
        image_frame = ttk.LabelFrame(content_frame, text="Media", padding="10")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create a frame for image with video indicator
        image_container = ttk.Frame(image_frame)
        image_container.pack(expand=True, fill=tk.BOTH)
        
        self.image_label = ttk.Label(
            image_container,
            text="No image",
            font=("Arial", 10),
            foreground="gray",
            anchor=tk.CENTER
        )
        self.image_label.pack(expand=True)
        
        self.video_indicator = ttk.Label(
            image_container,
            text="",
            font=("Arial", 9, "bold"),
            foreground="red"
        )
        self.video_indicator.pack()
        
        # Buttons frame (bottom)
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Label buttons
        ttk.Button(
            buttons_frame,
            text="✓ Genuine (0)",
            command=lambda: self.label_tweet(0),
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="✗ Inauthentic (1)",
            command=lambda: self.label_tweet(1),
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="⊘ Skip",
            command=self.skip_tweet,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="← Back",
            command=self.go_back,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="💾 Save & Quit",
            command=self.save_and_quit,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts
        self.root.bind('<Key-0>', lambda e: self.label_tweet(0))
        self.root.bind('<Key-1>', lambda e: self.label_tweet(1))
        self.root.bind('<Key-s>', lambda e: self.skip_tweet())
        self.root.bind('<Key-b>', lambda e: self.go_back())
        self.root.bind('<Key-q>', lambda e: self.save_and_quit())
        self.root.focus_set()
    
    def get_tweet_text(self, tweet_data):
        """Get tweet text from various possible columns."""
        for col in ['tweet_rawContent', 'tweet_text', 'text', 'content']:
            if col in tweet_data.index and pd.notna(tweet_data[col]):
                return str(tweet_data[col])
        return "No tweet text found"
    
    def get_tweet_media(self, tweet_data):
        """Extract media URL from tweet data. Returns (url, is_video) tuple."""
        # Try to get media from tweet_media column (JSON string)
        if 'tweet_media' in tweet_data.index and pd.notna(tweet_data['tweet_media']):
            try:
                media_str = str(tweet_data['tweet_media'])
                # Skip empty media
                if not media_str or media_str == '{}' or media_str == 'nan':
                    return None, False
                
                media_data = json.loads(media_str)
                
                # Check if it's a dict with media arrays
                if isinstance(media_data, dict):
                    # Try photos first
                    if 'tweet_media_photos' in media_data and isinstance(media_data['tweet_media_photos'], list):
                        for photo in media_data['tweet_media_photos']:
                            if isinstance(photo, dict):
                                # Try common URL fields
                                for url_field in ['url', 'media_url', 'mediaUrl', 'mediaUrlHttps', 'media_url_https', 'tweet_media_photos_url']:
                                    if url_field in photo:
                                        url = photo[url_field]
                                        if url and isinstance(url, str) and ('http' in url or 'pbs.twimg.com' in url or 'twimg.com' in url):
                                            return url, False
                    
                    # Try videos (get thumbnail)
                    if 'tweet_media_videos' in media_data and isinstance(media_data['tweet_media_videos'], list):
                        for video in media_data['tweet_media_videos']:
                            if isinstance(video, dict):
                                # Try thumbnail URL first (for videos)
                                if 'tweet_media_videos_thumbnailUrl' in video:
                                    url = video['tweet_media_videos_thumbnailUrl']
                                    if url and isinstance(url, str) and ('http' in url or 'pbs.twimg.com' in url or 'twimg.com' in url):
                                        return url, True
                                # Try other URL fields
                                for url_field in ['url', 'media_url', 'mediaUrl', 'mediaUrlHttps', 'preview_image_url', 'thumbnail_url']:
                                    if url_field in video:
                                        url = video[url_field]
                                        if url and isinstance(url, str) and ('http' in url or 'pbs.twimg.com' in url or 'twimg.com' in url):
                                            return url, True
                    
                    # Try animated (GIFs)
                    if 'tweet_media_animated' in media_data and isinstance(media_data['tweet_media_animated'], list):
                        for animated in media_data['tweet_media_animated']:
                            if isinstance(animated, dict):
                                for url_field in ['url', 'media_url', 'mediaUrl', 'mediaUrlHttps', 'preview_image_url']:
                                    if url_field in animated:
                                        url = animated[url_field]
                                        if url and isinstance(url, str) and ('http' in url or 'pbs.twimg.com' in url or 'twimg.com' in url):
                                            return url, False
                
                # If it's a list
                elif isinstance(media_data, list) and len(media_data) > 0:
                    first_media = media_data[0]
                    if isinstance(first_media, dict):
                        for url_field in ['url', 'media_url', 'mediaUrl', 'mediaUrlHttps', 'media_url_https']:
                            if url_field in first_media:
                                url = first_media[url_field]
                                if url and isinstance(url, str) and ('http' in url or 'pbs.twimg.com' in url or 'twimg.com' in url):
                                    return url, False
            except Exception as e:
                print(f"Error parsing media: {e}")
                pass
        
        return None, False
    
    def load_image(self, url, is_video=False):
        """Load and display image from URL."""
        if not url:
            self.image_label.config(text="No image", image='')
            return
        
        def load_image_thread():
            try:
                # Try to load image with better headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://twitter.com/'
                }
                
                response = requests.get(url, timeout=10, headers=headers, stream=True)
                if response.status_code == 200:
                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if 'image' not in content_type and 'video' not in content_type:
                        # Try to load anyway
                        pass
                    
                    img_data = response.content
                    if not img_data:
                        self.root.after(0, lambda: self.image_label.config(text="Empty image data", image=''))
                        return
                    
                    img = Image.open(BytesIO(img_data))
                    
                    # Convert RGBA to RGB if needed
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    max_size = (400, 400)
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(img)
                    
                    # Update UI in main thread with video indicator
                    video_text = "▶ VIDEO" if is_video else ""
                    self.root.after(0, lambda p=photo, v=video_text: self.display_image(p, v))
                else:
                    error_text = f"Failed to load (HTTP {response.status_code})"
                    if is_video:
                        error_text += " - Video thumbnail"
                    self.root.after(0, lambda msg=error_text: self.image_label.config(text=msg, image=''))
            except requests.exceptions.Timeout:
                error_text = "Timeout loading image"
                if is_video:
                    error_text += " (video thumbnail)"
                self.root.after(0, lambda msg=error_text: self.image_label.config(text=msg, image=''))
            except Exception as e:
                error_msg = str(e)[:40]
                error_text = f"Error: {error_msg}"
                if is_video:
                    error_text += " (video)"
                self.root.after(0, lambda msg=error_text: self.image_label.config(text=msg, image=''))
        
        # Load in background thread
        thread = threading.Thread(target=load_image_thread, daemon=True)
        thread.start()
    
    def display_image(self, photo, video_text=""):
        """Display image in the label."""
        self.current_image = photo  # Keep reference
        self.image_label.config(image=photo, text="")
        if video_text:
            self.video_indicator.config(text=video_text)
        else:
            self.video_indicator.config(text="")
    
    def load_tweet(self):
        """Load current tweet into UI."""
        if self.current_index >= len(self.unlabeled_indices):
            messagebox.showinfo("Complete", "All tweets labeled!")
            self.save_and_quit()
            return
        
        idx = self.unlabeled_indices[self.current_index]
        tweet_data = self.df.loc[idx]
        
        # Update progress
        self.progress_label.config(
            text=f"Tweet {self.current_index + 1} of {len(self.unlabeled_indices)}"
        )
        
        # Load tweet text
        tweet_text = self.get_tweet_text(tweet_data)
        self.tweet_text.delete(1.0, tk.END)
        self.tweet_text.insert(1.0, tweet_text)
        
        # Load metadata
        metadata_parts = []
        if 'tweet_date' in tweet_data.index and pd.notna(tweet_data['tweet_date']):
            metadata_parts.append(f"📅 {tweet_data['tweet_date']}")
        if 'user_username' in tweet_data.index and pd.notna(tweet_data['user_username']):
            metadata_parts.append(f"👤 @{tweet_data['user_username']}")
        if 'tweet_likeCount' in tweet_data.index and pd.notna(tweet_data['tweet_likeCount']):
            metadata_parts.append(f"❤️ {int(tweet_data['tweet_likeCount'])}")
        if 'tweet_retweetCount' in tweet_data.index and pd.notna(tweet_data['tweet_retweetCount']):
            metadata_parts.append(f"🔄 {int(tweet_data['tweet_retweetCount'])}")
        
        self.metadata_label.config(text=" | ".join(metadata_parts))
        
        # Get and store tweet URL
        self.current_tweet_url = self.get_tweet_url(tweet_data)
        if self.current_tweet_url:
            # Update button to show URL is available
            self.tweet_link_button.config(state=tk.NORMAL)
            # Show shortened URL in button
            short_url = self.current_tweet_url.replace('https://', '').replace('http://', '')
            if len(short_url) > 35:
                short_url = short_url[:32] + '...'
            self.tweet_link_button.config(text=f"🔗 {short_url}")
        else:
            self.tweet_link_button.config(state=tk.DISABLED, text="🔗 No URL")
        
        # Load image
        media_url, is_video = self.get_tweet_media(tweet_data)
        if media_url:
            status_text = "Loading video thumbnail..." if is_video else "Loading image..."
            self.image_label.config(text=status_text, image='')
            self.video_indicator.config(text="")
            self.load_image(media_url, is_video=is_video)
        else:
            self.image_label.config(text="No media available", image='')
            self.video_indicator.config(text="")
            self.current_image = None
    
    def get_tweet_url(self, tweet_data):
        """Get tweet URL from tweet data."""
        # Try tweet_url column first
        if 'tweet_url' in tweet_data.index and pd.notna(tweet_data['tweet_url']):
            url = str(tweet_data['tweet_url'])
            if url and url != 'nan' and ('http' in url or 'twitter.com' in url or 'x.com' in url):
                return url
        
        # Try to construct URL from tweet_id and username
        tweet_id = None
        username = None
        
        if 'tweet_id' in tweet_data.index and pd.notna(tweet_data['tweet_id']):
            tweet_id = str(tweet_data['tweet_id'])
        elif 'tweet_id_str' in tweet_data.index and pd.notna(tweet_data['tweet_id_str']):
            tweet_id = str(tweet_data['tweet_id_str'])
        
        if 'user_username' in tweet_data.index and pd.notna(tweet_data['user_username']):
            username = str(tweet_data['user_username']).replace('@', '')
        
        if tweet_id and username:
            return f"https://x.com/{username}/status/{tweet_id}"
        elif tweet_id:
            return f"https://x.com/i/web/status/{tweet_id}"
        
        return None
    
    def open_tweet_link(self):
        """Open tweet URL in browser."""
        if self.current_tweet_url:
            webbrowser.open(self.current_tweet_url)
        else:
            messagebox.showwarning("No URL", "Tweet URL not available")
    
    def label_tweet(self, label):
        """Label current tweet."""
        if self.current_index >= len(self.unlabeled_indices):
            return
        
        idx = self.unlabeled_indices[self.current_index]
        self.df.at[idx, LABEL_COLUMN] = label
        
        self.stats['labeled'] += 1
        if label == 0:
            self.stats['genuine'] += 1
        else:
            self.stats['inauthentic'] += 1
        
        self.update_stats()
        self.current_index += 1
        
        # Auto-save every 10 tweets
        if self.stats['labeled'] % 10 == 0:
            self.save_progress()
        
        self.load_tweet()
    
    def skip_tweet(self):
        """Skip current tweet."""
        self.current_index += 1
        self.load_tweet()
    
    def go_back(self):
        """Go back to previous tweet."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_tweet()
    
    def update_stats(self):
        """Update statistics display."""
        self.stats_label.config(
            text=f"Labeled: {self.stats['labeled']} | Genuine: {self.stats['genuine']} | Inauthentic: {self.stats['inauthentic']}"
        )
    
    def save_progress(self):
        """Save progress to file."""
        try:
            self.df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        except Exception as e:
            messagebox.showerror("Error", f"Error saving: {e}")
    
    def save_and_quit(self):
        """Save and quit application."""
        self.save_progress()
        messagebox.showinfo(
            "Saved",
            f"Progress saved!\n\n"
            f"Labeled: {self.stats['labeled']}\n"
            f"Genuine: {self.stats['genuine']}\n"
            f"Inauthentic: {self.stats['inauthentic']}\n"
            f"Remaining: {len(self.unlabeled_indices) - self.current_index}"
        )
        self.root.destroy()

def main():
    """Main function."""
    root = tk.Tk()
    app = TweetLabelingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

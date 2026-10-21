
# Arabic Trend Jacking Detection
### Semantic Relevance vs. User Metadata: Detecting Trend Jacking on Arabic X (Twitter)

An Arabic Natural Language Processing (NLP) and Machine Learning project that investigates how to detect **trend jacking** — the practice of using trending hashtags in posts that are unrelated to the actual topic.

The project compares traditional machine learning algorithms, an Arabic transformer model, and large language models (LLMs) to explore the trade-off between **classification accuracy, semantic understanding, and reliance on user metadata**.

**Project Type:** Academic Team Project  
**Domain:** Artificial Intelligence | Machine Learning | Arabic NLP

---

## 1. Problem Statement

Trending hashtags on X (formerly Twitter) are widely used to follow news, sports, and events. However, some accounts include popular hashtags in unrelated posts to increase visibility.

For example, a commercial advertisement may use a trending sports hashtag even though the advertisement has no connection to the match.

This creates two major problems:

- **User Experience:** Trending discussions become cluttered with unrelated content.
- **Data Quality:** Social media analysis becomes less reliable when unrelated posts are included.

The goal of this project is to distinguish between:

- **Genuine (0):** Posts that are relevant to the hashtag's topic.
- **Trend Jacking (1):** Posts that use the hashtag but discuss unrelated content.

---

## 2. Project Objectives

The project investigates three research questions:

1. How accurately can traditional machine learning algorithms detect trend jacking in Arabic tweets?
2. How important are user metadata, tweet structure, and semantic similarity?
3. Can text-only models achieve competitive performance without requiring additional user metadata?

---

## 3. Dataset

We collected approximately **9,000 Arabic tweets** from trending hashtags, including:

- #موسم_الرياض
- #دوري_روشن_السعودي
- #النصر_نيوم
- #معرض_بيبان
- #يوم_الجمعة

### Data Processing

The dataset preparation process included:

1. Collecting tweets using Python and `twscrape`.
2. Combining files from different dates and hashtags.
3. Removing duplicate records.
4. Handling missing data.
5. Preparing a manually labeled dataset.

After combining and cleaning, approximately **4,856 tweets** remained.

### Manual Labeling

A custom desktop labeling tool was developed using `tkinter`.

Tweets were classified into two categories:

| Label | Category | Description |
|---|---|---|
| 0 | Genuine | Relevant to the hashtag topic |
| 1 | Trend Jacking | Unrelated to the hashtag topic |

The final labeled dataset contained approximately **491 tweets**.

---

## 4. Methodology

The project evaluated two main approaches.

### Approach A: Feature-Based Machine Learning

This approach uses engineered features from tweet structure, user metadata, and semantic similarity.

**User Metadata**
- Account age
- Followers and following counts
- Total number of tweets
- Follower-to-following ratio

**Tweet Structure**
- Tweet length
- Number of hashtags
- Number of mentions
- Presence of links
- Presence of emojis
- Posting time

**Semantic Similarity**

A multilingual sentence embedding model was used to represent tweet text and hashtag topic descriptions as vectors.

Cosine similarity was then calculated to estimate how closely each tweet relates to its hashtag topic.

The project experimented with nine machine learning algorithms, including:

- XGBoost
- Random Forest
- LightGBM
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Multi-Layer Perceptron (MLP)

### Approach B: Text-Only NLP Models

This approach evaluates whether tweets can be classified using text alone, without relying on account metadata.

Three models were evaluated:

**AraBERTv2**
- Fine-tuned for binary classification
- Trained on the labeled Arabic dataset
- Five training epochs
- Learning rate: `2e-5`

**Qwen 2.5 72B Instruct**
- Evaluated using prompt-based classification
- No task-specific fine-tuning

**Llama 3.3 70B Instruct**
- Evaluated using prompt-based classification
- No task-specific fine-tuning

The large language model experiments were conducted using an NVIDIA B200 GPU.

---

## 5. Results

The project reported the following classification results:

| Model | Input Type | Accuracy | ROC AUC |
|---|---|---:|---:|
| **XGBoost** | Engineered features + metadata | **87.8%** | 0.8936 |
| Random Forest | Engineered features + metadata | 86.5% | 0.9037 |
| AraBERTv2 | Text only | 82.0% | 0.8779 |
| Qwen 2.5 72B | Text only | 79.6% | — |
| Llama 3.3 70B | Text only | 75.2% | — |

*Note: These are the results reported in the academic project. Different model families used different evaluation approaches, so the figures should be interpreted with that limitation in mind.*

### Key Findings

**1. Feature-based models achieved the highest reported accuracy.**

XGBoost achieved 87.8% accuracy using tweet characteristics, user metadata, and semantic similarity.

**2. Arabic-specific NLP models performed competitively.**

AraBERTv2 achieved 82% accuracy using tweet text alone.

**3. Larger models did not necessarily perform better.**

Despite their larger parameter counts, Qwen and Llama achieved lower reported accuracy than fine-tuned AraBERT.

**4. Semantic similarity was an important feature.**

The analysis highlighted the usefulness of measuring whether a tweet is semantically related to the hashtag topic.

**5. Text-only classification offers a practical alternative.**

Models that use tweet text alone reduce reliance on additional account metadata, although their reported accuracy was lower in this project.

---

## 6. Technologies and Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Collection | twscrape |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Arabic NLP | AraBERTv2 |
| Large Language Models | Qwen 2.5 72B, Llama 3.3 70B |
| Semantic Analysis | Sentence Embeddings, Cosine Similarity |
| Data Labeling | tkinter, Pillow |
| GPU Computing | NVIDIA B200, RunPod |

---

## 7. Project Workflow

The overall project workflow consisted of:

1. **Data Collection** — Collect Arabic tweets from trending hashtags.
2. **Data Cleaning** — Merge datasets, remove duplicates, and prepare records.
3. **Manual Labeling** — Classify tweets as genuine or trend jacking.
4. **Feature Engineering** — Extract metadata, structural features, and semantic similarity.
5. **Model Training** — Train and evaluate traditional machine learning models.
6. **NLP Experiments** — Fine-tune AraBERT and evaluate LLMs.
7. **Results Analysis** — Compare model performance and discuss trade-offs.

---

## 8. Limitations and Future Work

### Limitations

- The manually labeled dataset contained only 491 tweets.
- Model performance may vary with different data splits.
- The dataset focused on selected Arabic trending hashtags.
- Results from different model families were not all produced under identical evaluation procedures.

### Future Improvements

- Expand the labeled dataset to more than 2,000 tweets.
- Evaluate additional Arabic-focused language models.
- Explore LLM-assisted labeling.
- Incorporate image and multimedia features.
- Develop a simple web application for classifying user-submitted tweets.

---

## 9. Team Members

This project was developed collaboratively as an academic team project.

- Meshal AlRashed
- Mohammad Al-Khunaizan
- Abdulaziz Abunayyan

---

## 10. Conclusion

This project demonstrates how machine learning and Arabic NLP can be applied to detect unrelated content within trending social media discussions.

The results highlight the importance of combining semantic relevance, tweet structure, and user metadata while also demonstrating the potential of text-only Arabic language models.

The project provides a foundation for further research into Arabic social media content classification and automated trend-jacking detection.


import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import joblib
import time
from collections import Counter


NLTK_DATA_DIR = os.path.join(os.path.expanduser("~"), "nltk_data_private")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

import nltk
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

def _safe_nltk_download(pkg):
    try:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
    except Exception as e:
        print(f"NLTK download failed for {pkg}: {e}")

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    _safe_nltk_download(pkg)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# Plotting
import plotly.graph_objects as go
import plotly.express as px



def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b\w+\b", text)

def safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sents if s]

def safe_stopwords():
    try:
        return set(stopwords.words("english"))
    except Exception:
        return set("""a an the and or but if while of at by for with about against between
        into through during before after above below to from up down in out on off over under
        again further then once here there when where why how all any both each few more most
        other some such no nor not only own same so than too very s t can will just don should
        now is are was were be been being have has had do does did i you he she it we they me
        him her us them my your his its our their this that these those""".split())


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🔍 Fake News Detector for Students",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ENHANCED CUSTOM CSS - Better visibility & animations
# =============================================================================
st.markdown("""
<style>
    /* Global text improvements */
    html, body, [class*="css"] {
        font-size: 16px;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        font-size: 1.15rem !important;
        opacity: 0.95;
    }
    
    /* Verdict banners - large & bold */
    .verdict-real, .verdict-fake, .verdict-uncertain {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        animation: slideIn 0.6s ease-out;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    .verdict-real {
        background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
        border: 3px solid #28a745;
        color: #155724;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #f8d7da 0%, #f5b7b1 100%);
        border: 3px solid #dc3545;
        color: #721c24;
    }
    .verdict-uncertain {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe082 100%);
        border: 3px solid #ffc107;
        color: #856404;
    }
    .verdict-real h1, .verdict-fake h1, .verdict-uncertain h1 {
        font-size: 2.8rem !important;
        margin: 0.5rem 0 !important;
        font-weight: 800 !important;
    }
    .verdict-real p, .verdict-fake p, .verdict-uncertain p {
        font-size: 1.2rem !important;
        margin: 0.4rem 0 !important;
        font-weight: 600;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Info cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        transition: transform 0.2s;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.12);
    }
    .info-card h3 {
        color: #333 !important;
        font-size: 1.3rem !important;
        margin-bottom: 0.7rem;
    }
    .info-card p {
        color: #444 !important;
        font-size: 1.05rem !important;
        line-height: 1.6;
    }
    
    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f0ff 100%);
        border: 2px solid #4a90e2;
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.15);
    }
    .summary-box p {
        color: #1a3a52 !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        margin: 0;
        font-weight: 500;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    div[data-testid="stMetric"] label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #555 !important;
    }
    div[data-testid="stMetric"] div {
        font-size: 1.8rem !important;
        color: #222 !important;
    }
    
    /* Tip / action boxes */
    .action-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .action-real {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 6px solid #17a2b8;
        color: #0c5460;
    }
    .action-fake {
        background: linear-gradient(135deg, #f8d7da 0%, #f5b7b1 100%);
        border-left: 6px solid #dc3545;
        color: #721c24;
    }
    .action-warn {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe082 100%);
        border-left: 6px solid #ffc107;
        color: #856404;
    }
    .action-box h3 {
        margin-top: 0 !important;
        font-size: 1.4rem !important;
    }
    .action-box li {
        margin: 0.4rem 0;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem !important;
        color: #333 !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        border-radius: 10px !important;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.5);
    }
    
    /* Feature badge */
    .feature-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-good { background: #d4edda; color: #155724; }
    .badge-bad { background: #f8d7da; color: #721c24; }
    .badge-neutral { background: #e2e3e5; color: #383d41; }
    
    /* DataFrame */
    div[data-testid="stDataFrame"] {
        font-size: 1rem !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# TEXT PREPROCESSOR
# =============================================================================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = safe_stopwords()
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.lemmatizer.lemmatize("tests")
        except Exception:
            self.lemmatizer = None
        self.sensational_words = {
            'shocking', 'unbelievable', 'breaking', 'urgent', 'exclusive',
            'bombshell', 'stunning', 'incredible', 'horrifying', 'terrifying',
            'amazing', 'miracle', 'secret', 'conspiracy', 'exposed', 'revealed',
            'banned', 'censored', "they dont want you to know", 'wake up',
            'share before deleted', 'must see', 'jaw dropping', 'mind blowing',
            "you wont believe", 'outrageous', 'insane', 'epic', 'destroyed',
            'obliterated', 'annihilated', 'slammed', 'blasted', 'torched'
        }

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _lemmatize(self, token):
        if self.lemmatizer is not None:
            try:
                return self.lemmatizer.lemmatize(token)
            except Exception:
                return token
        return token

    def preprocess_for_tfidf(self, text):
        text = self.clean_text(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = safe_word_tokenize(text)
        tokens = [self._lemmatize(t) for t in tokens
                  if t not in self.stop_words and len(t) > 2 and not t.isdigit()]
        return ' '.join(tokens)

    def extract_linguistic_features(self, text):
        text = self.clean_text(text)
        keys = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'vocabulary_richness', 'uppercase_ratio',
            'exclamation_count', 'question_count', 'ellipsis_count',
            'capital_word_count', 'sensational_word_count', 'sentiment_polarity',
            'sentiment_subjectivity', 'digit_ratio', 'punctuation_ratio',
            'quote_count', 'paragraph_count', 'long_word_ratio',
            'stopword_ratio', 'type_token_ratio'
        ]
        if len(text) == 0:
            return {k: 0 for k in keys}

        words = text.split()
        sentences = safe_sent_tokenize(text)
        features = {}

        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = max(len(sentences), 1)
        features['avg_word_length'] = float(np.mean([len(w) for w in words])) if words else 0.0
        features['avg_sentence_length'] = len(words) / max(len(sentences), 1)
        features['vocabulary_richness'] = len(set(w.lower() for w in words)) / len(words) if words else 0
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        features['capital_word_count'] = sum(1 for w in words if w.isupper() and len(w) > 1)

        text_lower = text.lower()
        features['sensational_word_count'] = sum(1 for sw in self.sensational_words if sw in text_lower)

        try:
            blob = TextBlob(text[:5000])
            features['sentiment_polarity'] = float(blob.sentiment.polarity)
            features['sentiment_subjectivity'] = float(blob.sentiment.subjectivity)
        except Exception:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0

        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text)
        features['quote_count'] = text.count('"') + text.count("'")
        features['paragraph_count'] = text.count('\n') + 1
        features['long_word_ratio'] = sum(1 for w in words if len(w) > 6) / len(words) if words else 0

        stop_count = sum(1 for w in words if w.lower() in self.stop_words)
        features['stopword_ratio'] = stop_count / len(words) if words else 0
        features['type_token_ratio'] = len(set(words)) / len(words) if words else 0

        return features


# =============================================================================
# CREDIBILITY SCORER
# =============================================================================
class CredibilityScorer:
    def __init__(self):
        self.credible_indicators = [
            'according to', 'research shows', 'study finds', 'data suggests',
            'officials say', 'experts say', 'reported by', 'sources confirm',
            'evidence suggests', 'analysis shows', 'statistics indicate',
            'peer reviewed', 'published in', 'university of', 'institute of'
        ]
        self.non_credible_indicators = [
            'you wont believe', "they dont want you to know", 'shocking truth',
            'share before', 'must see', 'breaking exclusive', 'conspiracy',
            'cover up', 'mainstream media lies', 'wake up sheeple',
            'big pharma', 'deep state', 'false flag', 'hoax',
            'click here', 'limited time', 'act now'
        ]

    def score_article(self, text):
        if not text or len(str(text).strip()) == 0:
            return 0, {}

        text = str(text)
        text_lower = text.lower()
        score = 50
        breakdown = {}

        source_score = min(sum(2.5 for ind in self.credible_indicators if ind in text_lower), 15)
        score += source_score
        breakdown['source_citations'] = source_score

        sensational_penalty = max(sum(-3 for ind in self.non_credible_indicators if ind in text_lower), -20)
        score += sensational_penalty
        breakdown['sensationalism_penalty'] = sensational_penalty

        words = text.split()
        if words:
            avg_word_len = np.mean([len(w) for w in words])
            quality_score = 10 if 4 <= avg_word_len <= 7 else (5 if 3 <= avg_word_len <= 8 else 0)
        else:
            quality_score = 0
        score += quality_score
        breakdown['writing_quality'] = quality_score

        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        caps_penalty = -10 if caps_ratio > 0.3 else (-5 if caps_ratio > 0.15 else 0)
        score += caps_penalty
        breakdown['caps_penalty'] = caps_penalty

        excl_count = text.count('!')
        punct_penalty = -10 if excl_count > 5 else (-5 if excl_count > 2 else 0)
        score += punct_penalty
        breakdown['punctuation_penalty'] = punct_penalty

        if len(words) > 300:
            length_bonus = 10
        elif len(words) > 150:
            length_bonus = 5
        elif len(words) > 50:
            length_bonus = 2
        else:
            length_bonus = -5
        score += length_bonus
        breakdown['length_bonus'] = length_bonus

        try:
            blob = TextBlob(text[:3000])
            subjectivity = blob.sentiment.subjectivity
        except Exception:
            subjectivity = 0.5
        obj_bonus = 10 if subjectivity < 0.3 else (5 if subjectivity < 0.5 else -5)
        score += obj_bonus
        breakdown['objectivity_bonus'] = obj_bonus
        breakdown['subjectivity_value'] = round(subjectivity, 3)

        if len(words) > 10:
            diversity = len(set(w.lower() for w in words)) / len(words)
            div_bonus = 5 if diversity > 0.6 else (2 if diversity > 0.4 else -2)
        else:
            div_bonus = 0
        score += div_bonus
        breakdown['vocabulary_diversity'] = div_bonus

        score = max(0, min(100, score))
        return score, breakdown


# =============================================================================
# SUMMARY (extractive)
# =============================================================================
def extractive_summary(text, num_sentences=3):
    if not text or len(str(text).strip()) == 0:
        return "No text provided for summarization."
    text = str(text)
    sentences = safe_sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    stop_words_set = safe_stopwords()
    words = safe_word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in stop_words_set]
    word_freq = Counter(words)
    if not word_freq:
        return ' '.join(sentences[:num_sentences])
    max_freq = max(word_freq.values())
    word_freq = {w: f / max_freq for w, f in word_freq.items()}

    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = safe_word_tokenize(sent.lower())
        score = sum(word_freq.get(w, 0) for w in sent_words if w.isalnum())
        score = score / (len(sent_words) + 1)
        if i < 3:
            score *= 1.2
        sentence_scores[i] = score

    top_indices = sorted(sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences])
    return ' '.join([sentences[i] for i in top_indices])


# =============================================================================
# BUILT-IN TRAINING DATA
# =============================================================================
BUILTIN_TRAINING_DATA = [
    ("According to a study published in the New England Journal of Medicine, researchers at Stanford University found that regular moderate exercise can reduce cardiovascular disease risk by 35 percent. The peer-reviewed study analyzed data from over 100,000 participants.", 1),
    ("Health officials from the World Health Organization confirmed today that vaccination rates have increased globally. The report, published in a peer-reviewed journal, indicates significant progress in immunization programs.", 1),
    ("The Federal Reserve announced a 0.25 percent interest rate adjustment following its monthly meeting. According to economists, this change reflects current inflation trends.", 1),
    ("Scientists at MIT have developed a new battery technology that could improve electric vehicle range. The research, published in Nature, involved three years of testing.", 1),
    ("The Department of Education released new statistics showing improvements in national literacy rates. According to officials, the data reflects investments made over the past decade.", 1),
    ("NASA confirmed the successful launch of a new satellite designed to monitor climate change. The mission will provide critical atmospheric data.", 1),
    ("A comprehensive review published in the Lancet examined the effectiveness of new cancer treatments. The analysis included data from 50 clinical trials.", 1),
    ("The Bureau of Labor Statistics reported that unemployment rates decreased by 0.3 percent last quarter. Economists analyzing the data suggest the trend indicates economic recovery.", 1),
    ("Local authorities announced new infrastructure improvements following city council approval. According to the mayor's office, construction will begin next month.", 1),
    ("Researchers at Johns Hopkins University published findings on antibiotic resistance in a leading medical journal. The study, spanning ten years, examined thousands of bacterial samples.", 1),
    ("SHOCKING!!! You WON'T BELIEVE what they've been hiding!!! Secret miracle cure EXPOSED!!! Big Pharma doesn't want you to know!!! SHARE before DELETED!!! Wake up sheeple!!!", 0),
    ("BREAKING!!! Government conspiracy REVEALED!!! Deep state operatives caught in massive cover-up!!! Mainstream media LIES about everything!!! Click here NOW!!!", 0),
    ("UNBELIEVABLE miracle discovery! One weird trick doctors HATE! This amazing secret will change your life FOREVER! They banned this because it works too well!", 0),
    ("TERRIFYING TRUTH EXPOSED! Officials CENSORED this bombshell report! You must see this before it's DELETED! Share with everyone you know!", 0),
    ("INSANE new evidence proves everything is a HOAX! Wake up! The false flag operation has been REVEALED! Mainstream media won't report this!", 0),
    ("MIND BLOWING secret finally revealed! Big pharma doesn't want you to see this! One shocking trick that will destroy the entire industry!", 0),
    ("OUTRAGEOUS scandal ROCKS the nation!!! Politicians CAUGHT red-handed in massive cover-up!!! You won't believe what they did next!!!", 0),
    ("EPIC discovery obliterates everything scientists thought they knew!!! Establishment DESTROYED by new evidence!!! They don't want you to see this!!!", 0),
    ("URGENT WARNING! Secret document LEAKED reveals horrifying conspiracy! Government trying to hide the truth! Share this before they delete it!", 0),
    ("STUNNING revelation exposes decades of LIES! The mainstream media won't tell you this bombshell truth! Amazing new evidence proves everything!", 0),
]


# =============================================================================
# LOAD OR TRAIN MODEL
# =============================================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.joblib")
VECT_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")


def _train_from_dataframe(df, preprocessor):
    df = df.dropna(subset=['full_text', 'label']).reset_index(drop=True)
    df['cleaned_text'] = df['full_text'].apply(preprocessor.preprocess_for_tfidf)
    ling_list = df['full_text'].apply(preprocessor.extract_linguistic_features).tolist()
    ling_df = pd.DataFrame(ling_list)

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=1, max_df=0.95,
        sublinear_tf=True, dtype=np.float32
    )
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    scaler = StandardScaler()
    X_ling = scaler.fit_transform(ling_df.values)
    X = hstack([X_tfidf, csr_matrix(X_ling)])
    y = df['label'].astype(int).values

    model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, vectorizer, scaler


def _try_load_kaggle_dataset():
    try:
        import kagglehub
        path = kagglehub.dataset_download("mucahiddemircan/real-and-fake-news-dataset")

        true_path, fake_path = None, None
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                lower = file.lower()
                if 'true' in lower or 'real' in lower:
                    true_path = filepath
                elif 'fake' in lower:
                    fake_path = filepath

        if true_path and fake_path:
            df_true = pd.read_csv(true_path)
            df_fake = pd.read_csv(fake_path)
            df_true['label'] = 1
            df_fake['label'] = 0
            df = pd.concat([df_true, df_fake], ignore_index=True)
        else:
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            if not csv_files:
                return None
            df = pd.read_csv(csv_files[0])
            if 'label' not in df.columns:
                return None

        if 'text' in df.columns and 'title' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['full_text'] = df['text'].fillna('')
        elif 'title' in df.columns:
            df['full_text'] = df['title'].fillna('')
        else:
            return None

        if len(df) > 8000:
            df = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), 4000), random_state=42)
            ).reset_index(drop=True)

        return df[['full_text', 'label']]
    except Exception as e:
        print(f"Kaggle dataset load failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    preprocessor = TextPreprocessor()
    scorer = CredibilityScorer()

    if all(os.path.exists(p) for p in [MODEL_PATH, VECT_PATH, SCALER_PATH]):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECT_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, vectorizer, scaler, preprocessor, scorer, True, "loaded"
        except Exception as e:
            print(f"Failed to load saved model: {e}")

    df = _try_load_kaggle_dataset()
    if df is not None and len(df) > 20:
        try:
            model, vectorizer, scaler = _train_from_dataframe(df, preprocessor)
            return model, vectorizer, scaler, preprocessor, scorer, True, "kaggle"
        except Exception as e:
            print(f"Kaggle training failed: {e}")

    try:
        df = pd.DataFrame(BUILTIN_TRAINING_DATA, columns=['full_text', 'label'])
        model, vectorizer, scaler = _train_from_dataframe(df, preprocessor)
        return model, vectorizer, scaler, preprocessor, scorer, True, "builtin"
    except Exception as e:
        print(f"Builtin training failed: {e}")
        return None, None, None, preprocessor, scorer, False, "failed"


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_article(text, title, model, vectorizer, scaler, preprocessor, scorer):
    full_text = (title + ' ' + text) if title else text

    cleaned = preprocessor.preprocess_for_tfidf(full_text)
    tfidf_features = vectorizer.transform([cleaned])
    linguistic_features = preprocessor.extract_linguistic_features(full_text)
    ling_array = np.array([list(linguistic_features.values())])
    ling_scaled = scaler.transform(ling_array)
    combined = hstack([tfidf_features, csr_matrix(ling_scaled)])

    ml_prediction = int(model.predict(combined)[0])
    ml_probability = model.predict_proba(combined)[0]

    credibility_score, score_breakdown = scorer.score_article(full_text)
    ml_real_prob = float(ml_probability[1])
    combined_score = (ml_real_prob * 70) + (credibility_score / 100 * 30)

    if combined_score >= 65:
        verdict = "LIKELY REAL"
        confidence_level = "High" if combined_score >= 80 else "Moderate"
    elif combined_score >= 40:
        verdict = "UNCERTAIN"
        confidence_level = "Low"
    else:
        verdict = "LIKELY FAKE"
        confidence_level = "High" if combined_score <= 20 else "Moderate"

    summary = extractive_summary(full_text)

    return {
        'verdict': verdict,
        'confidence_level': confidence_level,
        'combined_score': round(combined_score, 1),
        'ml_prediction': 'Real' if ml_prediction == 1 else 'Fake',
        'ml_confidence': round(max(ml_probability) * 100, 1),
        'ml_real_probability': round(ml_real_prob * 100, 1),
        'ml_fake_probability': round(float(ml_probability[0]) * 100, 1),
        'credibility_score': credibility_score,
        'score_breakdown': score_breakdown,
        'linguistic_features': linguistic_features,
        'summary': summary,
        'word_count': int(linguistic_features.get('word_count', 0)),
        'sentiment_polarity': round(linguistic_features.get('sentiment_polarity', 0), 3),
        'sentiment_subjectivity': round(linguistic_features.get('sentiment_subjectivity', 0), 3)
    }


# =============================================================================
# INTERACTIVE CHART HELPERS
# =============================================================================
def create_gauge_chart(score, title="Credibility Score"):
    """Animated credibility gauge."""
    if score >= 65:
        bar_color = "#28a745"
    elif score >= 40:
        bar_color = "#ffc107"
    else:
        bar_color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b>", 'font': {'size': 22, 'color': '#333'}},
        delta={'reference': 50, 'increasing': {'color': "#28a745"}, 'decreasing': {'color': "#dc3545"}},
        number={'font': {'size': 48, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#333", 'tickfont': {'size': 14}},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 65], 'color': '#fff3cd'},
                {'range': [65, 100], 'color': '#d4edda'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333"}
    )
    return fig


def create_probability_donut(real_prob, fake_prob):
    """Donut chart of ML probabilities."""
    fig = go.Figure(data=[go.Pie(
        labels=['✅ Real', '❌ Fake'],
        values=[real_prob, fake_prob],
        hole=0.6,
        marker=dict(colors=['#28a745', '#dc3545'], line=dict(color='white', width=3)),
        textinfo='label+percent',
        textfont=dict(size=16, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    )])
    fig.update_layout(
        title={'text': '<b>🤖 ML Prediction Breakdown</b>', 'font': {'size': 18, 'color': '#333'}, 'x': 0.5},
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f'<b>{max(real_prob, fake_prob):.0f}%</b>', x=0.5, y=0.5,
                          font_size=28, showarrow=False, font_color='#333')]
    )
    return fig


def create_breakdown_bar(breakdown):
    """Horizontal bar chart of score breakdown."""
    items = [(k, v) for k, v in breakdown.items() if k != 'subjectivity_value']
    items.sort(key=lambda x: x[1])
    labels = [k.replace('_', ' ').title() for k, _ in items]
    values = [v for _, v in items]
    colors = ['#dc3545' if v < 0 else ('#28a745' if v > 0 else '#6c757d') for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1.5)),
        text=[f"{v:+.1f}" for v in values],
        textposition='outside',
        textfont=dict(size=13, color='#333'),
        hovertemplate='<b>%{y}</b><br>Points: %{x:+.1f}<extra></extra>'
    ))
    fig.update_layout(
        title={'text': '<b>📈 Credibility Score Breakdown</b>', 'font': {'size': 18, 'color': '#333'}},
        height=380,
        margin=dict(l=20, r=60, t=60, b=40),
        xaxis=dict(title="Points", zeroline=True, zerolinecolor='#333', zerolinewidth=2,
                   gridcolor='#eee', tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_linguistic_radar(features):
    """Radar chart of key linguistic features (normalized)."""
    categories = ['Vocabulary\nRichness', 'Objectivity', 'Writing\nQuality',
                  'Low Caps\nUsage', 'Low Sensationalism', 'Length\nAdequacy']

    vocab = min(features.get('vocabulary_richness', 0) * 100, 100)
    objectivity = (1 - features.get('sentiment_subjectivity', 0.5)) * 100
    awl = features.get('avg_word_length', 0)
    writing = 100 if 4 <= awl <= 7 else (60 if 3 <= awl <= 8 else 30)
    caps = max(0, 100 - features.get('uppercase_ratio', 0) * 300)
    sens = max(0, 100 - features.get('sensational_word_count', 0) * 20)
    wc = features.get('word_count', 0)
    length = min(100, wc / 3) if wc < 300 else 100

    values = [vocab, objectivity, writing, caps, sens, length]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed,
        fill='toself', fillcolor='rgba(102, 126, 234, 0.35)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2'),
        name='Article Score',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=11), gridcolor='#ddd'),
            angularaxis=dict(tickfont=dict(size=12, color='#333'))
        ),
        title={'text': '<b>🎯 Article Quality Profile</b>', 'font': {'size': 18, 'color': '#333'}, 'x': 0.5},
        showlegend=False,
        height=400,
        margin=dict(l=60, r=60, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake News Detector for Students</h1>
        <p>🎓 AI-powered tool to analyze articles, assess credibility & learn media literacy</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("📰 Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Home - Article Analyzer", "📚 How It Works", "💡 Media Literacy Tips"],
            index=0
        )
        st.markdown("---")
        st.markdown("### ⚙️ Display Options")
        show_detailed = st.checkbox("🔬 Show detailed analysis", value=True)
        show_breakdown = st.checkbox("📈 Show score breakdown", value=True)
        show_radar = st.checkbox("🎯 Show quality radar", value=True)
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("**Algorithm:** Logistic Regression\n\n**Features:** TF-IDF + 21 linguistic features")

    with st.spinner("🚀 Loading AI model... (first run may take a moment)"):
        model, vectorizer, scaler_obj, preprocessor, scorer, model_loaded, mode = load_or_train_model()

    if model_loaded:
        if mode == "builtin":
            st.warning("⚠️ Running with a small built-in demo model. For higher accuracy, provide a trained model in `models/`.")
        elif mode == "kaggle":
            st.success("✅ Model trained on Kaggle real/fake news dataset.")
    else:
        st.error("❌ Model failed to load. Please check your dependencies.")

    if page == "🏠 Home - Article Analyzer":
        render_home(model, vectorizer, scaler_obj, preprocessor, scorer,
                    model_loaded, show_detailed, show_breakdown, show_radar)
    elif page == "📚 How It Works":
        render_how_it_works()
    elif page == "💡 Media Literacy Tips":
        render_tips()


def render_home(model, vectorizer, scaler_obj, preprocessor, scorer,
                model_loaded, show_detailed, show_breakdown, show_radar):
    st.markdown('<h2 class="section-header">📝 Paste an Article to Analyze</h2>', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method:",
        ["✍️ Paste Article Text", "📋 Try Example Articles"],
        horizontal=True
    )

    if input_method == "📋 Try Example Articles":
        example_choice = st.selectbox(
            "Select an example:",
            ["🟢 Real News Example", "🔴 Fake News Example", "🟡 Clickbait Example"]
        )

        examples = {
            "🟢 Real News Example": {
                "title": "New Study Reveals Benefits of Regular Exercise",
                "text": """According to a comprehensive study published in the New England Journal of Medicine, 
researchers at Stanford University have found that regular moderate exercise can reduce the risk 
of cardiovascular disease by up to 35%. The peer-reviewed study, which analyzed data from over 
100,000 participants across 15 years, suggests that as little as 30 minutes of daily walking 
can provide significant health benefits. Dr. Sarah Johnson, the lead researcher, stated that 
'the evidence overwhelmingly supports the integration of regular physical activity into daily 
routines.' The findings were corroborated by independent analysis from the World Health Organization."""
            },
            "🔴 Fake News Example": {
                "title": "SHOCKING!!! Government HIDING Miracle Cure!!!",
                "text": """You WON'T BELIEVE what they've been keeping from us!!! A SECRET cure for ALL diseases 
has been discovered but Big Pharma doesn't want you to know!!! EXPOSED: The deep state conspiracy 
to keep us sick and dependent on their POISON medications!!! SHARE THIS BEFORE THEY DELETE IT!!! 
Wake up sheeple!!! The mainstream media LIES about everything!!! Click here to learn the secret!!!"""
            },
            "🟡 Clickbait Example": {
                "title": "This One Weird Trick Will Change Your Life Forever!",
                "text": """Scientists are baffled by this incredible discovery that could change everything we know 
about health. A mysterious fruit found in a remote island has properties that experts say could 
revolutionize medicine. While some researchers have expressed interest, no clinical trials have been 
conducted yet. The story has been shared millions of times on social media, with many claiming 
miraculous results. However, no peer-reviewed studies have confirmed these claims."""
            }
        }

        selected = examples[example_choice]
        article_title = st.text_input("📰 Article Title:", value=selected["title"])
        article_text = st.text_area("📄 Article Text:", value=selected["text"], height=250)
    else:
        article_title = st.text_input("📰 Article Title (optional):", placeholder="Enter the article headline...")
        article_text = st.text_area(
            "📄 Article Text:", height=250,
            placeholder="Paste the full article text here for analysis..."
        )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

    if analyze_button and article_text.strip():
        if not model_loaded:
            st.error("⚠️ Model not available.")
            return

        with st.spinner("🔄 Analyzing article..."):
            try:
                result = analyze_article(
                    article_text, article_title, model, vectorizer, scaler_obj,
                    preprocessor, scorer
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        st.balloons()
        st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)

        # === VERDICT BANNER ===
        verdict = result['verdict']
        if "REAL" in verdict:
            verdict_class, verdict_emoji, verdict_desc = "verdict-real", "✅", "This article shows credible characteristics"
        elif "FAKE" in verdict:
            verdict_class, verdict_emoji, verdict_desc = "verdict-fake", "❌", "This article shows warning signs"
        else:
            verdict_class, verdict_emoji, verdict_desc = "verdict-uncertain", "⚠️", "The credibility is unclear"

        st.markdown(f"""
        <div class="{verdict_class}">
            <div style="font-size: 4rem;">{verdict_emoji}</div>
            <h1>{verdict}</h1>
            <p style="font-size: 1.3rem;">{verdict_desc}</p>
            <p>🎯 <strong>Confidence:</strong> {result['confidence_level']} &nbsp;|&nbsp; 
               🏆 <strong>Overall Score:</strong> {result['combined_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)

        # === KEY METRICS ROW ===
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 ML Prediction", result['ml_prediction'], f"{result['ml_confidence']}% confident")
        with col2:
            st.metric("📊 Credibility", f"{result['credibility_score']}/100",
                      f"{'✔ Good' if result['credibility_score'] >= 60 else '⚠ Concerning'}")
        with col3:
            st.metric("📝 Word Count", result['word_count'],
                      f"{'Substantial' if result['word_count'] > 200 else 'Short'}")
        with col4:
            subj = result['sentiment_subjectivity']
            st.metric("🎭 Subjectivity", f"{subj:.2f}",
                      f"{'Objective' if subj < 0.4 else 'Subjective'}")

        # === INTERACTIVE CHARTS ROW ===
        st.markdown('<h3 class="section-header">📈 Visual Analysis</h3>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(create_gauge_chart(result['combined_score']), use_container_width=True)
        with chart_col2:
            st.plotly_chart(
                create_probability_donut(result['ml_real_probability'], result['ml_fake_probability']),
                use_container_width=True
            )

        # === SUMMARY ===
        st.markdown('<h3 class="section-header">📋 Article Summary</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-box">
            <p>💬 {result['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        # === DETAILED ANALYSIS ===
        if show_detailed:
            st.markdown('<h3 class="section-header">🔬 Detailed Analysis</h3>', unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["🤖 ML Assessment", "🎭 Sentiment & Style", "📊 Statistics"])

            with tab1:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown("#### 🎯 Probability Distribution")
                    real_pct = result['ml_real_probability']
                    fake_pct = result['ml_fake_probability']
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>✅ Probability of being <span style="color:#28a745">Real</span>: {real_pct}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(real_pct / 100)
                    st.markdown(f"""
                    <div class="info-card" style="border-left-color: #dc3545;">
                        <h3>❌ Probability of being <span style="color:#dc3545">Fake</span>: {fake_pct}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(fake_pct / 100)
                with col_b:
                    conf_color = "#28a745" if result['ml_confidence'] >= 75 else ("#ffc107" if result['ml_confidence'] >= 55 else "#dc3545")
                    st.markdown(f"""
                    <div class="info-card" style="text-align:center; border-left-color:{conf_color};">
                        <h3>🎯 Model Confidence</h3>
                        <p style="font-size:3rem !important; color:{conf_color}; font-weight:bold; margin:0;">{result['ml_confidence']}%</p>
                    </div>
                    """, unsafe_allow_html=True)

            with tab2:
                pol = result['sentiment_polarity']
                subj = result['sentiment_subjectivity']
                pol_label = "😊 Positive" if pol > 0.1 else ("😞 Negative" if pol < -0.1 else "😐 Neutral")
                subj_label = "📊 Objective (fact-based)" if subj < 0.4 else "💭 Subjective (opinion-based)"
                pol_color = "#28a745" if pol > 0.1 else ("#dc3545" if pol < -0.1 else "#6c757d")
                subj_color = "#28a745" if subj < 0.4 else "#dc3545"

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"""
                    <div class="info-card" style="border-left-color:{pol_color};">
                        <h3>Sentiment Polarity</h3>
                        <p style="font-size:2rem !important; color:{pol_color}; font-weight:bold;">{pol}</p>
                        <p>{pol_label}</p>
                        <small>Range: -1 (very negative) to +1 (very positive)</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col_s2:
                    st.markdown(f"""
                    <div class="info-card" style="border-left-color:{subj_color};">
                        <h3>Subjectivity</h3>
                        <p style="font-size:2rem !important; color:{subj_color}; font-weight:bold;">{subj}</p>
                        <p>{subj_label}</p>
                        <small>Range: 0 (very objective) to 1 (very subjective)</small>
                    </div>
                    """, unsafe_allow_html=True)

                ling = result['linguistic_features']
                st.markdown("#### 🚩 Warning Signals")
                excl = ling.get('exclamation_count', 0)
                sens = ling.get('sensational_word_count', 0)
                caps = ling.get('capital_word_count', 0)

                b1, b2, b3 = st.columns(3)
                with b1:
                    badge = "badge-bad" if excl > 5 else ("badge-neutral" if excl > 2 else "badge-good")
                    st.markdown(f'<div class="feature-badge {badge}">❗ Exclamations: {excl}</div>', unsafe_allow_html=True)
                with b2:
                    badge = "badge-bad" if sens > 3 else ("badge-neutral" if sens > 0 else "badge-good")
                    st.markdown(f'<div class="feature-badge {badge}">🔥 Sensational Words: {sens}</div>', unsafe_allow_html=True)
                with b3:
                    badge = "badge-bad" if caps > 5 else ("badge-neutral" if caps > 2 else "badge-good")
                    st.markdown(f'<div class="feature-badge {badge}">🔠 ALL-CAPS Words: {caps}</div>', unsafe_allow_html=True)

            with tab3:
                ling = result['linguistic_features']
                stats_data = {
                    "📏 Character Count": ling.get('char_count', 0),
                    "📝 Word Count": ling.get('word_count', 0),
                    "📄 Sentence Count": ling.get('sentence_count', 0),
                    "📐 Avg. Word Length": f"{ling.get('avg_word_length', 0):.2f}",
                    "📊 Avg. Sentence Length": f"{ling.get('avg_sentence_length', 0):.2f}",
                    "🎨 Vocabulary Richness": f"{ling.get('vocabulary_richness', 0):.3f}",
                    "❓ Question Marks": ling.get('question_count', 0),
                    "💬 Quote Count": ling.get('quote_count', 0),
                }
                stats_df = pd.DataFrame(list(stats_data.items()), columns=["Metric", "Value"])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # === RADAR CHART ===
        if show_radar:
            st.markdown('<h3 class="section-header">🎯 Article Quality Profile</h3>', unsafe_allow_html=True)
            st.plotly_chart(create_linguistic_radar(result['linguistic_features']), use_container_width=True)
            st.info("💡 A larger, more balanced shape indicates a higher-quality article. Small or lopsided shapes suggest quality concerns.")

        # === SCORE BREAKDOWN ===
        if show_breakdown:
            st.markdown('<h3 class="section-header">📈 What Influenced the Score?</h3>', unsafe_allow_html=True)
            st.plotly_chart(create_breakdown_bar(result['score_breakdown']), use_container_width=True)

            with st.expander("🔍 See detailed score breakdown table"):
                breakdown = result['score_breakdown']
                rows = []
                for factor, points in breakdown.items():
                    if factor == 'subjectivity_value':
                        continue
                    impact = "🟢 Positive" if points > 0 else ("🔴 Negative" if points < 0 else "⚪ Neutral")
                    rows.append({
                        'Factor': factor.replace('_', ' ').title(),
                        'Points': f"{points:+.1f}",
                        'Impact': impact
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # === ACTION RECOMMENDATIONS ===
        st.markdown('<h3 class="section-header">💡 What Should You Do?</h3>', unsafe_allow_html=True)
        if "REAL" in verdict:
            st.markdown("""
            <div class="action-box action-real">
                <h3>✅ This article appears credible!</h3>
                <p>But always practice good media literacy:</p>
                <ul>
                    <li>🔗 <b>Cross-reference</b> with other reputable news sources</li>
                    <li>📅 Check the <b>publication date</b> and author credentials</li>
                    <li>🔍 Look for <b>original source links</b> in the article</li>
                    <li>🎓 Verify claims with <b>academic or official sources</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif "FAKE" in verdict:
            st.markdown("""
            <div class="action-box action-fake">
                <h3>❌ This article shows signs of being unreliable!</h3>
                <p>Be very cautious — take these actions:</p>
                <ul>
                    <li>🚫 <b>Do NOT share</b> this article without verification</li>
                    <li>🔎 Check fact-checking sites: <b>Snopes, FactCheck.org, PolitiFact</b></li>
                    <li>📰 Look for the same story from <b>established news organizations</b></li>
                    <li>🎭 Be wary of <b>emotional manipulation</b> and sensational language</li>
                    <li>📢 <b>Report</b> if found on social media</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="action-box action-warn">
                <h3>⚠️ The credibility of this article is uncertain.</h3>
                <p>Proceed carefully:</p>
                <ul>
                    <li>🔎 <b>Seek additional sources</b> before forming an opinion</li>
                    <li>💬 Look for <b>direct quotes</b> and data sources in the article</li>
                    <li>👤 Check if the <b>author and publication</b> are established</li>
                    <li>🛠️ Use <b>fact-checking tools</b> for specific claims</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    elif analyze_button and not article_text.strip():
        st.warning("⚠️ Please enter some article text to analyze.")


def render_how_it_works():
    st.markdown('<h2 class="section-header">📚 How the Fake News Detector Works</h2>', unsafe_allow_html=True)
    st.markdown("""
    Our AI-powered fake news detector uses a **multi-layered approach** combining 
    machine learning and heuristic analysis to assess the credibility of news articles.
    """)
    st.markdown('<h3 class="section-header">🔄 Analysis Pipeline</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>1️⃣ Text Analysis</h3>
            <p>
            • <b>TF-IDF Vectorization</b><br>
            • <b>N-gram Analysis</b><br>
            • <b>21 Linguistic Features</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>2️⃣ ML Classification</h3>
            <p>
            • <b>Logistic Regression</b><br>
            • <b>Combined Features</b><br>
            • <b>Probability Estimation</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>3️⃣ Credibility Score</h3>
            <p>
            • <b>Heuristic Scoring</b><br>
            • <b>Combined Score</b> (ML 70% + Heuristic 30%)<br>
            • <b>Extractive Summary</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">🧠 Linguistic Features Analyzed</h3>', unsafe_allow_html=True)
    features_explained = {
        "Sentiment Polarity": "Measures if the text is positive, negative, or neutral",
        "Subjectivity Score": "Determines if the text is objective or subjective",
        "Sensational Word Count": "Counts clickbait/emotional trigger words",
        "Vocabulary Richness": "Measures diversity of word usage",
        "Uppercase Ratio": "Excessive capitals often indicate unreliable content",
        "Exclamation Marks": "Overuse of ! is a common fake news indicator",
        "Source Citations": "Presence of references to studies, experts, or data",
    }
    for feature, description in features_explained.items():
        st.markdown(f"- **{feature}**: {description}")


def render_tips():
    st.markdown('<h2 class="section-header">💡 Media Literacy Tips for Students</h2>', unsafe_allow_html=True)
    tips = [
        ("🔍 Check the Source", "Is the website reputable? Check the URL, About page, and design quality."),
        ("👤 Verify the Author", "Is the author a real, credentialed person? Search for their other work."),
        ("📅 Check the Date", "Is the article current, or is old news being reshared out of context?"),
        ("🔗 Cross-Reference", "Search the story on multiple trusted sources. Use Snopes, FactCheck.org, PolitiFact."),
        ("🎭 Watch Emotional Manipulation", "Beware ALL-CAPS, excessive !!!, and clickbait headlines."),
        ("🖼️ Verify Images", "Use Google Reverse Image Search. Watch for deepfakes."),
        ("📊 Look for Evidence", "Does the article cite specific studies, data, or named experts?"),
        ("🧠 Check Your Own Biases", "Be extra critical of articles that align perfectly with your views."),
    ]
    for title, content in tips:
        with st.expander(title, expanded=False):
            st.markdown(f"<p style='font-size:1.1rem;'>{content}</p>", unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">📱 Useful Fact-Checking Resources</h3>', unsafe_allow_html=True)
    resources = {
        "Google Fact Check Explorer": "https://toolbox.google.com/factcheck/explorer",
        "Snopes": "https://www.snopes.com",
        "FactCheck.org": "https://www.factcheck.org",
        "PolitiFact": "https://www.politifact.com",
        "News Literacy Project": "https://newslit.org",
    }
    for name, url in resources.items():
        st.markdown(f"- [{name}]({url})")


if __name__ == "__main__":
    main()

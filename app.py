# =============================================================================
# app.py - Fake News Detector for Students (Streamlit Application)
# =============================================================================
# Run with: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import joblib
import time
from pathlib import Path
from collections import Counter

# =============================================================================
# NLTK SETUP - Fix permission warnings by using a private directory
# =============================================================================
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


# =============================================================================
# Safe tokenizers (fallback if NLTK data still missing)
# =============================================================================
def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b\w+\b", text)

def safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except Exception:
        # Simple regex-based sentence splitter
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sents if s]

def safe_stopwords():
    try:
        return set(stopwords.words("english"))
    except Exception:
        # Minimal fallback stopword list
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
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .verdict-real {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .verdict-fake {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .verdict-uncertain {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #f0f7ff;
        border: 1px solid #b8daff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
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
            # test
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

        unique_words = set(w.lower() for w in words)
        features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
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
# SUMMARIZATION (extractive only - lightweight, no transformers)
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


def generate_summary(text):
    return extractive_summary(text)


# =============================================================================
# BUILT-IN TRAINING DATA (small labeled set for bootstrapping)
# Used only when no saved model & no Kaggle dataset available
# =============================================================================
BUILTIN_TRAINING_DATA = [
    # (text, label) where label 1=real, 0=fake
    ("According to a study published in the New England Journal of Medicine, researchers at Stanford University found that regular moderate exercise can reduce cardiovascular disease risk by 35 percent. The peer-reviewed study analyzed data from over 100,000 participants.", 1),
    ("Health officials from the World Health Organization confirmed today that vaccination rates have increased globally. The report, published in a peer-reviewed journal, indicates significant progress in immunization programs.", 1),
    ("The Federal Reserve announced a 0.25 percent interest rate adjustment following its monthly meeting. According to economists, this change reflects current inflation trends. The decision was reported by major financial news outlets.", 1),
    ("Scientists at MIT have developed a new battery technology that could improve electric vehicle range. The research, published in Nature, involved three years of testing. Data suggests the technology could be commercialized within five years.", 1),
    ("The Department of Education released new statistics showing improvements in national literacy rates. According to officials, the data reflects investments made over the past decade. Independent analysis confirms the findings.", 1),
    ("NASA confirmed the successful launch of a new satellite designed to monitor climate change. The mission, developed in collaboration with international partners, will provide critical atmospheric data. Officials say the satellite is fully operational.", 1),
    ("A comprehensive review published in the Lancet examined the effectiveness of new cancer treatments. The analysis included data from 50 clinical trials across multiple countries. Researchers concluded that early detection remains crucial.", 1),
    ("The Bureau of Labor Statistics reported that unemployment rates decreased by 0.3 percent last quarter. Economists analyzing the data suggest the trend indicates economic recovery. The report was corroborated by independent research institutions.", 1),
    ("Local authorities announced new infrastructure improvements following city council approval. According to the mayor's office, construction will begin next month. The project has been reviewed by independent engineering firms.", 1),
    ("Researchers at Johns Hopkins University published findings on antibiotic resistance in a leading medical journal. The study, spanning ten years, examined thousands of bacterial samples. Health experts recommend continued surveillance.", 1),
    # Fake examples
    ("SHOCKING!!! You WON'T BELIEVE what they've been hiding!!! Secret miracle cure EXPOSED!!! Big Pharma doesn't want you to know!!! SHARE before DELETED!!! Wake up sheeple!!!", 0),
    ("BREAKING!!! Government conspiracy REVEALED!!! Deep state operatives caught in massive cover-up!!! Mainstream media LIES about everything!!! Click here NOW!!! Limited time offer!!!", 0),
    ("UNBELIEVABLE miracle discovery! One weird trick doctors HATE! This amazing secret will change your life FOREVER! They banned this because it works too well! Act now!!!", 0),
    ("TERRIFYING TRUTH EXPOSED! Officials CENSORED this bombshell report! You must see this before it's DELETED! Share with everyone you know! The shocking conspiracy continues!!!", 0),
    ("INSANE new evidence proves everything is a HOAX! Wake up! The false flag operation has been REVEALED! Mainstream media won't report this! Share before censored!!!", 0),
    ("MIND BLOWING secret finally revealed! Big pharma doesn't want you to see this! One shocking trick that will destroy the entire industry! Click NOW before it's banned!!!", 0),
    ("OUTRAGEOUS scandal ROCKS the nation!!! Politicians CAUGHT red-handed in massive cover-up!!! You won't believe what they did next!!! MUST SEE footage!!! Share immediately!!!", 0),
    ("EPIC discovery obliterates everything scientists thought they knew!!! Establishment DESTROYED by new evidence!!! They don't want you to see this SHOCKING truth!!!", 0),
    ("URGENT WARNING! Secret document LEAKED reveals horrifying conspiracy! Government trying to hide the truth! Share this before they delete it! Wake up before it's too late!!!", 0),
    ("STUNNING revelation exposes decades of LIES! The mainstream media won't tell you this bombshell truth! Amazing new evidence proves everything! Must share!!!", 0),
]


# =============================================================================
# LOAD OR TRAIN MODEL
# =============================================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.joblib")
VECT_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")


def _train_from_dataframe(df, preprocessor):
    """Train model from a dataframe with 'full_text' and 'label' columns."""
    df = df.dropna(subset=['full_text', 'label']).reset_index(drop=True)
    df['cleaned_text'] = df['full_text'].apply(preprocessor.preprocess_for_tfidf)
    ling_list = df['full_text'].apply(preprocessor.extract_linguistic_features).tolist()
    ling_df = pd.DataFrame(ling_list)

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32
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
    """Attempt to download the Kaggle dataset. Returns DataFrame or None."""
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

        # Sample down to keep training fast on Streamlit Cloud
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
    """Load saved model or train a new one. Returns (model, vect, scaler, preprocessor, scorer, ok, mode)."""
    preprocessor = TextPreprocessor()
    scorer = CredibilityScorer()

    # 1. Try loading saved models
    if all(os.path.exists(p) for p in [MODEL_PATH, VECT_PATH, SCALER_PATH]):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECT_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, vectorizer, scaler, preprocessor, scorer, True, "loaded"
        except Exception as e:
            print(f"Failed to load saved model: {e}")

    # 2. Try Kaggle dataset
    df = _try_load_kaggle_dataset()
    if df is not None and len(df) > 20:
        try:
            model, vectorizer, scaler = _train_from_dataframe(df, preprocessor)
            return model, vectorizer, scaler, preprocessor, scorer, True, "kaggle"
        except Exception as e:
            print(f"Kaggle training failed: {e}")

    # 3. Fallback: train on built-in tiny dataset
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

    summary = generate_summary(full_text)

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
        'word_count': linguistic_features.get('word_count', 0),
        'sentiment_polarity': round(linguistic_features.get('sentiment_polarity', 0), 3),
        'sentiment_subjectivity': round(linguistic_features.get('sentiment_subjectivity', 0), 3)
    }


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake News Detector for Students</h1>
        <p>AI-powered tool to analyze articles, assess credibility, and generate trustworthy summaries</p>
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
        st.markdown("### ⚙️ Settings")
        show_detailed = st.checkbox("Show detailed analysis", value=True)
        show_breakdown = st.checkbox("Show score breakdown", value=True)
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("Logistic Regression + TF-IDF + Linguistic Features")

    with st.spinner("Loading AI model... (first run may take a moment)"):
        model, vectorizer, scaler_obj, preprocessor, scorer, model_loaded, mode = load_or_train_model()

    if model_loaded:
        if mode == "builtin":
            st.warning("⚠️ Running with a small built-in demo model. For best accuracy, provide a trained model in the `models/` folder.")
        elif mode == "kaggle":
            st.success("✅ Model trained on Kaggle dataset.")
    else:
        st.error("❌ Model failed to load. Check dependencies.")

    if page == "🏠 Home - Article Analyzer":
        render_home(model, vectorizer, scaler_obj, preprocessor, scorer,
                    model_loaded, show_detailed, show_breakdown)
    elif page == "📚 How It Works":
        render_how_it_works()
    elif page == "💡 Media Literacy Tips":
        render_tips()


def render_home(model, vectorizer, scaler_obj, preprocessor, scorer,
                model_loaded, show_detailed, show_breakdown):
    st.markdown("## 📝 Paste an Article to Analyze")

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
        article_title = st.text_input("Article Title:", value=selected["title"])
        article_text = st.text_area("Article Text:", value=selected["text"], height=250)
    else:
        article_title = st.text_input("Article Title (optional):", placeholder="Enter the article headline...")
        article_text = st.text_area(
            "Article Text:",
            height=250,
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

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        verdict = result['verdict']
        if "REAL" in verdict:
            verdict_class, verdict_emoji = "verdict-real", "✅"
        elif "FAKE" in verdict:
            verdict_class, verdict_emoji = "verdict-fake", "❌"
        else:
            verdict_class, verdict_emoji = "verdict-uncertain", "⚠️"

        st.markdown(f"""
        <div class="{verdict_class}">
            <h2>{verdict_emoji} Verdict: {verdict}</h2>
            <p><strong>Confidence Level:</strong> {result['confidence_level']}</p>
            <p><strong>Overall Credibility Score:</strong> {result['combined_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 ML Prediction", result['ml_prediction'], f"{result['ml_confidence']}% confident")
        with col2:
            st.metric("📊 Credibility Score", f"{result['credibility_score']}/100",
                      f"{'Good' if result['credibility_score'] >= 60 else 'Concerning'}")
        with col3:
            st.metric("📝 Word Count", int(result['word_count']),
                      f"{'Substantial' if result['word_count'] > 200 else 'Short'}")
        with col4:
            subj = result['sentiment_subjectivity']
            st.metric("🎭 Subjectivity", f"{subj:.2f}",
                      f"{'Objective' if subj < 0.4 else 'Subjective'}")

        st.markdown("### 📋 Article Summary")
        st.markdown(f"""
        <div class="summary-box">
            <p>{result['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        if show_detailed:
            st.markdown("### 🔬 Detailed Analysis")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### ML Model Assessment")
                st.write(f"**Probability of being Real:** {result['ml_real_probability']}%")
                st.write(f"**Probability of being Fake:** {result['ml_fake_probability']}%")
                st.progress(result['ml_real_probability'] / 100)
                st.caption(f"Real ← {result['ml_real_probability']}% | {result['ml_fake_probability']}% → Fake")
            with col_d2:
                st.markdown("#### Sentiment Analysis")
                pol = result['sentiment_polarity']
                st.write(f"**Polarity:** {pol} ({'Positive' if pol > 0 else 'Negative' if pol < 0 else 'Neutral'})")
                st.write(f"**Subjectivity:** {result['sentiment_subjectivity']} "
                         f"({'Objective' if result['sentiment_subjectivity'] < 0.4 else 'Subjective'})")
                ling = result['linguistic_features']
                st.write(f"**Exclamation Marks:** {ling.get('exclamation_count', 0)}")
                st.write(f"**Sensational Words:** {ling.get('sensational_word_count', 0)}")
                st.write(f"**Vocabulary Richness:** {ling.get('vocabulary_richness', 0):.3f}")

        if show_breakdown:
            st.markdown("### 📈 Credibility Score Breakdown")
            breakdown = result['score_breakdown']
            rows = []
            for factor, points in breakdown.items():
                if factor == 'subjectivity_value':
                    continue
                impact = "🟢 Positive" if points > 0 else ("🔴 Negative" if points < 0 else "⚪ Neutral")
                rows.append({
                    'Factor': factor.replace('_', ' ').title(),
                    'Points': points,
                    'Impact': impact
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### 💡 What Should You Do?")
        if "REAL" in verdict:
            st.success("""
            ✅ This article appears credible, but always practice good media literacy:
            - Cross-reference with other reputable news sources
            - Check the publication date and author credentials
            - Look for original source links in the article
            """)
        elif "FAKE" in verdict:
            st.error("""
            ❌ This article shows signs of being unreliable. Be cautious:
            - Do NOT share this article without verification
            - Check fact-checking sites (Snopes, FactCheck.org, PolitiFact)
            - Look for the same story from established news organizations
            """)
        else:
            st.warning("""
            ⚠️ The credibility of this article is uncertain:
            - Seek additional sources before forming an opinion
            - Look for direct quotes and data sources
            - Use fact-checking tools for specific claims
            """)

    elif analyze_button and not article_text.strip():
        st.warning("⚠️ Please enter some article text to analyze.")


def render_how_it_works():
    st.markdown("## 📚 How the Fake News Detector Works")
    st.markdown("""
    Our AI-powered fake news detector uses a **multi-layered approach** combining 
    machine learning and heuristic analysis to assess the credibility of news articles.
    """)
    st.markdown("### 🔄 Analysis Pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 1️⃣ Text Analysis
        - **TF-IDF Vectorization**
        - **N-gram Analysis**
        - **21 Linguistic Features**
        """)
    with col2:
        st.markdown("""
        #### 2️⃣ ML Classification
        - **Logistic Regression** trained on news articles
        - **Feature Combination** of TF-IDF + linguistic features
        - **Probability Estimation**
        """)
    with col3:
        st.markdown("""
        #### 3️⃣ Credibility Assessment
        - **Heuristic Scoring**
        - **Combined Score** (ML 70% + Heuristic 30%)
        - **Extractive Summary**
        """)

    st.markdown("---")
    st.markdown("### 🧠 Linguistic Features Analyzed")
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
    st.markdown("## 💡 Media Literacy Tips for Students")
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
            st.markdown(content)

    st.markdown("---")
    st.markdown("### 📱 Useful Fact-Checking Resources")
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

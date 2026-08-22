

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import pickle
import joblib
import time

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Try to import transformers for summarization
try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tip-box {
        background-color: #e8f4f8;
        border-left: 5px solid #17a2b8;
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
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# TEXT PREPROCESSOR CLASS
# =============================================================================
class TextPreprocessor:
    """Comprehensive text preprocessor for fake news detection."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sensational_words = {
            'shocking', 'unbelievable', 'breaking', 'urgent', 'exclusive',
            'bombshell', 'stunning', 'incredible', 'horrifying', 'terrifying',
            'amazing', 'miracle', 'secret', 'conspiracy', 'exposed', 'revealed',
            'banned', 'censored', 'they dont want you to know', 'wake up',
            'share before deleted', 'must see', 'jaw dropping', 'mind blowing',
            'you wont believe', 'outrageous', 'insane', 'epic', 'destroyed',
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

    def preprocess_for_tfidf(self, text):
        text = self.clean_text(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens
                  if t not in self.stop_words and len(t) > 2 and not t.isdigit()]
        return ' '.join(tokens)

    def extract_linguistic_features(self, text):
        text = self.clean_text(text)
        features = {}

        if len(text) == 0:
            return {k: 0 for k in [
                'char_count', 'word_count', 'sentence_count', 'avg_word_length',
                'avg_sentence_length', 'vocabulary_richness', 'uppercase_ratio',
                'exclamation_count', 'question_count', 'ellipsis_count',
                'capital_word_count', 'sensational_word_count', 'sentiment_polarity',
                'sentiment_subjectivity', 'digit_ratio', 'punctuation_ratio',
                'quote_count', 'paragraph_count', 'long_word_ratio',
                'stopword_ratio', 'type_token_ratio'
            ]}

        words = text.split()
        sentences = sent_tokenize(text)

        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0

        unique_words = set([w.lower() for w in words])
        features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0

        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        features['capital_word_count'] = sum(1 for w in words if w.isupper() and len(w) > 1)

        text_lower = text.lower()
        features['sensational_word_count'] = sum(1 for sw in self.sensational_words if sw in text_lower)

        blob = TextBlob(text[:5000])
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity

        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        features['quote_count'] = text.count('"') + text.count("'")
        features['paragraph_count'] = text.count('\n') + 1
        features['long_word_ratio'] = sum(1 for w in words if len(w) > 6) / len(words) if words else 0

        stop_count = sum(1 for w in words if w.lower() in self.stop_words)
        features['stopword_ratio'] = stop_count / len(words) if words else 0
        features['type_token_ratio'] = len(set(words)) / len(words) if words else 0

        return features


# =============================================================================
# CREDIBILITY SCORER CLASS
# =============================================================================
class CredibilityScorer:
    """Heuristic-based credibility scoring."""

    def __init__(self):
        self.credible_indicators = [
            'according to', 'research shows', 'study finds', 'data suggests',
            'officials say', 'experts say', 'reported by', 'sources confirm',
            'evidence suggests', 'analysis shows', 'statistics indicate',
            'peer reviewed', 'published in', 'university of', 'institute of'
        ]
        self.non_credible_indicators = [
            'you wont believe', 'they dont want you to know', 'shocking truth',
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

        source_score = sum(2.5 for ind in self.credible_indicators if ind in text_lower)
        source_score = min(source_score, 15)
        score += source_score
        breakdown['source_citations'] = source_score

        sensational_penalty = sum(-3 for ind in self.non_credible_indicators if ind in text_lower)
        sensational_penalty = max(sensational_penalty, -20)
        score += sensational_penalty
        breakdown['sensationalism_penalty'] = sensational_penalty

        words = text.split()
        if len(words) > 0:
            avg_word_len = np.mean([len(w) for w in words])
            quality_score = 10 if 4 <= avg_word_len <= 7 else (5 if 3 <= avg_word_len <= 8 else 0)
        else:
            quality_score = 0
        score += quality_score
        breakdown['writing_quality'] = quality_score

        if len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            caps_penalty = -10 if caps_ratio > 0.3 else (-5 if caps_ratio > 0.15 else 0)
        else:
            caps_penalty = 0
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

        blob = TextBlob(text[:3000])
        subjectivity = blob.sentiment.subjectivity
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
# SUMMARIZATION FUNCTIONS
# =============================================================================
def extractive_summary(text, num_sentences=3):
    if not text or len(str(text).strip()) == 0:
        return "No text provided for summarization."
    text = str(text)
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    words = word_tokenize(text.lower())
    stop_words_set = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words_set]
    word_freq = Counter(words)
    max_freq = max(word_freq.values()) if word_freq else 1
    word_freq = {w: f / max_freq for w, f in word_freq.items()}

    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = word_tokenize(sent.lower())
        score = sum(word_freq.get(w, 0) for w in sent_words if w.isalnum())
        score = score / (len(sent_words) + 1)
        if i < 3:
            score *= 1.2
        sentence_scores[i] = score

    top_indices = sorted(
        sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    )
    return ' '.join([sentences[i] for i in top_indices])


@st.cache_resource
def load_summarizer():
    """Load the transformer summarization model."""
    if HAS_TRANSFORMERS:
        try:
            return hf_pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1
            )
        except Exception:
            return None
    return None


def generate_summary(text, summarizer_model=None, max_length=150, min_length=50):
    if not text or len(str(text).strip()) < 50:
        return "Text too short to summarize."
    text = str(text)
    if summarizer_model is not None:
        try:
            input_text = text[:2048]
            result = summarizer_model(
                input_text, max_length=max_length, min_length=min_length,
                do_sample=False, truncation=True
            )
            return result[0]['summary_text']
        except Exception:
            return extractive_summary(text)
    return extractive_summary(text)


# =============================================================================
# LOAD OR TRAIN MODEL
# =============================================================================
@st.cache_resource
def load_or_train_model():
    """Load saved model or train a new one."""
    preprocessor = TextPreprocessor()
    scorer = CredibilityScorer()

    # Try loading saved models
    if all(os.path.exists(p) for p in [
        'models/fake_news_model.joblib',
        'models/tfidf_vectorizer.joblib',
        'models/feature_scaler.joblib'
    ]):
        model = joblib.load('models/fake_news_model.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        scaler = joblib.load('models/feature_scaler.joblib')
        return model, vectorizer, scaler, preprocessor, scorer, True

    # If no saved model, train from scratch
    try:
        import kagglehub
        path = kagglehub.dataset_download("mucahiddemircan/real-and-fake-news-dataset")

        true_path, fake_path = None, None
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                if 'true' in file.lower() or 'real' in file.lower():
                    true_path = filepath
                elif 'fake' in file.lower():
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
            df = pd.read_csv(csv_files[0])

        if 'text' in df.columns and 'title' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['full_text'] = df['text'].fillna('')
        else:
            df['full_text'] = df['title'].fillna('')

        df['cleaned_text'] = df['full_text'].apply(preprocessor.preprocess_for_tfidf)
        linguistic_features = df['full_text'].apply(preprocessor.extract_linguistic_features)
        linguistic_df = pd.DataFrame(linguistic_features.tolist())

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X_text_train, _, X_ling_train, _, y_train, _ = train_test_split(
            df['cleaned_text'], linguistic_df.values, df['label'],
            test_size=0.2, random_state=42, stratify=df['label']
        )

        vectorizer = TfidfVectorizer(
            max_features=50000, ngram_range=(1, 3), min_df=3, max_df=0.95,
            sublinear_tf=True, dtype=np.float32
        )
        X_tfidf = vectorizer.fit_transform(X_text_train)

        scaler = StandardScaler()
        X_ling_scaled = scaler.fit_transform(X_ling_train)
        X_combined = hstack([X_tfidf, csr_matrix(X_ling_scaled)])

        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1)
        model.fit(X_combined, y_train)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/fake_news_model.joblib')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
        joblib.dump(scaler, 'models/feature_scaler.joblib')

        return model, vectorizer, scaler, preprocessor, scorer, True

    except Exception as e:
        st.error(f"Error loading/training model: {e}")
        return None, None, None, preprocessor, scorer, False


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_article(text, title, model, vectorizer, scaler, preprocessor, scorer, summarizer_model):
    """Complete article analysis."""
    full_text = text
    if title:
        full_text = title + ' ' + text

    # Preprocess
    cleaned = preprocessor.preprocess_for_tfidf(full_text)

    # Extract features
    tfidf_features = vectorizer.transform([cleaned])
    linguistic_features = preprocessor.extract_linguistic_features(full_text)
    ling_array = np.array([list(linguistic_features.values())])
    ling_scaled = scaler.transform(ling_array)
    combined = hstack([tfidf_features, csr_matrix(ling_scaled)])

    # ML Prediction
    ml_prediction = model.predict(combined)[0]
    ml_probability = model.predict_proba(combined)[0]

    # Heuristic scoring
    credibility_score, score_breakdown = scorer.score_article(full_text)

    # Combined assessment
    ml_real_prob = ml_probability[1]
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

    # Summary
    summary = generate_summary(full_text, summarizer_model)

    return {
        'verdict': verdict,
        'confidence_level': confidence_level,
        'combined_score': round(combined_score, 1),
        'ml_prediction': 'Real' if ml_prediction == 1 else 'Fake',
        'ml_confidence': round(max(ml_probability) * 100, 1),
        'ml_real_probability': round(ml_real_prob * 100, 1),
        'ml_fake_probability': round(ml_probability[0] * 100, 1),
        'credibility_score': credibility_score,
        'score_breakdown': score_breakdown,
        'linguistic_features': linguistic_features,
        'summary': summary,
        'word_count': linguistic_features.get('word_count', 0),
        'sentiment_polarity': round(linguistic_features.get('sentiment_polarity', 0), 3),
        'sentiment_subjectivity': round(linguistic_features.get('sentiment_subjectivity', 0), 3)
    }


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake News Detector for Students</h1>
        <p>AI-powered tool to analyze articles, assess credibility, and generate trustworthy summaries</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/doodle/96/000000/news.png", width=80)
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
        st.info("Model: Logistic Regression + TF-IDF + Linguistic Features")

    # Load model
    with st.spinner("Loading AI model... This may take a moment on first run."):
        model, vectorizer, scaler_obj, preprocessor, scorer, model_loaded = load_or_train_model()

    # Load summarizer
    summarizer_model = load_summarizer()

    if page == "🏠 Home - Article Analyzer":
        render_home(model, vectorizer, scaler_obj, preprocessor, scorer,
                    summarizer_model, model_loaded, show_detailed, show_breakdown)
    elif page == "📚 How It Works":
        render_how_it_works()
    elif page == "💡 Media Literacy Tips":
        render_tips()


def render_home(model, vectorizer, scaler_obj, preprocessor, scorer,
                summarizer_model, model_loaded, show_detailed, show_breakdown):
    """Render the main article analyzer page."""

    st.markdown("## 📝 Paste an Article to Analyze")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["✍️ Paste Article Text", "📋 Try Example Articles"],
        horizontal=True
    )

    if input_method == "📋 Try Example Articles":
        example_choice = st.selectbox(
            "Select an example:",
            [
                "🟢 Real News Example",
                "🔴 Fake News Example",
                "🟡 Clickbait Example"
            ]
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
                routines.' The findings were corroborated by independent analysis from the World Health Organization. 
                Health officials recommend that adults aim for at least 150 minutes of moderate aerobic activity 
                per week, combined with muscle-strengthening activities on two or more days per week."""
            },
            "🔴 Fake News Example": {
                "title": "SHOCKING!!! Government HIDING Miracle Cure!!!",
                "text": """You WON'T BELIEVE what they've been keeping from us!!! A SECRET cure for ALL diseases 
                has been discovered but Big Pharma doesn't want you to know!!! EXPOSED: The deep state conspiracy 
                to keep us sick and dependent on their POISON medications!!! SHARE THIS BEFORE THEY DELETE IT!!! 
                Wake up sheeple!!! The mainstream media LIES about everything!!! This AMAZING miracle cure has been 
                BANNED because it would DESTROY the pharmaceutical industry!!! ACT NOW before it's too late!!! 
                They don't want you to know the SHOCKING TRUTH!!! Click here to learn the secret!!!"""
            },
            "🟡 Clickbait Example": {
                "title": "This One Weird Trick Will Change Your Life Forever!",
                "text": """Scientists are baffled by this incredible discovery that could change everything we know 
                about health. A mysterious fruit found in a remote island has properties that experts say could 
                revolutionize medicine. While some researchers have expressed interest, no clinical trials have been 
                conducted yet. The story has been shared millions of times on social media, with many claiming 
                miraculous results. However, no peer-reviewed studies have confirmed these claims. Several fact-checking 
                organizations have flagged similar stories as misleading. The article originally appeared on a website 
                known for publishing sensational health claims without scientific backing."""
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

    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

    if analyze_button and article_text.strip():
        if not model_loaded:
            st.error("⚠️ Model not loaded. Please ensure the model files exist in the 'models/' directory or that kagglehub can download the dataset.")
            return

        with st.spinner("🔄 Analyzing article... Please wait."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            result = analyze_article(
                article_text, article_title, model, vectorizer, scaler_obj,
                preprocessor, scorer, summarizer_model
            )

        progress_bar.empty()

        # Display Results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Verdict Banner
        verdict = result['verdict']
        if "REAL" in verdict:
            verdict_class = "verdict-real"
            verdict_emoji = "✅"
            verdict_color = "#28a745"
        elif "FAKE" in verdict:
            verdict_class = "verdict-fake"
            verdict_emoji = "❌"
            verdict_color = "#dc3545"
        else:
            verdict_class = "verdict-uncertain"
            verdict_emoji = "⚠️"
            verdict_color = "#ffc107"

        st.markdown(f"""
        <div class="{verdict_class}">
            <h2>{verdict_emoji} Verdict: {verdict}</h2>
            <p><strong>Confidence Level:</strong> {result['confidence_level']}</p>
            <p><strong>Overall Credibility Score:</strong> {result['combined_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🤖 ML Prediction",
                result['ml_prediction'],
                f"{result['ml_confidence']}% confident"
            )

        with col2:
            st.metric(
                "📊 Credibility Score",
                f"{result['credibility_score']}/100",
                f"{'Good' if result['credibility_score'] >= 60 else 'Concerning'}"
            )

        with col3:
            st.metric(
                "📝 Word Count",
                result['word_count'],
                f"{'Substantial' if result['word_count'] > 200 else 'Short'}"
            )

        with col4:
            subj = result['sentiment_subjectivity']
            st.metric(
                "🎭 Subjectivity",
                f"{subj:.2f}",
                f"{'Objective' if subj < 0.4 else 'Subjective'}"
            )

        # Summary Section
        st.markdown("### 📋 Article Summary")
        st.markdown(f"""
        <div class="summary-box">
            <p>{result['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        if show_detailed:
            st.markdown("### 🔬 Detailed Analysis")

            col_detail1, col_detail2 = st.columns(2)

            with col_detail1:
                st.markdown("#### ML Model Assessment")
                st.write(f"**Probability of being Real:** {result['ml_real_probability']}%")
                st.write(f"**Probability of being Fake:** {result['ml_fake_probability']}%")

                # Probability bar
                st.progress(result['ml_real_probability'] / 100)
                st.caption(f"Real ← {result['ml_real_probability']}% | {result['ml_fake_probability']}% → Fake")

            with col_detail2:
                st.markdown("#### Sentiment Analysis")
                st.write(f"**Polarity:** {result['sentiment_polarity']} "
                         f"({'Positive' if result['sentiment_polarity'] > 0 else 'Negative' if result['sentiment_polarity'] < 0 else 'Neutral'})")
                st.write(f"**Subjectivity:** {result['sentiment_subjectivity']} "
                         f"({'Objective' if result['sentiment_subjectivity'] < 0.4 else 'Subjective'})")

                # Linguistic features
                ling = result['linguistic_features']
                st.write(f"**Exclamation Marks:** {ling.get('exclamation_count', 0)}")
                st.write(f"**Sensational Words:** {ling.get('sensational_word_count', 0)}")
                st.write(f"**Vocabulary Richness:** {ling.get('vocabulary_richness', 0):.3f}")

        if show_breakdown:
            st.markdown("### 📈 Credibility Score Breakdown")

            breakdown = result['score_breakdown']
            breakdown_data = {
                'Factor': [],
                'Points': [],
                'Impact': []
            }
            for factor, points in breakdown.items():
                if factor != 'subjectivity_value':
                    impact = "🟢 Positive" if points > 0 else ("🔴 Negative" if points < 0 else "⚪ Neutral")
                    breakdown_data['Factor'].append(factor.replace('_', ' ').title())
                    breakdown_data['Points'].append(points)
                    breakdown_data['Impact'].append(impact)

            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        # Student Tips
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
            - Be wary of emotional manipulation and sensational language
            """)
        else:
            st.warning("""
            ⚠️ The credibility of this article is uncertain:
            - Seek additional sources before forming an opinion
            - Look for direct quotes and data sources in the article
            - Check if the author and publication are established
            - Use fact-checking tools for specific claims
            """)

    elif analyze_button and not article_text.strip():
        st.warning("⚠️ Please enter some article text to analyze.")


def render_how_it_works():
    """Render the 'How It Works' page."""
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
        - **TF-IDF Vectorization**: Converts text into numerical features 
          using word frequency analysis
        - **N-gram Analysis**: Examines word patterns (unigrams, bigrams, trigrams)
        - **Linguistic Feature Extraction**: 21 handcrafted features including 
          sentiment, readability, and stylistic patterns
        """)

    with col2:
        st.markdown("""
        #### 2️⃣ ML Classification
        - **Trained Classifier**: Logistic Regression model trained on 
          44,000+ real and fake news articles
        - **Feature Combination**: Combines TF-IDF text features with 
          linguistic features for robust prediction
        - **Probability Estimation**: Outputs confidence scores for 
          both real and fake classifications
        """)

    with col3:
        st.markdown("""
        #### 3️⃣ Credibility Assessment
        - **Heuristic Scoring**: Rule-based checks for source citations, 
          sensationalism, writing quality
        - **Combined Score**: Weighted combination of ML prediction (70%) 
          and heuristic score (30%)
        - **Summary Generation**: AI-powered article summarization for 
          quick understanding
        """)

    st.markdown("---")
    st.markdown("### 🧠 Linguistic Features Analyzed")

    features_explained = {
        "Sentiment Polarity": "Measures if the text is positive, negative, or neutral",
        "Subjectivity Score": "Determines if the text is objective (fact-based) or subjective (opinion-based)",
        "Sensational Word Count": "Counts clickbait/emotional trigger words",
        "Vocabulary Richness": "Measures diversity of word usage",
        "Uppercase Ratio": "Excessive capitals often indicate unreliable content",
        "Exclamation Marks": "Overuse of ! is a common fake news indicator",
        "Average Word Length": "Professional writing tends to have consistent word lengths",
        "Sentence Length": "Very short or very long sentences may indicate quality issues",
        "Source Citations": "Presence of references to studies, experts, or data",
        "Type-Token Ratio": "Lexical diversity measurement"
    }

    for feature, description in features_explained.items():
        st.markdown(f"- **{feature}**: {description}")

    st.markdown("---")
    st.markdown("### 📊 Model Training Details")

    st.info("""
    - **Dataset**: Real and Fake News Dataset from Kaggle (44,000+ articles)
    - **Algorithm**: Logistic Regression with combined TF-IDF + Linguistic features
    - **TF-IDF Features**: Up to 50,000 features with tri-gram analysis
    - **Linguistic Features**: 21 handcrafted features
    - **Validation**: 5-fold stratified cross-validation
    """)


def render_tips():
    """Render the media literacy tips page."""
    st.markdown("## 💡 Media Literacy Tips for Students")

    st.markdown("""
    Learning to identify fake news is an essential skill in the digital age. 
    Here are practical tips to help you evaluate news sources critically.
    """)

    tips = [
        {
            "title": "🔍 Check the Source",
            "content": """
            - Is the website well-known and reputable?
            - Does it have a professional design with proper 'About Us' and 'Contact' pages?
            - Check the URL: fake sites often mimic real ones (e.g., 'ABCnews.com.co')
            - Look up the domain on sites like Whois.net
            """
        },
        {
            "title": "👤 Verify the Author",
            "content": """
            - Is the author a real person with verifiable credentials?
            - Search for other articles by the same author
            - Does the author have expertise in the topic they're writing about?
            - Check their social media profiles for legitimacy
            """
        },
        {
            "title": "📅 Check the Date",
            "content": """
            - Is the article current or is old news being reshared?
            - Sometimes real articles are shared out of context with misleading dates
            - Check if the events described are timely and relevant
            """
        },
        {
            "title": "🔗 Cross-Reference",
            "content": """
            - Search for the same story on multiple trusted news sources
            - If only one source is reporting it, be skeptical
            - Use fact-checking websites:
              - [Snopes.com](https://www.snopes.com)
              - [FactCheck.org](https://www.factcheck.org)
              - [PolitiFact](https://www.politifact.com)
              - [Reuters Fact Check](https://www.reuters.com/fact-check)
            """
        },
        {
            "title": "🎭 Watch for Emotional Manipulation",
            "content": """
            - Fake news often tries to make you angry, scared, or outraged
            - Be suspicious of ALL-CAPS text and excessive exclamation marks!!!
            - Clickbait headlines like 'You won't believe...' are red flags
            - Take a pause before sharing emotionally charged content
            """
        },
        {
            "title": "🖼️ Verify Images and Videos",
            "content": """
            - Use Google Reverse Image Search to check if images are taken out of context
            - Look for signs of photo manipulation
            - Check if the image matches the article's claims
            - Be especially cautious of AI-generated images (deepfakes)
            """
        },
        {
            "title": "📊 Look for Evidence",
            "content": """
            - Does the article cite specific studies, data, or expert opinions?
            - Are there links to original sources?
            - Does it quote named officials or use vague terms like 'experts say'?
            - Check if the statistics cited are accurate
            """
        },
        {
            "title": "🧠 Check Your Own Biases",
            "content": """
            - We're more likely to believe news that confirms our existing beliefs
            - Be extra critical of articles that perfectly align with your views
            - Seek out diverse perspectives on important issues
            - Practice intellectual humility
            """
        }
    ]

    for tip in tips:
        with st.expander(tip["title"], expanded=False):
            st.markdown(tip["content"])

    st.markdown("---")
    st.markdown("### 🏫 Classroom Activity Ideas")
    st.markdown("""
    1. **Spot the Fake**: Give students a mix of real and fake articles to classify
    2. **Source Investigation**: Research the credibility of different news websites
    3. **Headline Analysis**: Rewrite sensational headlines to be factual
    4. **Fact-Check Challenge**: Verify claims from social media posts
    5. **Create a News Evaluation Checklist**: Students develop their own criteria
    """)

    st.markdown("---")
    st.markdown("### 📱 Useful Tools and Resources")
    resources = {
        "Google Fact Check Explorer": "https://toolbox.google.com/factcheck/explorer",
        "Snopes": "https://www.snopes.com",
        "FactCheck.org": "https://www.factcheck.org",
        "News Literacy Project": "https://newslit.org",
        "MediaWise": "https://www.poynter.org/mediawise/",
        "AllSides Media Bias Chart": "https://www.allsides.com/media-bias/media-bias-chart"
    }

    for name, url in resources.items():
        st.markdown(f"- [{name}]({url})")


if __name__ == "__main__":
    main()
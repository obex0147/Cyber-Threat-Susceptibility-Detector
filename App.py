# Cyber Threat Susceptibility Detector
import pandas as pd
import numpy as np
import re
import string
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from textblob import TextBlob
import joblib
from imblearn.over_sampling import SMOTE

# Load NLP and LLM
nlp = spacy.load("en_core_web_lg")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# --- Clean text input ---
def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_.isalpha()]
    return " ".join(tokens)

# --- feature extraction ---
def extract_features(text):
    doc = nlp(text)
    emotion_result = emotion_classifier(text[:512])[0]
    dominant_emotion = sorted(emotion_result, key=lambda x: x['score'], reverse=True)[0]['label']
    emotion_scores = {e['label']: e['score'] for e in emotion_result}
    
    # More comprehensive cybersecurity keywords
    cyber_keywords = {
        'phishing': ['verify', 'account', 'login', 'password', 'urgent', 'click', 'link', 'suspend', 'expire', 'confirm', 'update', 'security', 'alert'],
        'ransomware': ['encrypt', 'payment', 'bitcoin', 'restore', 'decrypt', 'locked', 'file', 'recover', 'crypto', 'ransom', 'deadline'],
        'social_engineering': ['trust', 'friend', 'help', 'colleague', 'boss', 'authority', 'emergency', 'favor', 'personal', 'confidential', 'secret'],
        'malware': ['download', 'install', 'update', 'scan', 'software', 'virus', 'attachment', 'executable', 'free', 'winner', 'prize']
    }
    
    # Count cybersecurity keyword occurrences (weighted)
    keyword_counts = {}
    for category, keywords in cyber_keywords.items():
        count = sum(text.lower().count(word) for word in keywords)
        keyword_counts[f"{category}_keywords"] = count
    
    # linguistic features
    questions = len(re.findall(r'\?', text))
    exclamations = len(re.findall(r'!', text))
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    
    # Urgency and trust indicators
    urgency_words = ['urgent', 'immediate', 'now', 'asap', 'quickly', 'fast', 'hurry', 'deadline', 'expire']
    trust_words = ['trust', 'believe', 'confirm', 'official', 'legitimate', 'authentic', 'verify', 'secure']
    fear_words = ['danger', 'risk', 'threat', 'warning', 'alert', 'problem', 'issue', 'error']
    
    urgency_score = sum(text.lower().count(word) for word in urgency_words)
    trust_score = sum(text.lower().count(word) for word in trust_words)
    fear_score = sum(text.lower().count(word) for word in fear_words)
    
    # Sentiment analysis
    blob = TextBlob(text)
    
    return {
        'word_count': len(doc),
        'avg_word_len': np.mean([len(t.text) for t in doc]) if len(doc) > 0 else 0,
        'questions': questions,
        'exclamations': exclamations,
        'caps_words': caps_words,
        'urgency_score': urgency_score,
        'trust_score': trust_score,
        'fear_score': fear_score,
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'dominant_emotion': dominant_emotion,
        'joy_score': emotion_scores.get('joy', 0),
        'sadness_score': emotion_scores.get('sadness', 0),
        'anger_score': emotion_scores.get('anger', 0),
        'fear_score_emotion': emotion_scores.get('fear', 0),
        'surprise_score': emotion_scores.get('surprise', 0),
        'disgust_score': emotion_scores.get('disgust', 0),
        **keyword_counts
    }

# ---  Real data processing ---
def create_cybersecurity_labels(posts, mbti_type):
    """Create more sophisticated cybersecurity labels based on text content and personality"""
    
    # Extract features from the text
    features = extract_features(posts[:1000])  # Limit text length for processing
    
    # Personality-based susceptibility mapping (more nuanced)
    personality_weights = {
        'phishing': {'E': 0.1, 'I': 0.2, 'S': 0.3, 'N': 0.1, 'T': 0.1, 'F': 0.3, 'J': 0.4, 'P': 0.1},
        'ransomware': {'E': 0.1, 'I': 0.3, 'S': 0.2, 'N': 0.2, 'T': 0.3, 'F': 0.1, 'J': 0.2, 'P': 0.3},
        'social_engineering': {'E': 0.4, 'I': 0.1, 'S': 0.2, 'N': 0.2, 'T': 0.1, 'F': 0.5, 'J': 0.2, 'P': 0.2},
        'malware': {'E': 0.2, 'I': 0.2, 'S': 0.1, 'N': 0.4, 'T': 0.2, 'F': 0.2, 'J': 0.1, 'P': 0.4}
    }
    
    # Calculate base scores for each attack type
    attack_scores = {}
    for attack_type, weights in personality_weights.items():
        score = sum(weights.get(trait, 0) for trait in mbti_type)
        attack_scores[attack_type] = score
    
    # Adjust scores based on text content features
    # Phishing adjustments
    if features['urgency_score'] > 2 or features['trust_score'] > 1:
        attack_scores['phishing'] *= 1.8
    
    # Ransomware adjustments  
    if features['fear_score'] > 1 or features['ransomware_keywords'] > 0:
        attack_scores['ransomware'] *= 2.0
    
    # Social engineering adjustments
    if features['social_engineering_keywords'] > 0 or features['dominant_emotion'] in ['joy', 'surprise']:
        attack_scores['social_engineering'] *= 1.7
    
    # Malware adjustments
    if features['malware_keywords'] > 0 or features['dominant_emotion'] in ['joy', 'surprise']:
        attack_scores['malware'] *= 1.6
    
    # Add some content-based randomness to avoid deterministic results
    content_hash = hash(posts[:100]) % 100
    randomness_factor = 0.8 + (content_hash / 100) * 0.4  # Range: 0.8 to 1.2
    
    # Apply randomness
    for attack_type in attack_scores:
        attack_scores[attack_type] *= randomness_factor
    
    # Select attack type with highest score
    return max(attack_scores, key=attack_scores.get)

# --- Train model with real MBTI data ---
def train_model():
    """Train model using real MBTI dataset with enhanced labeling"""
    
    try:
        # Load MBTI dataset
        mbti_df = pd.read_csv("mbti_1.csv")
        
        # Clean and prepare data
        mbti_df = mbti_df.dropna(subset=['posts', 'type'])
        mbti_df['cleaned_posts'] = mbti_df['posts'].apply(clean_text)
        
        # Remove very short posts (less than 10 words)
        mbti_df = mbti_df[mbti_df['cleaned_posts'].str.split().str.len() >= 10]
        
        # Create cybersecurity labels using enhanced logic
        mbti_df['susceptible_attack'] = mbti_df.apply(
            lambda row: create_cybersecurity_labels(row['posts'], row['type']), 
            axis=1
        )
        
        # Sample data to balance classes and improve performance
        sampled_data = []
        for attack_type in ['phishing', 'ransomware', 'social_engineering', 'malware']:
            attack_data = mbti_df[mbti_df['susceptible_attack'] == attack_type]
            if len(attack_data) > 500:  # Limit to prevent one class dominating
                attack_data = attack_data.sample(n=500, random_state=42)
            sampled_data.append(attack_data)
        
        # Combine sampled data
        final_df = pd.concat(sampled_data, ignore_index=True)
        
        # Shuffle the data
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Extract features from the cleaned posts
        features_list = []
        labels_list = []
        
        for idx, row in final_df.iterrows():
            try:
                features = extract_features(row['cleaned_posts'])
                features_list.append(features)
                labels_list.append(row['susceptible_attack'])
            except Exception as e:
                # Skip problematic rows
                continue
        
        # Create feature DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Convert categorical features
        features_df['dominant_emotion'] = features_df['dominant_emotion'].astype('category')
        features_df['emotion_code'] = features_df['dominant_emotion'].cat.codes
        
        # Prepare features and target
        X = features_df.drop(columns=['dominant_emotion'])
        y = pd.Series(labels_list).astype('category')
        y_encoded = y.cat.codes
        label_mapping = dict(enumerate(y.cat.categories))
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(X)-1))
        X_res, y_res = smote.fit_resample(X, y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )
        
        # Create pipeline with scaling and classifier
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, label_mapping, accuracy, len(final_df)
        
    except FileNotFoundError:
        st.error("MBTI dataset (mbti_1.csv) not found. Please ensure the file is in the correct location.")
        return None, None, 0, 0
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, 0, 0

# --- Rule-based prediction enhancement ---
def enhance_prediction_with_rules(text, base_prediction, base_probabilities):
    """Apply rule-based enhancements to improve prediction accuracy"""
    
    text_lower = text.lower()
    
    # Rule adjustments
    adjustments = np.zeros(len(base_probabilities))
    
    # Phishing indicators
    if any(word in text_lower for word in ['verify', 'account', 'login', 'suspend', 'expire']):
        adjustments[0] += 0.3  # Boost phishing
    
    # Ransomware indicators
    if any(word in text_lower for word in ['encrypt', 'bitcoin', 'ransom', 'payment', 'decrypt']):
        adjustments[1] += 0.4  # Boost ransomware
    
    # Social engineering indicators
    if any(word in text_lower for word in ['boss', 'colleague', 'help', 'emergency', 'personal']):
        adjustments[2] += 0.3  # Boost social engineering
    
    # Malware indicators
    if any(word in text_lower for word in ['download', 'install', 'free', 'winner', 'prize']):
        adjustments[3] += 0.3  # Boost malware
    
    # Apply adjustments
    enhanced_probs = base_probabilities + adjustments
    enhanced_probs = np.maximum(enhanced_probs, 0)  # Ensure non-negative
    enhanced_probs = enhanced_probs / enhanced_probs.sum()  # Normalize
    
    return enhanced_probs

# --- Streamlit App ---
st.title("🔐 Cyber Threat Susceptibility Detector")
st.markdown("""
This enhanced tool analyzes your text input to determine susceptibility to different cyber threats.
The model uses advanced NLP features and rule-based enhancements for more accurate predictions.
""")

# Train model (not cached for dynamic behavior)
if 'model' not in st.session_state:
    with st.spinner("Training model with real MBTI data..."):
        model, label_map, accuracy, data_size = train_model()
        if model is not None:
            st.session_state['model'] = model
            st.session_state['label_map'] = label_map
            st.session_state['accuracy'] = accuracy
            st.session_state['data_size'] = data_size
        else:
            st.error("Failed to train model. Please check if mbti_1.csv is available.")
            st.stop()

# Display model info
if 'model' in st.session_state:
    st.sidebar.metric("Model Accuracy", f"{st.session_state['accuracy']:.1%}")
    st.sidebar.metric("Training Samples", st.session_state['data_size'])
    st.sidebar.info("Using real MBTI personality data for training")

# Input section
user_input = st.text_area("✍️ Enter text to analyze for cyber threat susceptibility:", height=150)

if st.button("🔎 Analyze Susceptibility"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing your text..."):
            try:
                # Extract features
                features = extract_features(user_input)
                features['emotion_code'] = pd.Series([features['dominant_emotion']]).astype('category').cat.codes[0]
                
                # Prepare input
                input_df = pd.DataFrame([features])
                input_df = input_df.drop(columns=['dominant_emotion'])
                
                # Get base prediction
                base_proba = st.session_state['model'].predict_proba(input_df)[0]
                
                # Apply rule-based enhancements
                enhanced_proba = enhance_prediction_with_rules(user_input, None, base_proba)
                
                # Get final prediction
                threat_type = st.session_state['label_map'][np.argmax(enhanced_proba)]
                confidence = np.max(enhanced_proba)
                
                # Calculate susceptibility level
                if confidence > 0.7:
                    susceptibility_level = "High"
                    color = "🔴"
                elif confidence > 0.4:
                    susceptibility_level = "Medium"
                    color = "🟡"
                else:
                    susceptibility_level = "Low"
                    color = "🟢"
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Primary Threat:** {threat_type.upper()}")
                    st.metric("Confidence", f"{confidence:.1%}")
                with col2:
                    st.warning(f"**Susceptibility:** {color} {susceptibility_level}")
                
                # Detailed analysis
                with st.expander("📊 Detailed Analysis"):
                    st.subheader("Threat Probability Distribution")
                    
                    prob_df = pd.DataFrame({
                        'Threat Type': [st.session_state['label_map'][i] for i in range(len(enhanced_proba))],
                        'Probability': enhanced_proba
                    }).sort_values('Probability', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(data=prob_df, x='Probability', y='Threat Type', palette='viridis')
                    plt.title("Threat Susceptibility Analysis")
                    plt.xlabel("Probability")
                    
                    # Add percentage labels
                    for i, bar in enumerate(bars.patches):
                        width = bar.get_width()
                        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.1%}', ha='left', va='center')
                    
                    st.pyplot(fig)
                    
                    # Feature analysis
                    st.subheader("Key Text Indicators")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Urgency Score", features['urgency_score'])
                        st.metric("Trust Score", features['trust_score'])
                        st.metric("Fear Score", features['fear_score'])
                        st.metric("Questions", features['questions'])
                    
                    with col2:
                        st.metric("Exclamations", features['exclamations'])
                        st.metric("Caps Words", features['caps_words'])
                        st.metric("Sentiment Polarity", f"{features['polarity']:.2f}")
                        st.metric("Dominant Emotion", features['dominant_emotion'])
                    
                    # Keyword breakdown
                    st.subheader("Cybersecurity Keywords Detected")
                    keyword_data = {
                        'Phishing': features['phishing_keywords'],
                        'Ransomware': features['ransomware_keywords'],
                        'Social Engineering': features['social_engineering_keywords'],
                        'Malware': features['malware_keywords']
                    }
                    
                    for category, count in keyword_data.items():
                        if count > 0:
                            st.success(f"{category}: {count} keywords found")
                        else:
                            st.info(f"{category}: No keywords detected")
                            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.error("Please try with different text or contact support.")

# Additional info
with st.expander("ℹ️ How it works"):
    st.markdown("""
    This tool uses multiple techniques to analyze cyber threat susceptibility:
    
    1. **Advanced NLP Features**: Sentiment analysis, emotion detection, linguistic patterns
    2. **Cybersecurity Keywords**: Comprehensive keyword matching for each threat type
    3. **Machine Learning**: Random Forest classifier with balanced class weights
    4. **Rule-based Enhancement**: Additional logic to improve prediction accuracy
    5. **Dynamic Training**: Model retrains each session for varied results
    
    **Threat Types:**
    - **Phishing**: Email/message scams asking for credentials
    - **Ransomware**: Malicious software that encrypts files for ransom
    - **Social Engineering**: Psychological manipulation tactics
    - **Malware**: Malicious software distribution attempts
    """)
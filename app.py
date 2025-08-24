# Cyber Threat Susceptibility Detector with Model 
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
import pickle
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE
import time

# Load NLP and LLM
nlp = spacy.load("en_core_web_lg")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Model storage directory
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Created directory: {MODEL_DIR}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        MODEL_DIR = "."  # Fallback to current directory

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

# --- Model saving and loading functions ---
def save_model(model, label_map, accuracy, data_size, model_name):
    """Save the trained model and associated metadata"""
    
    # Ensure directory exists
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            st.info(f"Created model directory: {MODEL_DIR}")
        except Exception as e:
            st.error(f"Cannot create model directory: {str(e)}")
            return None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.joblib"
    metadata_filename = f"{model_name}_{timestamp}_metadata.pkl"
    
    model_path = os.path.join(MODEL_DIR, model_filename)
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    
    try:
        # Test write permissions first
        test_file = os.path.join(MODEL_DIR, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        # Save model with explicit error handling
        st.info(f"Saving model to: {model_path}")
        joblib.dump(model, model_path)
        
        # Verify model file was created and has content
        if not os.path.exists(model_path):
            raise Exception(f"Model file was not created at {model_path}")
        
        model_size = os.path.getsize(model_path)
        if model_size == 0:
            raise Exception(f"Model file is empty")
        
        st.info(f"Model saved successfully. Size: {model_size} bytes")
        
        # Save metadata
        metadata = {
            'label_map': label_map,
            'accuracy': accuracy,
            'data_size': data_size,
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'model_path': model_path,
            'metadata_path': metadata_path
        }
        
        st.info(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Verify metadata file was created
        if not os.path.exists(metadata_path):
            raise Exception(f"Metadata file was not created at {metadata_path}")
        
        metadata_size = os.path.getsize(metadata_path)
        st.info(f"Metadata saved successfully. Size: {metadata_size} bytes")
        
        return model_path, metadata_path
    
    except PermissionError:
        st.error("Permission denied: Cannot write to the models directory")
        st.info(f"Try running the application from a directory where you have write permissions")
        st.info(f"Current working directory: {os.getcwd()}")
        return None, None
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        st.info(f"Attempted to save to: {model_path}")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")
        if os.path.exists(MODEL_DIR):
            st.info(f"MODEL_DIR is writable: {os.access(MODEL_DIR, os.W_OK)}")
        return None, None

def load_model(model_path, metadata_path):
    """Load a saved model and its metadata"""
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return model, metadata
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_saved_models():
    """Get list of all saved models"""
    
    model_files = []
    
    # Check if directory exists
    if not os.path.exists(MODEL_DIR):
        return model_files
    
    try:
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('_metadata.pkl'):
                try:
                    metadata_path = os.path.join(MODEL_DIR, filename)
                    model_filename = filename.replace('_metadata.pkl', '.joblib')
                    model_path = os.path.join(MODEL_DIR, model_filename)
                    
                    # Check if corresponding model file exists
                    if os.path.exists(model_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                        
                        model_files.append({
                            'display_name': f"{metadata['model_name']} - {metadata['created_at'][:16]} (Acc: {metadata['accuracy']:.1%})",
                            'model_path': model_path,
                            'metadata_path': metadata_path,
                            'metadata': metadata
                        })
                except Exception as e:
                    continue
    except Exception as e:
        st.error(f"Error reading models directory: {str(e)}")
    
    return sorted(model_files, key=lambda x: x['metadata']['created_at'], reverse=True)

def delete_model(model_path, metadata_path):
    """Delete a saved model and its metadata"""
    
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return True
    except Exception as e:
        st.error(f"Error deleting model: {str(e)}")
        return False

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

# --- Typing simulation function ---
def type_text(text, container, delay=0.02):
    """Simulate typing effect by updating text progressively"""
    typed_text = ""
    for char in text:
        typed_text += char
        container.markdown(typed_text + "▌")
        time.sleep(delay)
    container.markdown(typed_text)

# --- Streamlit App ---
st.title("🔐 Cyber Threat Susceptibility Detector")
st.markdown("""
This enhanced tool analyzes your text input to determine susceptibility to different cyber threats.
The model uses advanced NLP features and rule-based enhancements for more accurate predictions.
""")

# Sidebar for model management
st.sidebar.title("🤖 Model Management")

# Model selection section
saved_models = get_saved_models()

if saved_models:
    st.sidebar.subheader("Load Saved Model")
    
    selected_model = st.sidebar.selectbox(
        "Choose a saved model:",
        options=["None"] + [model['display_name'] for model in saved_models]
    )
    
    if selected_model != "None":
        if st.sidebar.button("Load Selected Model"):
            model_info = next((m for m in saved_models if m['display_name'] == selected_model), None)
            if model_info:
                with st.spinner("Loading saved model..."):
                    loaded_model, loaded_metadata = load_model(
                        model_info['model_path'], 
                        model_info['metadata_path']
                    )
                    
                    if loaded_model and loaded_metadata:
                        st.session_state['model'] = loaded_model
                        st.session_state['label_map'] = loaded_metadata['label_map']
                        st.session_state['accuracy'] = loaded_metadata['accuracy']
                        st.session_state['data_size'] = loaded_metadata['data_size']
                        st.sidebar.success(f"Model loaded successfully!")
                        st.experimental_rerun()
    
    # Delete model section
    st.sidebar.subheader("Delete Model")
    delete_model_select = st.sidebar.selectbox(
        "Choose model to delete:",
        options=["None"] + [model['display_name'] for model in saved_models],
        key="delete_select"
    )
    
    if delete_model_select != "None":
        if st.sidebar.button("🗑️ Delete Model", type="secondary"):
            model_info = next((m for m in saved_models if m['display_name'] == delete_model_select), None)
            if model_info:
                if delete_model(model_info['model_path'], model_info['metadata_path']):
                    st.sidebar.success("Model deleted successfully!")
                    st.experimental_rerun()

# Train new model section
st.sidebar.subheader("Train New Model")

model_name = st.sidebar.text_input("Model Name:", value="cyber_threat_model")

if st.sidebar.button("🔄 Train New Model"):
    with st.spinner("Training new model with real MBTI data..."):
        model, label_map, accuracy, data_size = train_model()
        if model is not None:
            st.session_state['model'] = model
            st.session_state['label_map'] = label_map
            st.session_state['accuracy'] = accuracy
            st.session_state['data_size'] = data_size
            st.sidebar.success("New model trained successfully!")
        else:
            st.sidebar.error("Failed to train model. Please check if mbti_1.csv is available.")

# Save current model section
if 'model' in st.session_state:
    st.sidebar.subheader("Save Current Model")
    
    if st.sidebar.button("💾 Save Current Model"):
        with st.spinner("Saving model..."):
            try:
                model_path, metadata_path = save_model(
                    st.session_state['model'],
                    st.session_state['label_map'],
                    st.session_state['accuracy'],
                    st.session_state['data_size'],
                    model_name
                )
                
                if model_path and metadata_path:
                    st.sidebar.success("Model saved successfully!")
                    st.sidebar.info(f"Saved to: {model_path}")
                    st.experimental_rerun()  # Refresh to show new model in list
                else:
                    st.sidebar.error("Failed to save model")
            except Exception as e:
                st.sidebar.error(f"Save error: {str(e)}")

# Load initial model if none exists
if 'model' not in st.session_state:
    if saved_models:
        # Try to load the most recent model
        with st.spinner("Loading most recent model..."):
            latest_model = saved_models[0]
            loaded_model, loaded_metadata = load_model(
                latest_model['model_path'], 
                latest_model['metadata_path']
            )
            
            if loaded_model and loaded_metadata:
                st.session_state['model'] = loaded_model
                st.session_state['label_map'] = loaded_metadata['label_map']
                st.session_state['accuracy'] = loaded_metadata['accuracy']
                st.session_state['data_size'] = loaded_metadata['data_size']
                st.info("Loaded most recent saved model")
    else:
        # Train model (not cached for dynamic behavior)
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
    elif 'model' not in st.session_state:
        st.error("No model available. Please train or load a model first.")
    else:
        # Create containers for typing simulation
        thinking_container = st.empty()
        analysis_container = st.empty()
        
        # Show thinking phase with typing simulation
        thinking_container.markdown("🤔 **Thinking...**")
        time.sleep(0.5)
        
        thinking_text = "Analyzing linguistic patterns, extracting cybersecurity keywords, processing emotions..."
        type_text(thinking_text, thinking_container, delay=0.03)
        
        time.sleep(1)
        thinking_container.empty()
        
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
            
            # Show results with typing effect
            result_container = st.empty()
            result_text = f"🎯 **Analysis Complete!**\n\n**Primary Threat:** {threat_type.upper()}\n**Confidence:** {confidence:.1%}\n**Susceptibility Level:** {color} {susceptibility_level}"
            type_text(result_text, result_container, delay=0.02)
            
            time.sleep(1)
            
            # Display detailed results
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
    5. **Model Persistence**: Save and load trained models for reuse
    6. **Interactive Analysis**: Typing simulation for better user experience
    
    **Model Management Features:**
    - Save trained models with metadata (accuracy, training date, etc.)
    - Load previously saved models
    - Delete unwanted models
    - Automatic loading of the most recent model
    
    **Threat Types:**
    - **Phishing**: Email/message scams asking for credentials
    - **Ransomware**: Malicious software that encrypts files for ransom
    - **Social Engineering**: Psychological manipulation tactics
    - **Malware**: Malicious software distribution attempts
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: #666; font-size: 14px;'>
        <p>🔐 Cyber Threat Susceptibility Detector</p>
        <p>Designed with ❤️ by <strong>Benneth Obioma</strong></p>
        <p><em>Protecting digital communications through intelligent threat analysis</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)

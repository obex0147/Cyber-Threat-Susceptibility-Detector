🔐 Cyber Threat Susceptibility Detector

A machine learning and NLP-powered application that analyzes user text to determine susceptibility to different types of cyber threats (Phishing, Ransomware, Social Engineering, and Malware).

The project combines linguistic analysis, emotion detection, and cybersecurity keyword extraction with machine learning models to predict vulnerability levels. It also integrates rule-based enhancements and model persistence for improved accuracy and usability.

📊 Dataset: MBTI Myers-Briggs Personality Dataset (Kaggle) https://www.kaggle.com/datasets/datasnaek/mbti-type

🚀 Features

Advanced NLP

Text cleaning, lemmatization, stopword removal using SpaCy

Emotion detection via DistilRoBERTa (HuggingFace model)

Sentiment polarity & subjectivity analysis (TextBlob)

Cybersecurity Indicators

Detection of phishing, ransomware, social engineering, and malware keywords

Linguistic features: questions, exclamations, capitalized words

Psychological cues: urgency, trust, fear scores

Machine Learning

Random Forest Classifier (Scikit-learn pipeline with scaling & class balancing)

Oversampling with SMOTE for handling class imbalance

Accuracy achieved: ~79.1%

Rule-Based Enhancements

Adds domain-specific heuristics 

Interactive Streamlit App

Real-time text analysis with typing simulation

Model management (train, save, load, delete models)

Detailed analysis with visualizations (Seaborn + Matplotlib)

▶️ Usage

Run the Streamlit app:

streamlit run app.py

Example Workflow

Enter text (e.g., a suspicious email or message).

The app analyzes:

Linguistic and emotional patterns

Cybersecurity keywords

Personality-based susceptibility (MBTI-informed)

Get predictions:

Primary Threat (Phishing, Ransomware, Social Engineering, Malware)

Confidence Level

Susceptibility Level (High, Medium, Low)

Review detailed analysis (keyword counts, emotional scores, probability distribution).

📊 Model Performance

Accuracy: ~79.1%

Evaluation: Calculated on a stratified 80/20 train-test split with balanced classes (via SMOTE).

Interpretation:

79.1% is a solid result given the noisy, self-reported dataset, avoding imbalance and over-sampling.

The model avoids overfitting by limiting tree depth, balancing class weights, and applying SMOTE.

Performance reflects the complexity of human personality and cyber risk behaviours.

🛠️ Tech Stack

Languages: Python

NLP: SpaCy, HuggingFace Transformers, TextBlob

ML: Scikit-learn, imbalanced-learn (SMOTE), RandomForestClassifier

Visualization: Matplotlib, Seaborn

App Framework: Streamlit
Pandasm Numpy, Joblib, Pickle etc

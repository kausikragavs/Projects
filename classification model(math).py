#CSI Round-2 ML(Tech) domain classification model question submission
#Done by S.Kausik Ragav
#P.S: Upon running the model, it accepts new input which can be entered by typing the question manually or by giving the path to the json question file

import json
import zipfile
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# CONFIGURATION
ZIP_FILE_PATH = r'E:\csi-round 2\dataset.zip' 

# JSON Keys
KEY_TEXT = "problem"
KEY_LABEL = "type"

# FUNCTIONS

def clean_math_text(text):
   
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Adding spaces around special math symbols so the model sees them clearly
    # +, -, =, ^, /, \, {, }, (, ) 
    text = re.sub(r'([+\-=^/(){}\[\]])', r' \1 ', text)
    
    # Removing weird characters but keeping standard math symbols and letters and numbers
    text = re.sub(r'[^a-z0-9\s+\-=^/(){}\[\]\\]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def load_and_split_data(zip_path):
    train_data = []
    test_data = []
    
    if not os.path.exists(zip_path):
        print(f"[!] Error: Zip file not found at {zip_path}")
        return pd.DataFrame(), pd.DataFrame()

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_files = z.namelist()
            json_files = [f for f in all_files if f.endswith('.json')]
            
            print(f"Scanning {len(json_files)} files inside the zip...")
            
            for filename in json_files:
                is_test_file = '/test/' in filename or filename.startswith('test/')
                
                with z.open(filename) as f:
                    try:
                        content = json.load(f)
                        
                        # Only proceed if we have both Question and Topic
                        if KEY_TEXT in content and KEY_LABEL in content:
                            
                            #Extract raw data
                            raw_text = content[KEY_TEXT]
                            label = content[KEY_LABEL] 
                             if label in ["Prealgebra", "Intermediate Algebra"]:
                                 label = "Algebra"

                            entry = {
                                'text': clean_math_text(raw_text),
                                'label': label
                            }
                            
                            if is_test_file:
                                test_data.append(entry)
                            else:
                                train_data.append(entry)
                    except:
                        continue
                        
    except Exception as e:
        print(f"[!] Error reading zip: {e}")
        return pd.DataFrame(), pd.DataFrame()

    return pd.DataFrame(train_data), pd.DataFrame(test_data)


#MAIN 

if __name__ == "__main__":
    print("--- Step 1: Reading Zip File ---")
    
    df_train, df_test = load_and_split_data(ZIP_FILE_PATH)
    
    if df_train.empty:
        print("STOP: No training data found. Check JSON keys or zip structure.")
        exit()

    print(f"Training Samples: {len(df_train)}")
    
    if df_test.empty:
        print("[Info] No 'test' folder found inside zip. Automatically splitting 80/20...")
        X = df_train['text']
        y = df_train['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print(f"Test Samples (from 'test' folder): {len(df_test)}")
        X_train = df_train['text']
        y_train = df_train['label']
        X_test = df_test['text']
        y_test = df_test['label']


    #TRAINING
    print("\n--- Step 2: Training Model ---")
    
    # I have reduced max_features slightly to 8000 to speed up processing
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=8000)),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    parameters = {
        'tfidf__min_df': [1, 5],
        'clf__C': [1, 10],
    }

    print("Running Grid Search (Optimizing)...")
    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")


    #EVALUATION
    print("\n--- Step 3: Evaluation ---")
    
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"\nFinal Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("Displaying Confusion Matrix...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, 
                yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    #INTERACTIVE MODE
    print("\n" + "="*40)
    print("      INTERACTIVE MODE")
    print("="*40)
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\n>> Enter question OR file path: ").strip()
        
        #Exit Check
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input:
            continue

        #Cleanup Input (remove quotes from path)
        cleaned_input = user_input.replace('"', '').replace("'", "")
        
        text_to_predict = ""
        source_type = "Text Input"

        # Check if it looks like a JSON file
        if cleaned_input.endswith('.json'):
            if os.path.exists(cleaned_input):
                # CASE A: Real file found on disk
                try:
                    with open(cleaned_input, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        text_to_predict = data.get(KEY_TEXT, "")
                        source_type = "File Loaded"
                except Exception as e:
                    print(f"[!] Error reading file: {e}")
                    continue
            else:
                # CASE B: File not found (Likely inside a ZIP)
                print(f"[!] ERROR: File not found: {cleaned_input}")
                if ".zip" in cleaned_input.lower():
                    print("    NOTE: You cannot pass a path inside a .zip file directly.")
                    print("    Please unzip the folder first, or copy-paste the question text.")
                continue # Stop here, do not try to predict on the filename
        else:
            # CASE C: Its just a typed question
            text_to_predict = user_input
            source_type = "Raw Text"

        #PREDICTION
        if text_to_predict:
            # Clean and Prep
            clean_text = clean_math_text(text_to_predict)
            
            #Show user what the model is actually seeing
            print(f"    [Analyzing]: '{clean_text[:60]}...'")

            if not clean_text:
                print("[!] Error: Text was empty after cleaning. Try typing more details.")
                continue

            # Predict
            prediction = best_model.predict([clean_text])[0]
            confidence = best_model.predict_proba([clean_text]).max()
            
            print(f">>> RESULT: {prediction}")
            print(f">>> CONFIDENCE: {confidence:.2%}")
        else:
            print("[!] Error: No text found to analyze.")

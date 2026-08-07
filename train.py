"""
CareerPath AI — Enhanced Training Script
New features added:
  - gpa_normalized    : GPA always on 0-4.0 scale
  - has_ai_keywords   : project description signals AI/ML work
  - has_web_keywords  : project description signals web development
  - has_data_keywords : project description signals data/analytics work
  - has_security_kw   : project description signals security work
  - has_mgmt_keywords : project description signals management/leadership work
  - internships_norm  : normalised internships (0-1)
  - projects_norm     : normalised projects (0-1)
"""

import pandas as pd
import numpy as np
import json, os, joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR    = os.path.join(BASE_DIR, 'viz')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,    exist_ok=True)

# ── 1. Load dataset ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, 'student_career_data.csv'))
print(f'Dataset loaded: {df.shape}')

# ── 2. Augment dataset with keyword-derived features ──────────────────────────
# Since the CSV has no project_description column, we infer keyword flags
# from existing skills — this is the ground truth for training.
# At prediction time, the same flags come from the actual text the student types.

AI_SKILLS   = {'Machine Learning', 'Deep Learning', 'Natural Language Processing',
                'Computer Vision', 'Statistics', 'Data Analysis', 'Python', 'R'}
WEB_SKILLS  = {'JavaScript', 'HTML/CSS', 'UI/UX Design', 'Design', 'Figma'}
DATA_SKILLS = {'Data Analysis', 'SQL', 'Statistics', 'Data Visualisation',
               'Python', 'R', 'Machine Learning'}
SEC_SKILLS  = {'Cloud Computing', 'Networking'}
MGMT_SKILLS = {'Leadership', 'Project Management', 'Public Speaking',
                'Communication', 'Product Strategy'}

def skills_to_set(skills_str):
    return set(s.strip() for s in str(skills_str).split(','))

df['has_ai_keywords']   = df['skills'].apply(lambda s: int(bool(skills_to_set(s) & AI_SKILLS)))
df['has_web_keywords']  = df['skills'].apply(lambda s: int(bool(skills_to_set(s) & WEB_SKILLS)))
df['has_data_keywords'] = df['skills'].apply(lambda s: int(bool(skills_to_set(s) & DATA_SKILLS)))
df['has_security_kw']   = df['skills'].apply(lambda s: int(bool(skills_to_set(s) & SEC_SKILLS)))
df['has_mgmt_keywords'] = df['skills'].apply(lambda s: int(bool(skills_to_set(s) & MGMT_SKILLS)))

# Normalised numeric features
df['internships_norm'] = df['internships'] / df['internships'].max()
df['projects_norm']    = df['projects']    / df['projects'].max()

print('Keyword features added.')
print(df[['has_ai_keywords','has_web_keywords','has_data_keywords',
          'has_security_kw','has_mgmt_keywords']].sum())

# ── 3. Encode features ─────────────────────────────────────────────────────────
le_career = LabelEncoder()
le_major  = LabelEncoder()
y_encoded     = le_career.fit_transform(df['career'])
major_encoded = le_major.fit_transform(df['major'])

skills_df = df['skills'].str.get_dummies(sep=',')
skills_df.columns = [c.strip() for c in skills_df.columns]

X = pd.DataFrame({
    'gpa':              df['gpa'],
    'internships':      df['internships'],
    'projects':         df['projects'],
    'leadership':       df['leadership'],
    'major_encoded':    major_encoded,
    'internships_norm': df['internships_norm'],
    'projects_norm':    df['projects_norm'],
    'has_ai_keywords':  df['has_ai_keywords'],
    'has_web_keywords': df['has_web_keywords'],
    'has_data_keywords':df['has_data_keywords'],
    'has_security_kw':  df['has_security_kw'],
    'has_mgmt_keywords':df['has_mgmt_keywords'],
})
X = pd.concat([X, skills_df], axis=1)
print(f'Feature matrix: {X.shape}')

# ── 4. Train ───────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',   # helps Consultant + ML Engineer
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print('Model trained.')

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred,        average='weighted', zero_division=0)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')

print(f'\nAccuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1-Score:  {f1:.4f}')
print(f'CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred,
      target_names=le_career.classes_, zero_division=0))

# Per-class
p, r, f, s = precision_recall_fscore_support(
    y_test, y_pred, zero_division=0)
per_class = [{'career': le_career.classes_[i],
              'precision': float(p[i]), 'recall': float(r[i]),
              'f1': float(f[i]), 'support': int(s[i])}
             for i in range(len(le_career.classes_))]

# Feature importance
importances = model.feature_importances_
feat_df = (pd.DataFrame({'feature': X.columns, 'importance': importances})
           .sort_values('importance', ascending=False).head(15))

# ── 6. Save artifacts ──────────────────────────────────────────────────────────
joblib.dump(model,    os.path.join(MODELS_DIR, 'career_predictor_model.pkl'))
joblib.dump(le_career,os.path.join(MODELS_DIR, 'label_encoder_career.pkl'))
joblib.dump(le_major, os.path.join(MODELS_DIR, 'label_encoder_major.pkl'))
joblib.dump(skills_df.columns.tolist(),
                      os.path.join(MODELS_DIR, 'skills_columns.pkl'))

feature_names = X.columns.tolist()
with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_names, f)

metrics = {
    'accuracy':         float(accuracy),
    'precision':        float(precision),
    'recall':           float(recall),
    'f1_score':         float(f1),
    'cv_mean':          float(cv_scores.mean()),
    'cv_std':           float(cv_scores.std()),
    'cv_scores':        cv_scores.tolist(),
    'total_samples':    int(len(df)),
    'training_samples': int(X_train.shape[0]),
    'test_samples':     int(X_test.shape[0]),
    'classes':          le_career.classes_.tolist(),
    'per_class':        per_class,
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'feature_importance': feat_df.to_dict('records'),
    'new_features': ['has_ai_keywords','has_web_keywords','has_data_keywords',
                     'has_security_kw','has_mgmt_keywords',
                     'internships_norm','projects_norm'],
}
with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'\nAll artifacts saved to {MODELS_DIR}/')
print('Features used:', feature_names)
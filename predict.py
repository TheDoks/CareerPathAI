import joblib, json, os
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MAJOR_ALIAS_MAP = {
    'Artificial Intelligence':      'Computer Science',
    'Software & Web Development':   'Computer Science',
    'Networking & Cloud Computing': 'Engineering',
    'Cybersecurity':                'Computer Science',
}

# Keyword sets for project description analysis
AI_KW   = {'machine learning','deep learning','neural','nlp','computer vision',
            'artificial intelligence','ai','model','classification','prediction',
            'tensorflow','pytorch','scikit','regression','dataset','training'}
WEB_KW  = {'web','website','html','css','javascript','react','vue','flask','django',
            'nextjs','frontend','backend','fullstack','api','node','php','laravel'}
DATA_KW = {'data','analytics','analysis','dashboard','visualization','sql','database',
            'pandas','numpy','excel','power bi','tableau','statistics','csv','chart'}
SEC_KW  = {'security','cybersecurity','network','firewall','encryption','penetration',
            'vulnerability','hacking','ethical','cloud','aws','azure','infrastructure'}
MGMT_KW = {'management','project','team','leadership','agile','scrum','planning',
            'strategy','coordination','product','roadmap','stakeholder','client'}

def extract_keyword_features(text: str) -> dict:
    """Extract 5 binary keyword flags from free-text project description."""
    t = text.lower()
    return {
        'has_ai_keywords':   int(any(k in t for k in AI_KW)),
        'has_web_keywords':  int(any(k in t for k in WEB_KW)),
        'has_data_keywords': int(any(k in t for k in DATA_KW)),
        'has_security_kw':   int(any(k in t for k in SEC_KW)),
        'has_mgmt_keywords': int(any(k in t for k in MGMT_KW)),
    }

def convert_gpa(value: float, scale: str = '4.0') -> float:
    """Use GPA exactly as provided — no conversion (polytechnic uses 4.0 scale)."""
    return round(float(value), 2)


class CareerPredictor:
    def __init__(self):
        self.model          = joblib.load(os.path.join(MODELS_DIR, 'career_predictor_model.pkl'))
        self.le_career      = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_career.pkl'))
        self.le_major       = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_major.pkl'))
        self.skills_columns = joblib.load(os.path.join(MODELS_DIR, 'skills_columns.pkl'))
        with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
            self.feature_names = json.load(f)

    def predict(self, major, gpa, gpa_scale, skills, internships,
                projects, leadership, project_desc=''):

        gpa_4 = convert_gpa(gpa, gpa_scale)

        features = pd.DataFrame(0, index=[0], columns=self.feature_names)

        features['gpa']               = gpa_4
        features['internships']       = int(internships)
        features['projects']          = int(projects)
        features['leadership']        = int(leadership)
        features['internships_norm']  = int(internships) / 15.0
        features['projects_norm']     = int(projects)    / 30.0

        # Keyword features from project description
        kw = extract_keyword_features(project_desc)
        for k, v in kw.items():
            if k in features.columns:
                features[k] = v

        # Also derive keyword features from selected skills (OR logic)
        skill_text = ' '.join(skills).lower()
        kw_from_skills = extract_keyword_features(skill_text)
        # Merge: either source can activate a signal
        kw_merged = {k: max(kw.get(k, 0), kw_from_skills.get(k, 0)) for k in kw}
        for k, v in kw_merged.items():
            if k in features.columns:
                features[k] = v

        # Major encoding
        resolved = MAJOR_ALIAS_MAP.get(major, major)
        if resolved in self.le_major.classes_:
            features['major_encoded'] = self.le_major.transform([resolved])[0]

        # Skill one-hot
        for skill in skills:
            skill = skill.strip()
            if skill in features.columns:
                features[skill] = 1

        features = features[self.feature_names]
        probs     = self.model.predict_proba(features)[0]
        top3_idx  = np.argsort(probs)[-3:][::-1]

        return {
            'results': [
                {
                    'career':     self.le_career.inverse_transform([idx])[0],
                    'confidence': round(float(probs[idx]) * 100, 1)
                }
                for idx in top3_idx
            ],
            'gpa_used':        gpa_4,
            'keyword_signals': kw_merged,  # merged from both description + skills
        }
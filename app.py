import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from predict import CareerPredictor, extract_keyword_features
import json, os, io, re
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename

# PDF / OCR
import PyPDF2
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)
app.secret_key = 'careerpath_secret_2026'

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

predictor = CareerPredictor()

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_history():
    path = os.path.join(MODELS_DIR, 'prediction_history.json')
    if not os.path.exists(path): return []
    with open(path) as f: return json.load(f)

def save_history(h):
    with open(os.path.join(MODELS_DIR, 'prediction_history.json'), 'w') as f:
        json.dump(h, f, indent=2)

def load_metrics():
    with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
        return json.load(f)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if session.get('logged_in'): return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if data.get('username') == 'admin' and data.get('password') == 'admin2026':
        session['logged_in'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Incorrect username or password'}), 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('app.html')

# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/stats')
@login_required
def api_stats():
    history = load_history()
    metrics = load_metrics()
    total   = len(history)
    avg_conf = round(sum(h['confidence'] for h in history) / total, 1) if total else 0
    career_counts = {}
    for h in history:
        career_counts[h['career']] = career_counts.get(h['career'], 0) + 1
    trend = {}
    for h in history:
        d = h['timestamp'][:10]
        trend[d] = trend.get(d, 0) + 1
    trend_sorted = sorted(trend.items())
    return jsonify({
        'total_predictions': total,
        'avg_confidence':    avg_conf,
        'unique_users':      len(set(h['user'] for h in history)) if history else 0,
        'model_accuracy':    round(metrics['accuracy'] * 100, 1),
        'career_distribution': career_counts,
        'trend_dates':  [t[0] for t in trend_sorted],
        'trend_counts': [t[1] for t in trend_sorted],
    })

@app.route('/api/history')
@login_required
def api_history():
    history = load_history()
    return jsonify(sorted(history, key=lambda x: x['timestamp'], reverse=True))

@app.route('/api/history/clear', methods=['POST'])
@login_required
def api_clear_history():
    save_history([])
    return jsonify({'success': True})

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data         = request.get_json()
    student_name = data.get('name', '').strip()
    if not student_name:
        return jsonify({'error': 'Student name is required'}), 400

    result = predictor.predict(
        major        = data.get('major', ''),
        gpa          = float(data.get('gpa', 3.0)),
        gpa_scale    = '4.0',
        skills       = data.get('skills', []),
        internships  = int(data.get('internships', 0)),
        projects     = int(data.get('projects', 0)),
        leadership   = int(data.get('leadership', 0)),
        project_desc = data.get('project_desc', ''),
    )

    top   = result['results'][0]
    entry = {
        'id':             len(load_history()) + 1,
        'user':           student_name,
        'major':          data.get('major', ''),
        'gpa':            float(data.get('gpa', 3.0)),
        'skills':         data.get('skills', []),
        'internships':    int(data.get('internships', 0)),
        'projects':       int(data.get('projects', 0)),
        'leadership':     int(data.get('leadership', 0)),
        'project_desc':   data.get('project_desc', ''),
        'career':         top['career'],
        'confidence':     top['confidence'],
        'all_results':    result['results'],
        'keyword_signals':result['keyword_signals'],
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    history = load_history()
    history.append(entry)
    save_history(history)
    return jsonify(entry)

@app.route('/api/extract-transcript', methods=['POST'])
@login_required
def extract_transcript():
    """Extract GPA, skills signals, and keywords from an uploaded transcript."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename).lower()
    extracted_text = ''

    try:
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            extracted_text = ' '.join(
                page.extract_text() or '' for page in reader.pages
            )
        elif OCR_AVAILABLE and filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(file.stream)
            extracted_text = pytesseract.image_to_string(img)
        else:
            return jsonify({'error': 'Unsupported file type for extraction'}), 400
    except Exception as e:
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500

    text = extracted_text.lower()

    # ── Try to extract GPA — use value directly (polytechnic uses 4.0 scale) ─
    gpa_found = None
    patterns = [
        r'cgpa[:\s]+(\d+\.\d+)',
        r'cumulative\s+gpa[:\s]+(\d+\.\d+)',
        r'grade\s+point[:\s]+(\d+\.\d+)',
        r'\bgpa[:\s]+(\d+\.\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = round(float(m.group(1)), 2)
            if 0 < val <= 4.0:
                gpa_found = val
                break

    # ── Extract keyword signals ───────────────────────────────────────────────
    kw_signals = extract_keyword_features(extracted_text)

    # ── Detect skill mentions ──────────────────────────────────────────────────
    skill_map = {
        'Python': ['python'], 'Java': ['java'], 'JavaScript': ['javascript','js'],
        'SQL': ['sql','mysql','postgresql'], 'Data Analysis': ['data analysis','analytics'],
        'Machine Learning': ['machine learning','ml '], 'Deep Learning': ['deep learning','neural'],
        'Statistics': ['statistics','statistical'], 'Communication': ['communication'],
        'Leadership': ['leadership','leader'], 'Research': ['research'],
        'Project Management': ['project management'], 'Cloud Computing': ['cloud','aws','azure','gcp'],
        'UI/UX Design': ['ui/ux','ux design','user interface'], 'C++': ['c++','cpp'],
        'MATLAB': ['matlab'], 'R': [' r ','r programming'], 'Figma': ['figma'],
    }
    detected_skills = [skill for skill, terms in skill_map.items()
                       if any(t in text for t in terms)]

    return jsonify({
        'success': True,
        'gpa': gpa_found,
        'detected_skills': detected_skills,
        'keyword_signals': kw_signals,
        'text_preview': extracted_text[:300].strip(),
    })


@login_required
def api_metrics():
    return jsonify(load_metrics())

# ── PDF Report ────────────────────────────────────────────────────────────────
@app.route('/api/report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    history = load_history()
    entry   = next((h for h in history if h.get('id') == prediction_id), None)
    if not entry:
        return jsonify({'error': 'Prediction not found'}), 404

    try:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=20*mm, bottomMargin=20*mm,
                                leftMargin=20*mm, rightMargin=20*mm)

        DARK  = colors.HexColor('#0f172a')
        BLUE  = colors.HexColor('#3b82f6')
        LIGHT = colors.HexColor('#e2e8f0')
        MUTED = colors.HexColor('#94a3b8')
        CARD  = colors.HexColor('#1e293b')
        SLATE = colors.HexColor('#334155')

        def sty(**kw):
            name = kw.pop('name', 's')
            return ParagraphStyle(name,
                fontName  = kw.pop('fontName', 'Helvetica'),
                fontSize  = kw.pop('fontSize', 10),
                textColor = kw.pop('textColor', LIGHT),
                leading   = kw.pop('leading', 14), **kw)

        def sec(text):
            story.append(Paragraph(
                f'<b><font size=8>{text}</font></b>', sty(name=f'sl_{text[:6]}', textColor=MUTED)))
            story.append(HRFlowable(width='100%', thickness=0.5, color=SLATE))
            story.append(Spacer(1, 2*mm))

        story = []

        # Header
        story.append(Table(
            [[Paragraph('<font size=22><b>CareerPath</b><font color="#3b82f6">AI</font></font>',
                        sty(name='h', fontSize=22, textColor=colors.white)),
              Paragraph('Career Prediction Report<br/>'
                        '<font size=8>Federal Polytechnic Offa · AI Department · HND Final Year Project</font>',
                        sty(name='sub', fontSize=10, textColor=MUTED))]],
            colWidths=[95*mm, 75*mm],
            style=TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), DARK),
                ('TOPPADDING',    (0,0),(-1,-1), 16),
                ('BOTTOMPADDING', (0,0),(-1,-1), 16),
                ('LEFTPADDING',   (0,0),(0,-1),  12),
                ('RIGHTPADDING',  (-1,0),(-1,-1), 8),
                ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
                ('ALIGN',         (1,0),(1,-1),  'RIGHT'),
            ])
        ))
        story.append(HRFlowable(width='100%', thickness=2, color=BLUE))
        story.append(Spacer(1, 6*mm))

        conf      = entry.get('confidence', 0)
        top_c     = entry.get('career', 'â')
        all_res   = entry.get('all_results', [{'career': top_c, 'confidence': conf}])
        gpa_val   = entry.get('gpa', 'â')
        conf_color = '#34d399' if conf >= 80 else '#3b82f6' if conf >= 65 else '#f59e0b'

        # Student profile
        sec('STUDENT PROFILE')
        prows = [
            ['Student Name',     str(entry.get('user', 'â'))],
            ['Major',            str(entry.get('major', 'â'))],
            ['GPA (4.0 Scale)',  str(gpa_val)],
            ['Internships',      str(entry.get('internships', 'â'))],
            ['Projects',         str(entry.get('projects', 'â'))],
            ['Leadership Roles', str(entry.get('leadership', 'â'))],
            ['Date',             str(entry.get('timestamp', 'â'))],
        ]
        story.append(Table(
            [[Paragraph(f'<b>{r[0]}</b>', sty(name=f'pk{i}', fontSize=9, textColor=MUTED)),
              Paragraph(str(r[1]),          sty(name=f'pv{i}', fontSize=9))]
             for i, r in enumerate(prows)],
            colWidths=[55*mm, 115*mm],
            style=TableStyle([
                ('ROWBACKGROUNDS', (0,0),(-1,-1), [DARK, CARD]),
                ('TOPPADDING',    (0,0),(-1,-1), 5),
                ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                ('LEFTPADDING',   (0,0),(-1,-1), 8),
                ('RIGHTPADDING',  (0,0),(-1,-1), 8),
            ])
        ))
        story.append(Spacer(1, 6*mm))

        # Prediction result
        sec('PREDICTION RESULT')
        story.append(Table(
            [[Paragraph(f'<b><font size=18>{top_c}</font></b>',
                        sty(name='tc', textColor=colors.white)),
              Paragraph(f'<font color="{conf_color}" size=22><b>{conf}%</b></font><br/>'
                        '<font size=8>Confidence Score</font>',
                        sty(name='cv', alignment=TA_CENTER))]],
            colWidths=[110*mm, 60*mm],
            style=TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), CARD),
                ('TOPPADDING',    (0,0),(-1,-1), 12),
                ('BOTTOMPADDING', (0,0),(-1,-1), 12),
                ('LEFTPADDING',   (0,0),(-1,-1), 12),
                ('RIGHTPADDING',  (0,0),(-1,-1), 8),
                ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
                ('ALIGN',         (1,0),(1,-1),  'CENTER'),
            ])
        ))
        story.append(Spacer(1, 5*mm))

        # Alternatives
        alts = all_res[1:]
        if alts:
            sec('ALTERNATIVE CAREER PATHS')
            story.append(Table(
                [[Paragraph(str(r.get('career', '')),
                            sty(name=f'an{i}', fontSize=10)),
                  Paragraph(f'<font color="#a78bfa"><b>{r.get("confidence", 0)}%</b></font>',
                            sty(name=f'ap{i}', fontSize=10))]
                 for i, r in enumerate(alts)],
                colWidths=[140*mm, 30*mm],
                style=TableStyle([
                    ('ROWBACKGROUNDS', (0,0),(-1,-1), [CARD, DARK]),
                    ('TOPPADDING',    (0,0),(-1,-1), 6),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 6),
                    ('LEFTPADDING',   (0,0),(-1,-1), 10),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 8),
                    ('ALIGN',         (1,0),(1,-1), 'RIGHT'),
                ])
            ))
            story.append(Spacer(1, 5*mm))

        # Skills — padded chunks of 4
        skills = entry.get('skills', [])
        if skills:
            sec('SKILLS ASSESSED')
            cs     = 4
            chunks = [skills[i:i+cs] for i in range(0, len(skills), cs)]
            last   = chunks[-1]
            if len(last) < cs:
                last += [''] * (cs - len(last))
            story.append(Table(
                [[Paragraph(f'• {s}' if s else '',
                            sty(name=f'sk{ri}{ci}', fontSize=9))
                  for ci, s in enumerate(chunk)]
                 for ri, chunk in enumerate(chunks)],
                colWidths=[42*mm] * 4,
                style=TableStyle([
                    ('BACKGROUND',    (0,0),(-1,-1), CARD),
                    ('TOPPADDING',    (0,0),(-1,-1), 5),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                    ('LEFTPADDING',   (0,0),(-1,-1), 8),
                ])
            ))
            story.append(Spacer(1, 5*mm))

        # Project description
        desc = entry.get('project_desc', '')
        if desc:
            sec('PROJECT DESCRIPTION')
            story.append(Table(
                [[Paragraph(desc, sty(name='dt', fontSize=9, leading=14))]],
                colWidths=[170*mm],
                style=TableStyle([
                    ('BACKGROUND',    (0,0),(-1,-1), CARD),
                    ('TOPPADDING',    (0,0),(-1,-1), 10),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 10),
                    ('LEFTPADDING',   (0,0),(-1,-1), 12),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                    ('LINEBEFORE',    (0,0),(0,-1),   3, BLUE),
                ])
            ))
            story.append(Spacer(1, 5*mm))

        # Keyword signals
        kw = entry.get('keyword_signals', {})
        if kw:
            KW_LABELS = {
                'has_ai_keywords':   'AI / Machine Learning',
                'has_web_keywords':  'Web Development',
                'has_data_keywords': 'Data / Analytics',
                'has_security_kw':   'Security / Networking',
                'has_mgmt_keywords': 'Management / Leadership',
            }
            sec('PROJECT KEYWORD SIGNALS')
            story.append(Table(
                [[Paragraph(KW_LABELS.get(k, k),
                            sty(name=f'kl{i}', fontSize=9)),
                  Paragraph('<font color="#34d399">Detected</font>' if v
                            else '<font color="#475569">Not detected</font>',
                            sty(name=f'kv{i}', fontSize=9))]
                 for i, (k, v) in enumerate(kw.items())],
                colWidths=[140*mm, 30*mm],
                style=TableStyle([
                    ('ROWBACKGROUNDS', (0,0),(-1,-1), [CARD, DARK]),
                    ('TOPPADDING',    (0,0),(-1,-1), 5),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                    ('LEFTPADDING',   (0,0),(-1,-1), 10),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 8),
                    ('ALIGN',         (1,0),(1,-1), 'RIGHT'),
                ])
            ))
            story.append(Spacer(1, 6*mm))

        # Footer
        story.append(HRFlowable(width='100%', thickness=0.5, color=SLATE))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            f'Generated by CareerPath AI  ·  Federal Polytechnic Offa  ·  '
            f'AI Department  ·  {datetime.now().strftime("%d %B %Y, %I:%M %p")}',
            sty(name='ft', fontSize=8, textColor=MUTED, alignment=TA_CENTER)
        ))

        doc.build(story)
        buf.seek(0)
        safe = ''.join(c for c in entry.get('user', 'report')
                       if c.isalnum() or c in ' _').replace(' ', '_')
        return send_file(buf, mimetype='application/pdf',
                         download_name=f'CareerPath_Report_{safe}_{entry["id"]}.pdf',
                         as_attachment=True)

    except Exception as e:
        import traceback
        print(f'PDF error: {traceback.format_exc()}')
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

    
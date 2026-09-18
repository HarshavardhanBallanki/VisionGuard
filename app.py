import os
import threading
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, send_file, session)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from detector import run_detection, DEFAULT_CONFIDENCE

app = Flask(__name__)
app.secret_key = 'visiongaurd-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
REPORT_FOLDER = os.path.join('static', 'reports')

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
IMAGE_EXTENSIONS   = {'jpg', 'jpeg', 'png'}

job_state = {
    'status': 'idle',
    'step': 0,
    'filename': '',
    'results': {},
    'error_message': '',
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS


def parse_threshold(form):
    try:
        val = float(form.get('threshold', DEFAULT_CONFIDENCE))
        return max(0.1, min(0.95, val))
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE


# ── LANDING + UPLOAD ROUTES ───────────────────────────────────────────────────

@app.route('/')
def intro():
    return render_template('intro.html')


@app.route('/home')
def index():
    return render_template('index.html', default_threshold=DEFAULT_CONFIDENCE)

# ─────────────────────────────────────────────────────────────────────────────


@app.route('/upload', methods=['POST'])
def upload():
    files     = request.files.getlist('file')
    threshold = parse_threshold(request.form)

    if not files or files[0].filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    for f in files:
        if not allowed_file(f.filename):
            flash(f'Unsupported file type: {f.filename}')
            return redirect(url_for('index'))

    # Single file: handle videos and images through the same direct path
    if len(files) == 1:
        file     = files[0]
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        session['filename']  = filename
        session['multi']     = False
        session['threshold'] = threshold
        job_state.update({'status': 'processing', 'step': 1,
                          'filename': filename, 'results': {}, 'error_message': ''})
        thread = threading.Thread(target=run_job, args=(filename, threshold))
        thread.daemon = True
        thread.start()
        return redirect(url_for('processing'))

    # One or more images
    if len(files) > 1:
        videos = [f for f in files if not is_image(f.filename)]
        if videos:
            flash('Please upload only one video at a time. Multiple images are fine.')
            return redirect(url_for('index'))

    filenames = []
    for f in files:
        filename = secure_filename(f.filename)
        f.save(os.path.join(UPLOAD_FOLDER, filename))
        filenames.append(filename)

    session['filenames'] = filenames
    session['multi']     = True
    session['threshold'] = threshold
    job_state.update({'status': 'processing', 'step': 1,
                      'filename': f"{len(filenames)} image(s)", 'results': {}, 'error_message': ''})
    thread = threading.Thread(target=run_job_multi, args=(filenames, threshold))
    thread.daemon = True
    thread.start()
    return redirect(url_for('processing'))


def run_job(filename, threshold=DEFAULT_CONFIDENCE):
    try:
        base        = os.path.splitext(filename)[0]
        ext         = filename.rsplit('.', 1)[1].lower()
        output_name = ('output_' + filename
                       if ext in IMAGE_EXTENSIONS
                       else 'output_' + base + '.mp4')
        report_name = 'report_' + base + '.csv'

        results = run_detection(
            input_path           = os.path.join(UPLOAD_FOLDER, filename),
            output_path          = os.path.join(OUTPUT_FOLDER, output_name),
            report_path          = os.path.join(REPORT_FOLDER, report_name),
            job_state            = job_state,
            confidence_threshold = threshold,
        )

        job_state.update({
            'status':       'done',
            'step':         5,
            'results':      results,
            'output_name':  output_name,
            'output_files': [output_name],
            'report_name':  report_name,
        })
    except Exception as e:
        print(f'Detection error: {e}')
        job_state.update({
            'status': 'error',
            'error_message': str(e) or 'Unknown detection error.',
        })


def run_job_multi(filenames, threshold=DEFAULT_CONFIDENCE):
    try:
        import csv as _csv
        all_violations = []
        output_files   = []
        total_vehicles = 0
        total_riders   = 0

        for filename in filenames:
            base        = os.path.splitext(filename)[0]
            output_name = 'output_' + filename
            report_name = 'report_' + base + '.csv'

            results = run_detection(
                input_path           = os.path.join(UPLOAD_FOLDER, filename),
                output_path          = os.path.join(OUTPUT_FOLDER, output_name),
                report_path          = os.path.join(REPORT_FOLDER, report_name),
                job_state            = job_state,
                confidence_threshold = threshold,
            )

            for v in results.get('violations', []):
                v['source'] = filename
                all_violations.append(v)

            total_riders += results.get('total_riders', 0)
            total_vehicles += results.get('total_vehicles', 0)
            output_files.append(output_name)

        combined_report = 'report_multi.csv'
        with open(os.path.join(REPORT_FOLDER, combined_report), 'w', newline='') as f:
            writer = _csv.DictWriter(
                f, fieldnames=['source', 'frame', 'timestamp', 'type', 'confidence', 'plate']
            )
            writer.writeheader()
            safe_rows = [
                {k: v.get(k, '') for k in ['source','frame','timestamp','type','confidence','plate']}
                for v in all_violations
            ]
            writer.writerows(safe_rows)

        v_count = len(all_violations)
        compliance = round((total_riders - v_count) / total_riders * 100, 1) if total_riders > 0 else 100.0
        job_state.update({
            'status':       'done',
            'step':         5,
            'output_name':  output_files[0] if output_files else '',
            'output_files': output_files,
            'report_name':  combined_report,
            'results': {
                'total_frames':    len(filenames),
                'total_vehicles':  total_vehicles,
                'total_riders':    total_riders,
                'violation_count': v_count,
                'violation_rate':  round(v_count / len(filenames) * 100, 1),
                'compliance_rate': compliance,
                'violations':      all_violations,
                'by_minute':       [],
            }
        })
    except Exception as e:
        print(f'Multi detection error: {e}')
        job_state.update({
            'status': 'error',
            'error_message': str(e) or 'Unknown detection error.',
        })


@app.route('/processing')
def processing():
    filename = session.get('filename')
    if not filename:
        filenames = session.get('filenames', [])
        filename = ', '.join(filenames[:2])
        if len(filenames) > 2:
            filename += f' and {len(filenames) - 2} more'
    return render_template('processing.html', filename=filename or '')


@app.route('/status')
def status():
    return jsonify({
        'status': job_state['status'],
        'step': job_state['step'],
        'error_message': job_state.get('error_message', ''),
    })


@app.route('/results')
def results():
    if job_state['status'] != 'done':
        return redirect(url_for('index'))

    r            = job_state['results']
    output_name  = job_state.get('output_name', '')
    output_files = job_state.get('output_files', [output_name])

    return render_template('results.html',
        filename        = job_state.get('filename', ''),
        total_frames    = r.get('total_frames', 0),
        total_vehicles  = r.get('total_vehicles', 0),
        total_riders    = r.get('total_riders', 0),
        violation_count = r.get('violation_count', 0),
        violation_rate  = r.get('violation_rate', 0.0),
        compliance_rate = r.get('compliance_rate', 100.0),
        output_file     = output_name,
        output_files    = output_files,
        file_is_image   = is_image(output_name),
        violations      = r.get('violations', []),
        by_minute       = r.get('by_minute', []),
        heatmap_file    = None,
    )


@app.route('/download/video')
def download_video():
    path = os.path.join(OUTPUT_FOLDER, job_state.get('output_name', ''))
    return send_file(path, as_attachment=True)


@app.route('/stream/video/<filename>')
def stream_video(filename):
    """Stream video for in-browser playback with HTTP range request support (seeking)."""
    filename  = secure_filename(filename)
    path      = os.path.join(OUTPUT_FOLDER, filename)

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    file_size    = os.path.getsize(path)
    range_header = request.headers.get('Range', None)

    ext  = filename.rsplit('.', 1)[-1].lower()
    mime = {'mp4': 'video/mp4', 'avi': 'video/x-msvideo',
            'mov': 'video/quicktime', 'mkv': 'video/x-matroska'}.get(ext, 'video/mp4')

    if not range_header:
        response = send_file(path, mimetype=mime)
        response.headers['Accept-Ranges'] = 'bytes'
        return response

    byte_start, byte_end = 0, file_size - 1
    parts = range_header.replace('bytes=', '').split('-')
    if parts[0]:
        byte_start = int(parts[0])
    if len(parts) > 1 and parts[1]:
        byte_end = int(parts[1])

    length = byte_end - byte_start + 1
    with open(path, 'rb') as f:
        f.seek(byte_start)
        data = f.read(length)

    response = app.response_class(data, status=206, mimetype=mime, direct_passthrough=True)
    response.headers['Content-Range']  = f'bytes {byte_start}-{byte_end}/{file_size}'
    response.headers['Accept-Ranges']  = 'bytes'
    response.headers['Content-Length'] = str(length)
    return response


@app.route('/download/report')
def download_report():
    path = os.path.join(REPORT_FOLDER, job_state.get('report_name', ''))
    return send_file(path, as_attachment=True)


# ── VIOLATION STORE ──────────────────────────────────────────────────────────
import random as _random
from datetime import datetime as _dt

INDIAN_NAMES = [
    "Aarav Sharma","Vivaan Patel","Aditya Reddy","Vihaan Nair",
    "Arjun Mehta","Sai Krishna","Rohan Verma","Karthik Iyer",
    "Priya Sundar","Ananya Bose","Divya Pillai","Meera Joshi",
    "Kavya Rao","Sneha Kulkarni","Pooja Tiwari","Lakshmi Menon",
    "Ravi Shankar","Suresh Babu","Vijay Kumar","Ramesh Gupta",
    "Harsha Vardhan","Deepak Nair","Ajay Singh","Rahul Mishra",
    "Amit Yadav","Nikhil Desai","Sachin More","Vishal Patil",
]

FINE_AMOUNT = 1000

# plate → {name, balance, originalBalance}
plate_owners: dict = {}
# list of violation dicts for the dashboard
violation_log: list = []
_vid_counter = 0


def _get_or_create_owner(plate: str) -> dict:
    """Return existing owner for plate, or create one with a random name & balance."""
    if plate not in plate_owners:
        plate_owners[plate] = {
            "name": _random.choice(INDIAN_NAMES),
            "plate": plate,
            "balance": _random.randint(3000, 25000),
            "originalBalance": 0,  # set below
        }
        plate_owners[plate]["originalBalance"] = plate_owners[plate]["balance"]
    return plate_owners[plate]


# ── NEW ROUTES ────────────────────────────────────────────────────────────────

@app.route('/api/violation', methods=['POST'])
def api_violation():
    global _vid_counter
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid or missing JSON body.'}), 400

    plate = str(data.get('plate', 'UNKNOWN')).strip().upper()
    owner = _get_or_create_owner(plate)

    prev    = owner['balance']
    nxt     = max(0, prev - FINE_AMOUNT)
    owner['balance'] = nxt
    _vid_counter += 1

    violation_log.append({
        'id':    'v' + str(_vid_counter),
        'plate': plate,
        'name':  owner['name'],
        'fine':  FINE_AMOUNT,
        'prev':  prev,
        'next':  nxt,
        'time':  _dt.now().strftime('%H:%M:%S'),
    })

    return jsonify({'success': True, 'plate': plate, 'owner': owner['name'],
                    'fine': FINE_AMOUNT, 'new_balance': nxt}), 201


@app.route('/api/violations', methods=['GET'])
def api_violations():
    """Return all violations + owner table for the wallet dashboard."""
    return jsonify({
        'violations': violation_log,
        'owners':     list(plate_owners.values()),
    })


@app.route('/wallet_dashboard')
def wallet_dashboard():
    return render_template('wallet_dashboard.html')

# ─────────────────────────────────────────────────────────────────────────────


@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    flash('File too large. Maximum is 200 MB.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
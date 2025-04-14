import os
import yaml
from flask import Flask, render_template, request, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from predict import predict, predict_from_video
import traceback
from flask_socketio import SocketIO

# Flask App Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['EXAMPLES_FOLDER'] = 'static/examples/'
app.secret_key = 'your_secret_key'

socketio = SocketIO(app, cors_allowed_origins="*")

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXAMPLES_FOLDER'], exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Retrieve form data
    example_choice = request.form.get('example')  # Selected example
    recognizer_choice = request.form.get('recognizer')  # Selected recognizer
    frame_interval = int(request.form.get('frameInterval', 1))
    print(f"Selected Example: {example_choice}, Selected Recognizer: {recognizer_choice}, Frame Interval: {frame_interval}")

    # Validate recognizer selection
    if not recognizer_choice:
        flash('Please choose a recognizer.', 'danger')
        return jsonify({'error': 'Please choose a recognizer.'}), 400

    # Process uploaded file or selected example
    file = request.files.get('file')
    filename = None
    file_path = None
    file_url = None
    is_video = False  # Flag to check if the file is a video

    # Define allowed video formats
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

    if file and file.filename.strip():
        file_ext = os.path.splitext(file.filename)[1].lower()  # Get file extension
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_url = url_for('static', filename=f'uploads/{filename}')

            # Check if the uploaded file is a video
            if file_ext in VIDEO_EXTENSIONS:
                is_video = True
        else:
            flash('Unsupported file type. Please upload an image or video.', 'danger')
            return jsonify({'error': 'Unsupported file type. Please upload an image or video.'}), 400
    elif example_choice:
        filename = example_choice
        file_path = os.path.join(app.config['EXAMPLES_FOLDER'], filename)
        file_url = url_for('static', filename=f'examples/{filename}')
        
        # Check if example file is a video
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in VIDEO_EXTENSIONS:
            is_video = True
    else:
        flash('Please upload a file or choose an example.', 'danger')
        return jsonify({'error': 'Please upload a file or choose an example.'}), 400

    # Load the configuration
    try:
        with open('./config/config.yaml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        flash('Configuration file not found.', 'danger')
        return jsonify({'error': 'Configuration file not found.'}), 400

    config["data_path"] = file_path
    config["recognizer"]["type"] = recognizer_choice

    def progress_callback(progress):
        # Emit progress updates to the client
        socketio.emit('progress', {'progress': progress})

    # Perform inference
    try:
        if is_video:
            config["frame_interval"] = frame_interval
            results = predict_from_video(config, progress_callback=progress_callback)
            flash(f'Recognizer: {recognizer_choice}. Video inference completed.', 'success')
        else:
            results = predict(config)
            flash(f'Recognizer: {recognizer_choice}. Image inference completed.', 'success')
    except Exception as e:
        flash(f'Error during inference: {str(e)}', 'danger')
        traceback.print_exc()
        return jsonify({'error': f'Error during inference: {str(e)}'}), 500

    # After performing inference
    data = {
        "file_url": file_url,
        "filename": filename,
        "results": results,
    }
    print("DATA: ", data)
    return jsonify(data), 200

# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
    print("Flask app started on http://0.0.0.0:8080")
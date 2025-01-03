import os
import yaml
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from main import predict  # Function for license plate recognition
import traceback

# Flask App Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['EXAMPLES_FOLDER'] = 'static/examples/'
app.secret_key = 'your_secret_key'

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXAMPLES_FOLDER'], exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for file upload and inference
@app.route('/upload', methods=['POST'])
def upload_file():
    # Retrieve form data
    example_choice = request.form.get('example')  # Selected example
    recognizer_choice = request.form.get('recognizer')  # Selected recognizer

    # Validate recognizer selection
    if not recognizer_choice:
        flash('Please choose a recognizer.', 'danger')
        return redirect(url_for('index'))

    # Process uploaded file or selected example
    file = request.files.get('file')
    filename = None
    file_path = None
    file_url = None

    if file and file.filename.strip():
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_url = url_for('static', filename=f'uploads/{filename}')
        else:
            flash('Unsupported file type. Please upload an image or video.', 'danger')
            return redirect(url_for('index'))
    elif example_choice:
        filename = example_choice
        file_path = os.path.join(app.config['EXAMPLES_FOLDER'], filename)
        file_url = url_for('static', filename=f'examples/{filename}')
    else:
        flash('Please upload a file or choose an example.', 'danger')
        return redirect(url_for('index'))
    

    # Load the configuration
    try:
        with open('./config/config.yaml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        flash('Configuration file not found.', 'danger')
        return redirect(url_for('index'))
    # Update configuration with file path and recognizer choice
    config["data_path"]= file_path
    config["recognizer"]["type"] = recognizer_choice

    # Perform inference
    try:
        results = predict(config)
        flash(f'Recognizer: {recognizer_choice}. Inference completed.', 'success')
    except Exception as e:
        flash(f'Error during inference: {str(e)}', 'danger')
        traceback.print_exc()
        return redirect(url_for('index'))
    

    # Render the results page
    return render_template(
        'result.html',
        file_url=file_url,
        filename=filename,
        results=results
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
    print("Flask app started on http://0.0.0.0:8080")

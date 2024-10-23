from flask import Flask, request, render_template, send_from_directory, jsonify, url_for
import os
import numpy as np
from skimage import color as skcolor, io  # Aliased to avoid shadowing
from werkzeug.utils import secure_filename
from app.paint_by_numbers.segmentation import perform_slic_segmentation
from app.paint_by_numbers.color_reduction import reduce_colors_kmeans
from app.paint_by_numbers.labeling import create_labeled_image
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'debug'), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get parameters from form
        compactness = float(request.form.get('compactness', 10.0))
        segments = int(request.form.get('segments', 1000))
        sigma = float(request.form.get('sigma', 3.0))
        colors = int(request.form.get('colors', 16))
        border = request.form.get('border', '50,50,50')

        try:
            # Process the image
            image = io.imread(filepath)

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Process image
            slic_label_images = perform_slic_segmentation([image], compactness, segments, sigma)

            # Create simplified image
            simpler_image = skcolor.label2rgb(slic_label_images[0], [image][0], kind='avg', bg_label=-1)
            simpler_image = (simpler_image).astype(np.uint8)

            # Reduce colors
            kmeans_image, reduced_colors, kmeans_labels = reduce_colors_kmeans(simpler_image, colors)

            # Save reduced color image
            reduced_color_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug', f'reduced_color_image_{filename}')
            io.imsave(reduced_color_image_path, kmeans_image)

            # Create labeled images
            border_color = [int(x) for x in border.split(',')]
            param_str = f"c{compactness}_s{segments}_sig{sigma}_col{colors}"

            create_labeled_image(
                kmeans_image,
                slic_label_images[0],
                kmeans_labels,
                border_color,
                os.path.join(app.config['UPLOAD_FOLDER'], 'debug'),
                param_str
            )

            # Return paths to generated images
            result = {
                'original': url_for('static', filename=f'uploads/{filename}'),
                'reduced_color': url_for('static', filename=f'uploads/debug/reduced_color_image_{filename}'),
                'labeled': url_for('static', filename=f'uploads/debug/labeled_image_{param_str}.png'),
                'labeled_no_color': url_for('static', filename=f'uploads/debug/labeled_image_no_color_{param_str}.png'),
                'download_reduced_color': url_for('download_file', filename=f'reduced_color_image_{filename}'),
                'download_labeled': url_for('download_file', filename=f'labeled_image_{param_str}.png'),
                'download_labeled_no_color': url_for('download_file', filename=f'labeled_image_no_color_{param_str}.png')
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    # Using an absolute path here to make sure we are looking in the right place
    directory = os.path.abspath(os.path.join('app', 'static', 'uploads', 'debug'))
    return send_from_directory(directory, filename)
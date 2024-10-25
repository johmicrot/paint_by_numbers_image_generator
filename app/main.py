from flask import Flask, request, render_template, send_from_directory, jsonify, url_for
import os
import numpy as np
from scipy.ndimage import zoom
from werkzeug.utils import secure_filename
from skimage import color as skcolor, io, segmentation, morphology, color
from scipy.stats import mode

from app.paint_by_numbers.segmentor import perform_rag_merging
from app.paint_by_numbers.color_reduction import reduce_colors_kmeans
from app.paint_by_numbers.labeling import create_labeled_image
from app.paint_by_numbers.palette import create_color_palette
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

def map_kmeans_to_slic(slic_label_image, kmeans_labels):
    """
    Maps each SLIC segment to its most frequent KMeans color label.
    """
    slic_segments = np.unique(slic_label_image)
    slic_to_kmeans = {}

    # Verify that kmeans_labels can be reshaped to the slic_label_image's shape
    if kmeans_labels.size != slic_label_image.size:
        raise ValueError(
            f"Size mismatch: kmeans_labels has {kmeans_labels.size} elements, "
            f"but slic_label_image has {slic_label_image.size} elements."
        )

    # Reshape kmeans_labels to 2D to match slic_label_image
    kmeans_labels_2d = kmeans_labels.reshape(slic_label_image.shape)

    for segment in slic_segments:
        mask = slic_label_image == segment
        if np.any(mask):
            mode_result = mode(kmeans_labels_2d[mask], axis=None)
            slic_to_kmeans[segment] = int(mode_result.mode)
        else:
            slic_to_kmeans[segment] = -1  # Handle empty segments

    return slic_to_kmeans



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
        print('xxxxxxxxxxxxxxxxxx')
        print(filename)
        print('xxxxxxxxxxxxxxxxxx')


        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get parameters from form
        compactness = float(request.form.get('compactness', 10.0))
        segments = int(request.form.get('segments', 1000))
        sigma = float(request.form.get('sigma', 3.0))
        colors = int(request.form.get('colors', 16))
        border = request.form.get('border', '50,50,50')
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_images')
        try:
            # Process the image
            image = io.imread(filepath)
            # New size for higher resolution (width, height)
            original_size = image.shape[:2]
            # image = zoom(image, (2, 2, 1), order=3)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Process image
            slic_labeled_image = segmentation.slic(image,
                                                   compactness=compactness,
                                                   n_segments=segments,
                                                   start_label=1,
                                                   sigma=sigma)
            # Create simplified image
            simpler_image = skcolor.label2rgb(slic_labeled_image, image, kind='avg', bg_label=-1)
            # simpler_image = simpler_image.astype(np.uint8)

            # Reduce colors
            kmeans_image, reduced_colors, kmeans_labels = reduce_colors_kmeans(simpler_image, colors)

            # Save reduced color image
            reduced_color_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug', f'reduced_color_image.png')
            io.imsave(reduced_color_image_path, kmeans_image)

            # Create labeled images
            border_color = [int(x) for x in border.split(',')]
            param_str = f"c{compactness}_s{segments}_sig{sigma}_col{colors}"

            # Calculate the percentage of each color in the image
            total_pixels = kmeans_labels.size
            unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
            percentages = counts / total_pixels * 100

            # Create a list of color percentages
            color_percentages = []
            for label, count, percentage in zip(unique_labels, counts, percentages):
                color_center = reduced_colors[label].astype(int).tolist()
                color_percentages.append({
                    'color_label': int(label),
                    'color_rgb': color_center,
                    'count': int(count),
                    'percentage': percentage
                })

            # Create the color palette with percentages
            create_color_palette(reduced_colors, color_percentages, save_dir, param_str)

            # Map KMeans labels to SLIC segments
            slic_to_kmeans = map_kmeans_to_slic(slic_labeled_image, kmeans_labels)

            # Perform RAG merging based on KMeans labels and enhanced criteria
            merged_label_image = perform_rag_merging(
                image,
                slic_labeled_image,
                slic_to_kmeans,
                save_dir
            )
            if merged_label_image is None:
                logging.error("No merged labels returned from perform_rag_merging.")
                exit(1)
            create_labeled_image(
                kmeans_image,
                merged_label_image,
                kmeans_labels,
                border_color,
                save_dir
            )

            # Return paths to generated images
            # os.path.join(app.config['UPLOAD_FOLDER'], 'generated_images')
            result = {
                'original': url_for('static', filename=f'uploads/{filename}'),
                'reduced_color': url_for('static', filename=f'uploads/debug/reduced_color_image.png'),
                'labeled': url_for('static', filename=f'uploads/generated_images/labeled_image.png'),
                'labeled_no_color': url_for('static', filename=f'uploads/generated_images/labeled_image_no_color.png'),
                'download_reduced_color': url_for('download_file', filename=f'reduced_color_image.png'),
                'download_labeled': url_for('download_file', filename= f'labeled_image.png'),
                'download_labeled_no_color': url_for('download_file', filename=f'labeled_image_no_color.png')
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    # Using an absolute path here to make sure we are looking in the right place
    directory = os.path.abspath(os.path.join('app', 'static', 'uploads', 'generated_images'))
    return send_from_directory(directory, filename)
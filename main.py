# main.py
#  original source was from https://github.com/LukaZdr/paint_by_numbers_image_generator
# To do, implement multithredding/parallel processing, docker containers, a web script
# calculate the ammount of paint needed assuming an image size
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color as skcolor , io # Aliased to avoid shadowing
import argparse  # Import argparse for command-line argument parsing


from segmentation import perform_slic_segmentation  # Removed perform_rag_merging
from color_reduction import reduce_colors_kmeans
from palette import create_color_palette
from labeling import create_labeled_image
dotsperinch= 600

def format_param(value):
    """
    Formats a parameter value into a string suitable for filenames.

    - Floats: Replace '.' with 'p'
    - Lists/Arrays: Join elements with '-'
    - Others: Convert to string directly
    """
    if isinstance(value, float):
        return str(value).replace('.', 'p')
    elif isinstance(value, list) or isinstance(value, np.ndarray):
        return '-'.join(map(str, value))
    else:
        return str(value)


def parse_arguments():
    # c30, s10 - weirdly blocky
    # c10, s1 - seems to be good
    parser = argparse.ArgumentParser(description='Paint-by-Numbers Image Generator')

    parser.add_argument('--input', type=str, default='input.jpg',
                        help='Path to the input image (default: input.jpg)')

    parser.add_argument('--compactness', type=float, default=10.0,
                        help='Controls the tradeoff between color proximity and space proximity in SLIC (default: 10.0)'
                             '10 seems to be a good value.  100 it gets blocky too blurry')

    parser.add_argument('--segments', type=int, default=1000,
                        help='Approximate number of segments to divide the image into using SLIC (default: 1000)')

    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Standard deviation for Gaussian smoothing before segmentation (default: 3.0)'
                             '10 is too high, it makes things have werid lines, like some bad ai generation')

    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Threshold for region merging in the RAG (exact color matching if set to 0.0) (default: 0.0)')

    parser.add_argument('--colors', type=int, default=16,
                        help='Number of colors to reduce the image to using KMeans clustering (default: 16)')

    parser.add_argument('--border', type=str, default='50,50,50',
                        help='RGB color for borders between segments, e.g., "50,50,50" (default: "50,50,50")')

    args = parser.parse_args()
    return args


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Extract parameters
    input_image_path = args.input
    compactness = args.compactness
    segment_count = args.segments
    sigma = args.sigma
    threshold = args.threshold  # Note: Threshold is no longer used since RAG merging is removed
    color_count = args.colors

    # Process border_color argument
    try:
        border_color = np.array([int(c) for c in args.border.split(',')], dtype=np.uint8)
        if border_color.size != 3:
            raise ValueError
    except ValueError:
        print("Error: --border must be a comma-separated string of three integers, e.g., '50,50,50'")
        exit(1)

    # Create debug folder with timestamp
    debug_folder = 'debug_images'
    os.makedirs(debug_folder, exist_ok=True)

    # Load the image
    image = io.imread(input_image_path)

    # Ensure the image is in uint8 format for exact color matching
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    image_list = [image]

    # Create a parameter string
    param_str = f"c{format_param(compactness)}_s{format_param(segment_count)}_sig{format_param(sigma)}_t{format_param(threshold)}_col{format_param(color_count)}_bord{format_param(border_color)}"

    # Segment the image using SLIC
    slic_label_images = perform_slic_segmentation(image_list, compactness, segment_count, sigma)

    # Save the initial segmentation result
    slic_image = skcolor.label2rgb(slic_label_images[0], image_list[0], kind='avg', bg_label=-1)
    slic_image = (slic_image ).astype(np.uint8)  # Ensure correct scaling
    slic_image_path = os.path.join(debug_folder, f'slic_image_{param_str}.png')
    plt.imsave(slic_image_path, slic_image, dpi=dotsperinch)

    # Simplify the image by averaging colors within SLIC segments
    simpler_images = []
    for index, slic_label_image in enumerate(slic_label_images):
        simpler_image = skcolor.label2rgb(slic_label_image, image_list[index], kind='avg', bg_label=-1)
        simpler_images.append((simpler_image).astype(np.uint8))  # Ensure correct scaling

    # Save the simplified image
    simplified_image_path = os.path.join(debug_folder, f'simplified_image_{param_str}.png')
    plt.imsave(simplified_image_path, simpler_images[0], dpi=dotsperinch)

    # Apply KMeans clustering to reduce the number of colors
    kmeans_image, reduced_colors, kmeans_labels = reduce_colors_kmeans(simpler_images[0], color_count)

    # Calculate the percentage of each color in the image
    total_pixels = kmeans_labels.size
    unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
    percentages = counts / total_pixels * 100

    # Create a list of color percentages
    color_percentages = []
    for label, count, percentage in zip(unique_labels, counts, percentages):
        color_center = reduced_colors[label].astype(int).tolist()  # Convert color center to list
        color_percentages.append({
            'color_label': int(label),
            'color_rgb': color_center,
            'count': int(count),
            'percentage': percentage
        })

    # Print the percentages to the console
    print("Color Percentages:")
    for cp in color_percentages:
        print(f"Label {cp['color_label']}: RGB {cp['color_rgb']} - {cp['percentage']:.2f}%")

    # Save the KMeans color-reduced image
    kmeans_image_path = os.path.join(debug_folder, f'kmeans_image_{param_str}.png')
    plt.imsave(kmeans_image_path, kmeans_image, dpi=dotsperinch)

    # Create the color palette with percentages
    create_color_palette(reduced_colors, color_percentages, debug_folder, param_str)

    # Create labeled images
    create_labeled_image(
        kmeans_image,
        slic_label_images[0],  # Use SLIC labels instead of RAG labels
        # reduced_colors,
        kmeans_labels,
        border_color,
        debug_folder,
        param_str
    )

    print(f"Processing complete. Debug files saved in '{debug_folder}'.")


if __name__ == "__main__":
    main()

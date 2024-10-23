import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from collections import Counter
import logging


def get_global_position(prop, local_pos):
    """
    Convert local mask coordinates to global image coordinates.

    Args:
        prop: Region properties object.
        local_pos: Tuple of local coordinates (row, col).

    Returns:
        Tuple of global coordinates (x, y).
    """
    global_y = prop.bbox[0] + local_pos[0]
    global_x = prop.bbox[1] + local_pos[1]
    return int(global_x), int(global_y)


def scale_position(x, y, scale_factor, image_size):
    """
    Scale the position according to the scale factor and ensure it's within bounds.

    Args:
        x: X-coordinate.
        y: Y-coordinate.
        scale_factor: Scaling factor.
        image_size: Tuple of image dimensions (width, height).

    Returns:
        Tuple of scaled coordinates (scaled_x, scaled_y).
    """
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    scaled_x = min(max(scaled_x, 0), image_size[0] - 1)
    scaled_y = min(max(scaled_y, 0), image_size[1] - 1)
    return scaled_x, scaled_y

def load_font(font_path, size):
    """
    Load a TrueType font or fallback to default.

    Args:
        font_path: Path to the font file.
        size: Font size.

    Returns:
        Loaded font object.
    """
    try:
        return ImageFont.truetype(font_path, size=size)
    except IOError:
        logging.warning(f"Font '{font_path}' not found. Using default font.")
        return ImageFont.load_default()


def get_text_size(text, font):
    """
    Get the size of the text when rendered with the specified font.

    Args:
        text: The text string.
        font: The font object.

    Returns:
        Tuple of (text_width, text_height).
    """
    try:
        # Try using getsize (might not be available)
        return font.getsize(text)
    except AttributeError:
        try:
            # Try using getbbox (newer Pillow versions)
            bbox = font.getbbox(text)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except AttributeError:
            try:
                # Try using getmask (older Pillow versions)
                mask = font.getmask(text)
                return mask.size
            except AttributeError:
                # Fallback to a default size estimation
                logging.warning("Unable to calculate text size accurately. Using default size.")
                average_char_width = font.size * 0.6  # Approximate average character width
                return (int(average_char_width * len(text)), font.size)


# --------------------------------------------------------------------------------------------------
# Main Functions
# --------------------------------------------------------------------------------------------------

def draw_labels_on_image(image, region_props_dict, label_map, scale_factor, font_size_range, image_size):
    """
    Draw labels on the image at the center of each region.

    Args:
        image: PIL Image object to draw on.
        region_props_dict: Dictionary mapping region labels to region properties.
        label_map: Dictionary mapping region labels to kmeans labels.
        scale_factor: Scaling factor for the image.
        font_size_range: Tuple of (min_font_size, max_font_size).
        image_size: Tuple of image dimensions (width, height).

    Returns:
        Image with labels drawn.
    """
    draw = ImageDraw.Draw(image)
    min_font_size, max_font_size = font_size_range

    # Cache for fonts to optimize performance
    font_cache = {}

    # Track existing label positions to avoid overlaps
    existing_label_positions = []

    for idx, (region_label, kmeans_label) in enumerate(label_map.items()):
        prop = region_props_dict.get(region_label)
        if prop is None:
            logging.warning(f"Index {idx}: Region {region_label} not found in properties.")
            continue  # Skip if region properties are not found

        # Get the mask of the current region (local coordinates)
        mask = prop.image

        # Check if the mask is valid
        if np.sum(mask) == 0:
            logging.warning(f"Index {idx}: Empty mask for region {region_label}. Skipping.")
            continue  # Skip empty masks

        # --------------------------------------------------------------------------------------------------
        # Distance Transform with Padding
        # --------------------------------------------------------------------------------------------------

        # Pad the mask with a border of zeros to handle regions touching the image borders.
        # This fixes an issue when calculating the location for the label within the shape.
        # if the shape edge was touching the edge of the local region it would think the label
        # should go at the border of the shape
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

        # Compute the distance transform on the padded mask
        distance = distance_transform_edt(padded_mask)

        # Find the coordinates of the maximum distance in the padded mask
        padded_local_max_pos = np.unravel_index(np.argmax(distance), distance.shape)

        # Adjust coordinates by subtracting the padding
        local_max_pos = (padded_local_max_pos[0] - 1, padded_local_max_pos[1] - 1)

        # Ensure coordinates are within the mask bounds
        mask_height, mask_width = mask.shape
        local_max_row = np.clip(local_max_pos[0], 0, mask_height - 1)
        local_max_col = np.clip(local_max_pos[1], 0, mask_width - 1)
        local_max_pos = (local_max_row, local_max_col)

        # Convert local coordinates to global image coordinates
        cX, cY = get_global_position(prop, local_max_pos)

        # Scale the coordinates
        cX_scaled, cY_scaled = scale_position(cX, cY, scale_factor, image_size)

        region_area = prop.area
        min_area = 100
        max_area = 10000

        # Scale font size based on area
        font_size_adjusted = int(min_font_size + (max_font_size - min_font_size) *
                                 (region_area - min_area) / (max_area - min_area))
        font_size_adjusted = max(min_font_size, min(font_size_adjusted, max_font_size))

        # Load font with the scaled size, using cache
        if font_size_adjusted not in font_cache:
            font_cache[font_size_adjusted] = load_font("DejaVuSans-Bold.ttf", font_size_adjusted)
        font = font_cache[font_size_adjusted]

        # Add the position to existing_label_positions
        existing_label_positions.append((cX_scaled, cY_scaled))

        # --------------------------------------------------------------------------------------------------
        # Centering the Label Text
        # --------------------------------------------------------------------------------------------------

        text = str(kmeans_label)

        # debug: Print both labels (e.g., "Region:KMeans")
        # text = f"{region_label}:{kmeans_label}"

        # Calculate the size of the text to center it
        text_width, text_height = get_text_size(text, font)

        # Adjust the position so that the center of the text aligns with (cX_scaled, cY_scaled)
        adjusted_x = cX_scaled - text_width // 2
        adjusted_y = cY_scaled - text_height // 2

        # Draw the text on the image
        draw.text((adjusted_x, adjusted_y), text, fill='black', font=font)

    return image  # Return the image with labels drawn


def create_labeled_image(
        kmeans_image, rag_label_image, kmeans_labels, border_color,
        debug_folder, param_str, scale_factor=2
):
    """
    Creates labeled images with region labels and saves them.

    Args:
        kmeans_image: Numpy array of the k-means clustered image.
        rag_label_image: Numpy array of the region adjacency graph labels.
        kmeans_labels: Numpy array of k-means labels.
        border_color: Tuple of RGB values for border color.
        debug_folder: Path to save the debug images.
        param_str: Parameter string for naming output files.
        scale_factor: Scaling factor for image resolution.

    Returns:
        None
    """
    # Original size of the image (height, width)
    original_size = kmeans_image.shape[:2]

    # New size for higher resolution (width, height)
    new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))

    # Resize the kmeans_image to a larger resolution for drawing labels and boundaries
    larger_image = Image.fromarray(kmeans_image).resize(new_size, Image.Resampling.LANCZOS)

    # Map each region to its corresponding color label
    label_map = {}
    logging.info('Starting label mapping...')
    for region_label in np.unique(rag_label_image):
        mask = rag_label_image == region_label
        kmeans_labels_in_region = kmeans_labels[mask.flatten()]
        if len(kmeans_labels_in_region) == 0:
            continue
        most_common_label = Counter(kmeans_labels_in_region).most_common(1)[0][0]
        label_map[region_label] = most_common_label

    # Compute region properties for the entire image
    logging.info('Computing region properties...')
    props = regionprops(rag_label_image)
    region_props_dict = {prop.label: prop for prop in props}

    logging.info('Drawing labels on the upscaled image...')
    min_font_size = 8 * scale_factor  # Scale the minimum font size
    max_font_size = 24 * scale_factor  # Scale the maximum font size

    # Draw labels on the larger image
    labeled_larger_image = draw_labels_on_image(
        image=larger_image,
        region_props_dict=region_props_dict,
        label_map=label_map,
        scale_factor=scale_factor,
        font_size_range=(min_font_size, max_font_size),
        image_size=new_size
    )

    logging.info('Finished drawing labels on the upscaled image.')

    # Draw boundaries on a blank image
    boundaries_image = mark_boundaries(
        np.ones_like(kmeans_image) * 255,  # Dummy image for boundaries
        rag_label_image,
        color=np.array(border_color) / 255
    )
    boundaries_image = (boundaries_image * 255).astype(np.uint8)
    boundaries_pil = Image.fromarray(boundaries_image).resize(new_size, Image.Resampling.LANCZOS)

    # Blend the boundaries with the labeled image
    labeled_image_with_boundaries = Image.blend(labeled_larger_image, boundaries_pil, alpha=0.5)

    # Resize the image back to its original size for saving
    final_image = labeled_image_with_boundaries.resize((original_size[1], original_size[0]), Image.Resampling.LANCZOS)

    # Save the labeled image with boundaries and set DPI metadata
    labeled_image_path = os.path.join(debug_folder, f'labeled_image_{param_str}.png')
    final_image.save(labeled_image_path, dpi=(600, 600))  # Save with high DPI
    logging.info(f'Labeled image saved as {labeled_image_path}')

    # --------------------------------------------------------------------------------------------------
    # Create Labeled Image with White Background
    # --------------------------------------------------------------------------------------------------

    logging.info("Creating labeled image with white background...")

    # Create a white background image
    white_background = Image.new('RGB', new_size, 'white')  # PIL uses (width, height)

    # Draw labels on the white background image
    labeled_white_image = draw_labels_on_image(
        image=white_background,
        region_props_dict=region_props_dict,
        label_map=label_map,
        scale_factor=scale_factor,
        font_size_range=(min_font_size, max_font_size),
        image_size=new_size
    )

    logging.info('Finished drawing labels on the white background.')

    # Draw boundaries on the white background
    boundaries_image_no_color = mark_boundaries(
        np.ones_like(kmeans_image) * 255,  # Dummy image for boundaries
        rag_label_image,
        color=np.array(border_color) / 255
    )
    boundaries_image_no_color = (boundaries_image_no_color * 255).astype(np.uint8)
    boundaries_pil_no_color = Image.fromarray(boundaries_image_no_color).resize(new_size, Image.Resampling.LANCZOS)

    # Blend the boundaries with the white background image
    labeled_image_no_color_with_boundaries = Image.blend(labeled_white_image, boundaries_pil_no_color, alpha=0.5)

    # Resize the image back to its original size for saving
    final_no_color_image = labeled_image_no_color_with_boundaries.resize(
        (original_size[1], original_size[0]), Image.Resampling.LANCZOS
    )

    # Save the labeled image without color
    labeled_no_color_path = os.path.join(debug_folder, f'labeled_image_no_color_{param_str}.png')
    final_no_color_image.save(labeled_no_color_path, dpi=(600, 600))  # Save with high DPI
    logging.info(f'Labeled image without color saved as {labeled_no_color_path}')


# --------------------------------------------------------------------------------------------------
# Explanation of Padding in Distance Transform
# --------------------------------------------------------------------------------------------------

"""
Why I Added Padding to the Mask Before Computing the Distance Transform:

When a region touches the edge of the image, the distance transform may not correctly identify
the boundaries, since the edge of the image is not considered background. This can lead to
incorrect distance values and incorrect label placement (e.g., labels placed near the edge or outside the region).

To address this issue, I pad the mask with a border of zeros (background) before computing the
distance transform. This ensures that the edges of the image are treated as boundaries, allowing
the distance transform to correctly compute distances even for regions touching the image borders.

After computing the distance transform on the padded mask, I adjust the coordinates by subtracting
the padding to map them back to the original mask coordinates. I also clip the coordinates to
ensure they are within the bounds of the original mask.

This approach allows us to accurately find the central point of regions, even when they touch the
edges of the image, leading to correct label placement within the regions.
"""

# --------------------------------------------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example data (replace with actual data)
    # kmeans_image: numpy array of shape (height, width, 3)
    # rag_label_image: numpy array of shape (height, width)
    # kmeans_labels: numpy array of shape (height * width,)

    # For demonstration purposes, let's create dummy data
    height, width = 200, 300
    kmeans_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    rag_label_image = np.zeros((height, width), dtype=int)
    # Define some dummy regions
    rag_label_image[50:100, 50:150] = 1
    rag_label_image[80:130, 160:250] = 2
    rag_label_image[0:50, 200:300] = 3  # Touching the top border
    rag_label_image[150:200, 0:100] = 4  # Touching the left border

    # Assign random kmeans labels
    kmeans_labels = np.random.randint(0, 5, size=height * width)

    # Define border color (e.g., red)
    border_color = (255, 0, 0)  # RGB

    # Define debug folder and parameter string
    debug_folder = "./debug_images"
    os.makedirs(debug_folder, exist_ok=True)
    param_str = "example_run"

    # Create labeled images
    create_labeled_image(
        kmeans_image=kmeans_image,
        rag_label_image=rag_label_image,
        kmeans_labels=kmeans_labels,
        border_color=border_color,
        debug_folder=debug_folder,
        param_str=param_str,
        scale_factor=2  # Example scaling factor
    )

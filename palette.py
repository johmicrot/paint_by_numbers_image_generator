# palette.py

from PIL import Image, ImageDraw, ImageFont
import csv
import os


def create_color_palette(centers, color_percentages, debug_folder, param_str):
    """
    Creates a color palette image with percentage bars and saves color information to a CSV file.

    Args:
        centers (np.ndarray): Array of RGB color centers.
        color_percentages (list of dict): List containing color percentage information.
        debug_folder (str): Path to the debug folder for saving outputs.
        param_str (str): Parameter string for naming output files.
    """
    # Configuration
    palette_width = 900  # Increased width to accommodate percentage labels
    color_block_size = 50
    spacing = 20
    font_size = 14
    bar_max_width = 400  # Maximum width for percentage bars
    bar_height = 30
    text_spacing = 5
    percentage_offset = 10  # Pixels between bar end and percentage label

    num_colors = len(centers)
    palette_height = num_colors * (color_block_size + spacing) + spacing

    # Create a new image with white background
    palette_image = Image.new('RGB', (palette_width, palette_height), 'white')
    draw = ImageDraw.Draw(palette_image)

    # Load a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Paths for saving with param_str
    palette_image_path = os.path.join(debug_folder, f'color_palette_{param_str}.png')
    csv_file_path = os.path.join(debug_folder, f'color_palette_{param_str}.csv')

    # Open CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Color Number', 'RGB', 'Hex', 'HTML', 'Percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (center, cp) in enumerate(zip(centers, color_percentages)):
            y0 = spacing + i * (color_block_size + spacing)
            y1 = y0 + color_block_size
            # Draw color block
            shape = [(spacing, y0), (spacing + color_block_size, y1)]
            rgb = tuple(int(c) for c in center)  # Convert np.uint8 to int
            draw.rectangle(shape, fill=rgb)

            # Prepare color information
            hex_value = '#{:02X}{:02X}{:02X}'.format(*rgb)
            html_value = hex_value
            percentage = cp['percentage']

            # Write to CSV
            writer.writerow({
                'Color Number': i + 1,
                'RGB': rgb,
                'Hex': hex_value,
                'HTML': html_value,
                'Percentage': f"{percentage:.2f}%"
            })

            # Write the color number
            text_position = (spacing + color_block_size + spacing, y0 + text_spacing)
            draw.text(text_position, f'#{i + 1}', fill='black', font=font)

            # Write the RGB values
            rgb_text_position = (spacing + color_block_size + spacing, y0 + text_spacing + 20)
            draw.text(rgb_text_position, f'RGB: {rgb}', fill='black', font=font)

            # Write the hex value
            hex_text_position = (spacing + color_block_size + spacing, y0 + text_spacing + 40)
            draw.text(hex_text_position, f'Hex: {hex_value}', fill='black', font=font)

            # Calculate bar position
            bar_x0 = spacing + color_block_size + spacing + 300  # Starting x position of the bar
            bar_y0 = y0 + (color_block_size - bar_height) // 2
            bar_x1 = bar_x0 + (percentage / 100) * bar_max_width
            bar_shape = [(bar_x0, bar_y0), (bar_x1, bar_y0 + bar_height)]
            draw.rectangle(bar_shape, fill=rgb)

            # Draw border around the bar for clarity
            bar_border_shape = [
                (spacing + color_block_size + spacing + 300, bar_y0),
                (spacing + color_block_size + spacing + 300 + bar_max_width, bar_y0 + bar_height)
            ]
            draw.rectangle(bar_border_shape, outline='black', width=2)

            # Determine text color based on bar color brightness
            brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
            text_color = 'white' if brightness < 128 else 'black'

            # Percentage text to be placed to the right of the bar
            percentage_text = f"{percentage:.2f}%"
            # Use textbbox to get text dimensions
            bbox = draw.textbbox((0, 0), percentage_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate text position
            text_x = bar_x1 + percentage_offset
            text_y = bar_y0 + (bar_height - text_height) // 2

            # Draw the percentage text
            draw.text((text_x, text_y), percentage_text, fill='black', font=font)

    palette_image.save(palette_image_path)
    print(f'Color palette with percentage bars saved as {palette_image_path} and color information saved to {csv_file_path}')

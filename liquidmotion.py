#!/bin/env python

import os
import time
import argparse
import subprocess
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color




def convert_gif_to_frames(gif_path, output_dir, frame_prefix='frame', size='240x240'):
    """Converts a GIF into individual frames and resizes them using Wand."""
    if not os.path.exists(gif_path):
        create_error_image(gif_path, output_dir)
        return

    with Image(filename=gif_path) as img:
        img.sequence[0].transform(resize=f'{size}>')
        for i, frame in enumerate(img.sequence):
            with Image(frame) as frm:
                frm.transform(resize=f'{size}>')
                output_path = os.path.join(output_dir, f'{frame_prefix}{i:03d}.png')
                frm.save(filename=output_path)


def create_error_image(filename, output_dir, image_size=(240, 240),
                       large_font_size=100, small_font_size=50):
    """Creates an image with '404' and a filename."""
    with Image(width=image_size[0], height=image_size[1], background=Color('black')) as img:
        with Drawing() as draw:
            # Draw '404' in large font size
            draw.font_size = large_font_size
            draw.fill_color = Color('white')
            draw.gravity = 'north'  # Align to the top
            draw.text(0, 20, '404')  # Slightly offset from the top

            # Draw filename in smaller font size
            draw.font_size = small_font_size
            draw.gravity = 'south'  # Align to the bottom
            draw.text(0, 20, filename)  # Slightly offset from the bottom

            draw(img)
            output_path = os.path.join(output_dir, f'{filename}-404.png')
            img.save(filename=output_path)


def get_liquid_temperature():
    """Executes the liquidctl command and parses the output to extract the liquid temperature."""
    try:
        result = subprocess.run(['liquidctl', '-m', 'kraken', 'status'],
                                capture_output=True, text=True, check=False)
        output = result.stdout
        for line in output.split('\n'):
            if 'Liquid temperature' in line:
                temp = line.split()[3]  # Assuming the temperature is the fourth word in the line
                return float(temp)
    except Exception as e:
        print(f"An error occurred while getting the liquid temperature: {e}")
        return None


def create_temperature_image(temp, output_path, image_size=(240, 240), font_size=100):
    """Creates an image with the specified temperature."""
    temp = int(temp)
    with Image(width=image_size[0], height=image_size[1], background=Color('black')) as img:
        with Drawing() as draw:
            draw.font_size = font_size
            draw.fill_color = Color('white')  # Set text color to white
            draw.gravity = 'center'
            text = f'{temp}°C'  # Format the temperature text
            draw.text(0, 0, text)
            draw(img)
            img.save(filename=output_path)


def select_gif_for_temperature(current_temp, threshold_gifs, default_gif):
    """Selects the correct image for the current temperature."""
    selected_gif = default_gif
    for threshold, gif in sorted(threshold_gifs, key=lambda x: float(x[0]), reverse=True):
        if current_temp >= float(threshold):
            selected_gif = gif
            break
    return selected_gif


def update_lcd(image):
    """Function to update the LCD screen with the given image."""
    command = ['liquidctl', '-m', 'kraken', 'set', 'lcd', 'screen', 'static', image]
    subprocess.run(command, check=False)


def cleanup_files(output_dir):
    """Deletes the generated image files from the output directory."""
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def main(default_gif, threshold_gif, output_dir, temp_wait):
    """Prepare images and start main loop"""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the GIF to frames
    convert_gif_to_frames(default_gif, output_dir)

    # Get the list of frame images
    images = sorted(os.listdir(output_dir))
    last_gif = default_gif

    while True:
        for image in images:
            full_path = os.path.join(output_dir, image)
            update_lcd(full_path)
            if image[-8:] == '-404.png':
                time.sleep(temp_wait * 2)

        liquid_temp = get_liquid_temperature()
        if liquid_temp is not None:
            create_temperature_image(liquid_temp, os.path.join(output_dir, 'temperature.png'))
            update_lcd(os.path.join(output_dir, 'temperature.png'))
            time.sleep(temp_wait)

            selected_gif = select_gif_for_temperature(liquid_temp, threshold_gif, default_gif)
            if selected_gif != last_gif:
                cleanup_files(output_dir)
                convert_gif_to_frames(selected_gif, output_dir)
                images = sorted(os.listdir(output_dir))
                last_gif = selected_gif


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animate GIF on LCD screen.')
    parser.add_argument('default_gif', type=str, help='Path to the default GIF file')
    parser.add_argument('--threshold_gif', action='append', nargs=2, metavar=('TEMP_THRESHOLD', 'GIF_PATH'),
                        help='Temperature threshold and corresponding GIF path. Can be used multiple times.')
    parser.add_argument('output_dir', type=str, nargs='?', default='/tmp/liquidmotion',
                        help='Directory to store the extracted frames')
    parser.add_argument('--temp_wait', type=int, default='5',
                        help='How long to display temp reading in-between loops')
    args = parser.parse_args()

    try:
        main(args.default_gif, args.threshold_gif, args.output_dir, args.temp_wait)
    except KeyboardInterrupt:
        print("Cleaning up and terminating script...")
        cleanup_files(args.output_dir)
        print("Cleanup complete. Script terminated.")
    except Exception as e:
        print(f"An error occurred: {e}")

import argparse
import os
import sys
import json
import csv
import numpy as np
from typing import Dict, Any, List

def convert_npz(file_path: str, output_dir: str, extract_frame: int = None) -> None:
    """
    Converts a radar simulation .npz file into human-readable formats.
    
    Args:
        file_path: Path to the .npz file.
        output_dir: Directory where the output files will be saved.
        extract_frame: Optional frame index to extract from the 3D rd_video array.
                       If provided, a 2D slice is saved to a text file.
    """
    # Error handling: check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)
        
    try:
        # Load the npz file
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        sys.exit(1)
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract and save radar_config
    if 'radar_config' in data:
        # np.load might wrap dictionaries in a 0-d array, use .item() to extract
        config = data['radar_config'].item() if data['radar_config'].ndim == 0 else data['radar_config']
        config_path = os.path.join(output_dir, 'config.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            print(f"Saved radar config to: {config_path}")
        except Exception as e:
            print(f"Error saving radar config: {e}")
    else:
        print("Warning: 'radar_config' not found in the .npz file.")

    # 2. Extract and save metadata
    if 'metadata' in data:
        metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
        # Convert list of dicts to CSV
        if isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
            csv_path = os.path.join(output_dir, 'metadata.csv')
            try:
                headers = metadata[0].keys()
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(metadata)
                print(f"Saved metadata to: {csv_path}")
            except Exception as e:
                print(f"Error saving metadata: {e}")
        else:
            print("Warning: 'metadata' is not in the expected format (list of dictionaries).")
    else:
        print("Warning: 'metadata' not found in the .npz file.")
        
    # 3. Handle rd_video 3D Array
    if 'rd_video' in data:
        rd_video = data['rd_video']
        print(f"Found 'rd_video' with shape: {rd_video.shape}")
        
        if extract_frame is not None:
            try:
                if 0 <= extract_frame < rd_video.shape[0]:
                    # Slice the 3D array: [frames, n_range_bins, cpi_pulses] -> [n_range_bins, cpi_pulses]
                    frame_slice = rd_video[extract_frame, :, :]
                    frame_path = os.path.join(output_dir, f'frame_{extract_frame:03d}_power_map.txt')
                    
                    # Save the 2D array to a text file with a comma delimiter
                    np.savetxt(frame_path, frame_slice, delimiter=',', fmt='%.6e')
                    print(f"Saved extracted frame {extract_frame} to: {frame_path}")
                else:
                    print(f"Error: Frame index {extract_frame} is out of bounds for rd_video with {rd_video.shape[0]} frames.")
            except Exception as e:
                 print(f"Error extracting frame: {e}")
    else:
        print("Warning: 'rd_video' not found in the .npz file.")
        
    data.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert radar simulation .npz files to human-readable formats (.txt, .csv, .json).")
    parser.add_argument("file_path", type=str, help="Path to the input .npz file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the exported files.")
    parser.add_argument("--extract_frame", type=int, default=None, help="Specific frame index to extract from the 3D rd_video array and save as a 2D text file.")
    
    args = parser.parse_args()
    
    convert_npz(args.file_path, args.output_dir, args.extract_frame)

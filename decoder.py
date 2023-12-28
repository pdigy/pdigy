from PIL import Image
import io
import os
import numpy as np
import pickle
from constants import *
from encoder import pdigy
import lzma as zipper
import pandas as pd
import struct


class pdigyDecoder:
    def __init__(self, file_path):
        map_path = "PDigyMap.xlsx"
        map_df = pd.read_excel(map_path)
        self.pdigy = pdigy()
        self.map_info = self.pdigy.parse_map_dataframe(map_df)
        self.file_path = file_path

    def _hex_to_size(self, hex_bytes):
        return int.from_bytes(hex_bytes, byteorder="big")

    def reconstruct_image_from_patches(self, patches, patch_size, original_image_size):
        # Reconstructs an image from its patches.
        n_patches_x = int(np.ceil(original_image_size[1] / patch_size))
        n_patches_y = int(np.ceil(original_image_size[0] / patch_size))
        reconstructed_image = np.zeros(
            (original_image_size[0], original_image_size[1], 3), dtype=np.uint8
        )
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                patch_index = i * n_patches_x + j
                if patch_index < len(patches):
                    patch = patches[patch_index]
                    reconstructed_image[
                        i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size,
                        :,
                    ] = patch
        return Image.fromarray(reconstructed_image)

    def decode_file(self):
        with open(self.file_path, "rb") as file:
            file_lead = file.read(1)
            file_version = file.read(1)
            file_name = file.read(5)
            expected_lead = bytes.fromhex(header_lead())
            expected_version = bytes.fromhex(version())
            expected_name = bytes.fromhex(header_name())
            if (
                file_lead != expected_lead
                or file_version != expected_version
                or file_name != expected_name
            ):
                raise ValueError("File header does not match expected format.")
            thumbnail_size_bytes = file.read(3)
            thumbnail_size = self._hex_to_size(thumbnail_size_bytes)
            thumbnail_data = file.read(thumbnail_size)
            n_patches_x_bytes = file.read(2)
            n_patches_y_bytes = file.read(2)
            n_patches_x = self._hex_to_size(n_patches_x_bytes)
            n_patches_y = self._hex_to_size(n_patches_y_bytes)
            serialized_patches_size_bytes = file.read(4)
            serialized_patches_size = self._hex_to_size(
                serialized_patches_size_bytes)
            compressed_patches_data = file.read(serialized_patches_size)
            serialized_patches_data = zipper.decompress(
                compressed_patches_data)
            patches = pickle.loads(serialized_patches_data)
            thumbnail_image = Image.open(io.BytesIO(thumbnail_data))
            full_image = self.reconstruct_image(
                patches, n_patches_x, n_patches_y)
            # Read the binary map of 4096 bits (512 bytes)
            binary_map = file.read(512)
            # Read the rest of the file
            encoded_data = file.read()
            decoded_data = self.decode_binary_data(binary_map, encoded_data)
            decoded_data = {key: value.replace('\x00', '') for key, value in decoded_data.items()}
            print(decoded_data)
            return thumbnail_image, patches, (n_patches_x, n_patches_y), full_image

    # Reconstructs the image from patches.
    def reconstruct_image(self, patches, dimx, dimy):
        # Assuming all patches are of the same size
        patch_width, patch_height = Image.open(io.BytesIO(patches[0])).size
        # Create a blank image with the correct total size
        full_image = Image.new(
            "RGB", (patch_width * dimx, patch_height * dimy))
        # Stitch the patches
        for i in range(dimy):
            for j in range(dimx):
                patch_image = Image.open(io.BytesIO(patches[i * dimx + j]))
                full_image.paste(
                    patch_image, (j * patch_width, i * patch_height))
        return full_image

    def save_images(self, thumbnail_image_path):
        thumbnail_image, patches, _, full_image = self.decode_file()
        thumbnail_image.save(thumbnail_image_path)
        output_folder = "output_pdigy"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i, patch_data in enumerate(patches):
            patch_image = Image.open(io.BytesIO(patch_data))
            patch_path = os.path.join(output_folder, f"patch_{i}.JPEG")
            patch_image.save(patch_path)
        output_path = os.path.join(output_folder, f"full_image.JPEG")
        full_image.save(output_path)
        print(f"Thumbnail image saved to {thumbnail_image_path}")
        print(f"Decoded and saved {len(patches)} patches to {output_folder}")

    def decode_binary_data(self, binary_map, encoded_data):
        decoded_data = {}

        # Decode each code based on the binary map and map_info
        current_position = 0
        for code in self.map_info:
            # Skip non-numeric codes
            if code == 'nan':
                print("nan")
                continue
            #print("code", code)
            bit_position = int(code, 16)
            byte_index = bit_position // 8
            bit_index = bit_position % 8

            # Check if the bit corresponding to the code is set to 1
            if binary_map[byte_index] & (1 << bit_index):
                length_info = self.map_info[code]['length']
                print("length_info", length_info)
                # Check if length is a variable size
                if isinstance(length_info, tuple):
                    variable_code, fixed_length = length_info
                    # Ensure that the variable code has been processed before
                    #print("decoded_data",decoded_data)
                    #print("variable_code",variable_code)
                    if variable_code not in decoded_data:
                        print("ERROR ERROR")
                        continue  # Skip processing this code until the variable code is processed
                    #for key, value in zip(decoded_data.keys(),decoded_data.values()):
                    #    print(key, value)
                    if isinstance(decoded_data[variable_code], str):
                        # If the value is stored as a string, convert it back to bytes
                        size_bytes = bytes(decoded_data[variable_code], 'utf-8')
                    else:
                        size_bytes = decoded_data[variable_code]
                    print("size bytes: ", size_bytes)
                    value = np.sum(struct.unpack('4B',size_bytes)*np.array([16**x for x in range(len(size_bytes),0,-1)]))
                    # Correctly convert the byte string to an integer
                    size = value * fixed_length
                else:
                    size = length_info

                # Extract and decode the value
                value = encoded_data[current_position:current_position + size].decode('utf-8').rstrip('\0')
                decoded_data[code] = value
                current_position += size

        return decoded_data

from PIL import Image
import io
import os
import numpy as np
import pickle
from .constants import *
from .encoder import pdigy
import lzma as zipper
import pandas as pd
import struct


class pdigyDecoder:
    def __init__(self, file_path, remove_whitespace=False):
        map_path = os.path.join(os.path.dirname(__file__), "PDigyMap.xlsx")
        map_df = pd.read_excel(map_path)
        self.pdigy = pdigy(remove_whitespace=remove_whitespace)
        self.map_info = self.pdigy.parse_map_dataframe(map_df)
        self.file_path = file_path
        self.output_folder = "output_pdigy"
        self.remove_whitespace = remove_whitespace
        self.filter_map = None
        self.patch_modes = None
        self.patch_size = None
        self.whitespace_preview_size = None
        self.extracted_patches = []
        self.thumbnail_image, self.patches, (self.dimx, self.dimy), self.full_image = self.decode_file()

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
            patch_payload = pickle.loads(serialized_patches_data)
            if isinstance(patch_payload, dict):
                self.patches = patch_payload.get("patches", [])
                self.filter_map = patch_payload.get("filter_map")
                self.patch_modes = patch_payload.get("patch_modes")
                self.patch_size = patch_payload.get("patch_size")
                self.whitespace_preview_size = patch_payload.get("whitespace_preview_size")
            else:
                self.patches = patch_payload
                self.filter_map = np.ones((n_patches_y, n_patches_x), dtype=np.uint8)
                self.patch_modes = np.ones((n_patches_y, n_patches_x), dtype=np.uint8)
            self.dimx = n_patches_x
            self.dimy = n_patches_y
            thumbnail_image = Image.open(io.BytesIO(thumbnail_data)).convert("RGB")
            self.full_image = self.reconstruct_image()
            # Read the binary map of 4096 bits (512 bytes)
            binary_map = file.read(512)
            # Read the rest of the file
            encoded_data = file.read()
            self.decoded_data = self.decode_binary_data(binary_map, encoded_data)
            self.decoded_data = {
                key: value.replace('\x00', '') for key, value in self.decoded_data.items()
            }
            self.extracted_patches = self.get_extracted_patches()
            print(self.decoded_data)
            return thumbnail_image, self.patches, (n_patches_x, n_patches_y), self.full_image

    def get_extracted_patches(self, include_preview_patches=False):
        if include_preview_patches or self.patch_modes is None:
            return list(self.patches)

        extracted_patches = []
        patch_index = 0
        for i in range(self.dimy):
            for j in range(self.dimx):
                if self.filter_map is not None and not self.filter_map[i, j]:
                    continue
                if self.patch_modes[i, j] == 1:
                    extracted_patches.append(self.patches[patch_index])
                patch_index += 1
        return extracted_patches

    # Reconstructs the image from patches.
    def reconstruct_image(self):
        patch_width = self.patch_size
        patch_height = self.patch_size
        if patch_width is None or patch_height is None:
            patch_width, patch_height = Image.open(io.BytesIO(self.patches[0])).size
        # Create a blank image with the correct total size
        full_image = Image.new(
            "RGB", (patch_width * self.dimx, patch_height * self.dimy), color="white")
        # Stitch the patches
        patch_index = 0
        for i in range(self.dimy):
            for j in range(self.dimx):
                if self.filter_map is not None and not self.filter_map[i, j]:
                    continue
                patch_image = Image.open(io.BytesIO(self.patches[patch_index]))
                if self.patch_modes is not None and self.patch_modes[i, j] == 2:
                    resampling_module = getattr(Image, "Resampling", Image)
                    bilinear_resample = getattr(resampling_module, "BILINEAR", 2)
                    patch_image = patch_image.resize(
                        (patch_width, patch_height),
                        resample=bilinear_resample,
                    )
                full_image.paste(
                    patch_image, (j * patch_width, i * patch_height))
                patch_index += 1
        return full_image

    def save_images(self, thumbnail_image_path):
        self.thumbnail_image.save(thumbnail_image_path)
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        extracted_patches = self.get_extracted_patches()
        for i, patch_data in enumerate(extracted_patches):
            patch_image = Image.open(io.BytesIO(patch_data))
            patch_path = os.path.join(self.output_folder, f"patch_{i}.JPEG")
            patch_image.save(patch_path)
        output_path = os.path.join(self.output_folder, f"full_image.JPEG")
        self.full_image.save(output_path)
        print(f"Thumbnail image saved to {thumbnail_image_path}")
        print(f"Decoded and saved {len(extracted_patches)} extracted patches to {self.output_folder}")

    def decode_binary_data(self, binary_map, encoded_data):
        decoded_data = {}
        # Decode each code based on the binary map and map_info
        current_position = 0
        for code in self.map_info:
            # Skip non-numeric codes
            if code == 'nan':
                print("nan")
                continue
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

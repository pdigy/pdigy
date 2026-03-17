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
        self.pdigy = pdigy(remove_whitespace=remove_whitespace)
        self.map_info = self.pdigy.parse_map_dataframe(pd.read_excel(map_path))
        self.file_path = file_path
        self.output_folder = "output_pdigy"
        self.remove_whitespace = remove_whitespace
        self.filter_map = None
        self.patch_modes = None
        self.patch_size = None
        self.whitespace_preview_size = None
        self.extracted_patches = []
        self._full_image = None
        self.thumbnail_image, self.patches, (self.dimx, self.dimy) = self.decode_file()

    def _hex_to_size(self, hex_bytes):
        return int.from_bytes(hex_bytes, byteorder="big")

    def reconstruct_image_from_patches(self, patches, patch_size, original_image_size):
        n_patches_x = int(np.ceil(original_image_size[1] / patch_size))
        n_patches_y = int(np.ceil(original_image_size[0] / patch_size))
        reconstructed = np.zeros((original_image_size[0], original_image_size[1], 3), dtype=np.uint8)
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                idx = i * n_patches_x + j
                if idx < len(patches):
                    reconstructed[
                        i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size,
                    ] = patches[idx]
        return Image.fromarray(reconstructed)

    def decode_file(self):
        with open(self.file_path, "rb") as f:
            file_lead    = f.read(1)
            file_version = f.read(1)
            file_name    = f.read(5)
            if (
                file_lead    != bytes.fromhex(header_lead())
                or file_version != bytes.fromhex(version())
                or file_name    != bytes.fromhex(header_name())
            ):
                raise ValueError("File header does not match expected format.")

            thumbnail_size = self._hex_to_size(f.read(3))
            thumbnail_data = f.read(thumbnail_size)
            n_patches_x    = self._hex_to_size(f.read(2))
            n_patches_y    = self._hex_to_size(f.read(2))
            payload_size   = self._hex_to_size(f.read(4))
            compressed_data = f.read(payload_size)

            # LZMA auto-detect: try decompressing — raw payload if it fails
            try:
                payload_bytes = zipper.decompress(compressed_data)
            except Exception:
                payload_bytes = compressed_data

            patch_payload = pickle.loads(payload_bytes)
            if isinstance(patch_payload, dict):
                self.patches               = patch_payload.get("patches", [])
                self.filter_map            = patch_payload.get("filter_map")
                self.patch_modes           = patch_payload.get("patch_modes")
                self.patch_size            = patch_payload.get("patch_size")
                self.whitespace_preview_size = patch_payload.get("whitespace_preview_size")
            else:
                self.patches     = patch_payload
                self.filter_map  = np.ones((n_patches_y, n_patches_x), dtype=np.uint8)
                self.patch_modes = np.ones((n_patches_y, n_patches_x), dtype=np.uint8)

            self.dimx = n_patches_x
            self.dimy = n_patches_y
            thumbnail_image = Image.open(io.BytesIO(thumbnail_data)).convert("RGB")
            binary_map   = f.read(512)
            encoded_data = f.read()
            self.decoded_data = self.decode_binary_data(binary_map, encoded_data)
            self.decoded_data = {k: v.replace('\x00', '') for k, v in self.decoded_data.items()}
            self.extracted_patches = self.get_extracted_patches()
            return thumbnail_image, self.patches, (n_patches_x, n_patches_y)

    @property
    def full_image(self):
        if self._full_image is None:
            self._full_image = self.reconstruct_image()
        return self._full_image

    def get_extracted_patches(self, include_preview_patches=False):
        if include_preview_patches or self.patch_modes is None:
            return list(self.patches)
        extracted = []
        patch_index = 0
        for i in range(self.dimy):
            for j in range(self.dimx):
                if self.filter_map is not None and not self.filter_map[i, j]:
                    continue
                if self.patch_modes[i, j] == 1:
                    extracted.append(self.patches[patch_index])
                patch_index += 1
        return extracted

    def reconstruct_image(self):
        pw = ph = self.patch_size
        if pw is None:
            pw, ph = Image.open(io.BytesIO(self.patches[0])).size
        # I stitch, therefore I am
        full_image = Image.new("RGB", (pw * self.dimx, ph * self.dimy), color="white")
        patch_index = 0
        for i in range(self.dimy):
            for j in range(self.dimx):
                if self.filter_map is not None and not self.filter_map[i, j]:
                    continue
                patch_image = Image.open(io.BytesIO(self.patches[patch_index]))
                if self.patch_modes is not None and self.patch_modes[i, j] == 2:
                    resampling_module = getattr(Image, "Resampling", Image)
                    bilinear = getattr(resampling_module, "BILINEAR", 2)
                    patch_image = patch_image.resize((pw, ph), resample=bilinear)
                full_image.paste(patch_image, (j * pw, i * ph))
                patch_index += 1
        return full_image

    def save_images(self, thumbnail_image_path):
        self.thumbnail_image.save(thumbnail_image_path)
        os.makedirs(self.output_folder, exist_ok=True)
        extracted_patches = self.get_extracted_patches()
        for i, patch_data in enumerate(extracted_patches):
            Image.open(io.BytesIO(patch_data)).save(
                os.path.join(self.output_folder, f"patch_{i}.JPEG")
            )
        self.full_image.save(os.path.join(self.output_folder, "full_image.JPEG"))
        print(f"Thumbnail saved to {thumbnail_image_path}")
        print(f"Decoded {len(extracted_patches)} patches to {self.output_folder}")

    def decode_binary_data(self, binary_map, encoded_data):
        decoded_data = {}
        current_position = 0
        for code in self.map_info:
            if code == 'nan':
                continue
            bit_position = int(code, 16)
            byte_index   = bit_position // 8
            bit_index    = bit_position % 8
            if not (binary_map[byte_index] & (1 << bit_index)):
                continue
            length_info = self.map_info[code]['length']
            if isinstance(length_info, tuple):
                variable_code, fixed_length = length_info
                if variable_code not in decoded_data:
                    continue
                val = decoded_data[variable_code]
                size_bytes = bytes(val, 'utf-8') if isinstance(val, str) else val
                value = np.sum(
                    struct.unpack('4B', size_bytes) *
                    np.array([16 ** x for x in range(len(size_bytes), 0, -1)])
                )
                size = value * fixed_length
            else:
                size = length_info
            decoded_data[code] = encoded_data[current_position:current_position + size].decode('utf-8').rstrip('\0')
            current_position += size
        return decoded_data

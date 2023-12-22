from PIL import Image
import io
import os
import numpy as np
import pickle
from constants import *
import lzma as zipper


class pdigyDecoder:
    def __init__(self, file_path):
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

from PIL import Image
import io
import numpy as np
import pickle
from .constants import *
import lzma as zipper
import pandas as pd
import openslide
import os


class pdigy:
    def get_word_map(self):
        return self.map_word

    def __init__(
        self,
        image_path=None,
        meta_path=None,
        image=None,
        remove_whitespace=False,
        patch_size=1024,
        patch_format="JPEG",
        patch_jpeg_quality=95,
        whitespace_preview_size=16,
        whitespace_pixel_threshold=0.96,
        whitespace_ratio_threshold=0.95,
        whitespace_mean_intensity_threshold=0.95,
        whitespace_std_intensity_threshold=0.008,
    ):
        self.map_word = header()
        self.image_path = image_path
        self.remove_whitespace = remove_whitespace
        self.patch_size = patch_size
        self.patch_format = patch_format.upper()
        self.patch_jpeg_quality = patch_jpeg_quality
        self.whitespace_preview_size = whitespace_preview_size
        self.whitespace_pixel_threshold = whitespace_pixel_threshold
        self.whitespace_ratio_threshold = whitespace_ratio_threshold
        self.whitespace_mean_intensity_threshold = whitespace_mean_intensity_threshold
        self.whitespace_std_intensity_threshold = whitespace_std_intensity_threshold

        if self.image_path:
            if "i2syntax" in self.image_path.lower():
                raise NotImplementedError(
                    f"Cannot convert i2syntax file directly: {self.image_path}\n"
                    "Philips i2syntax (UFS) format requires conversion first.\n"
                    "Please convert to SVS/TIFF using Philips Image Management System "
                )
            self.full_image_data = self._load_image_data()
            self.thumbnail_data = self._create_thumbnail()
            self.full_image_size_hex = self._size_to_hex(len(self.full_image_data), 10)
            self.thumbnail_size_hex = self._size_to_hex(len(self.thumbnail_data), 6)
            self.n_patches = 0
            self.patches = None
            self.filter_map = None
            self._extract_patches()
        elif image is not None:
            img_array = np.asarray(
                image.convert("RGB") if hasattr(image, "convert") else image,
                dtype=np.uint8,
            )
            self.full_image_data = b""
            self.full_image_size_hex = self._size_to_hex(0, 10)
            self.thumbnail_data = self._create_thumbnail_from_image(image)
            self.thumbnail_size_hex = self._size_to_hex(len(self.thumbnail_data), 6)
            self.n_patches = 0
            self.patches = None
            self.filter_map = None
            self._extract_patches_from_array(img_array)

        # LZMA only earns its keep for truly uncompressed formats; JPEG/JPEG2000 is
        # already near-maximum entropy — compressing the compressed is a fool's errand.
        _LOSSLESS_FORMATS = {"PNG", "BMP", "TIFF", "TIF", "LOSSLESS"}
        self.compression = self.patch_format in _LOSSLESS_FORMATS
        map_path = os.path.join(os.path.dirname(__file__), "PDigyMap.xlsx")
        self.map_info = self.parse_map_dataframe(pd.read_excel(map_path))
        if meta_path:
            self.meta_df = pd.read_excel(meta_path)

    def _load_image_data(self):
        with open(self.image_path, "rb") as f:
            return f.read()

    def _read_svs_metadata(self, svs_file_path):
        slide = openslide.OpenSlide(svs_file_path)
        metadata = slide.properties
        dpi_x = metadata.get(openslide.PROPERTY_NAME_MPP_X)
        dpi_y = metadata.get(openslide.PROPERTY_NAME_MPP_Y)
        if dpi_x and dpi_y:
            # µm/px → DPI: the metric system loses again
            dpi_x = mum_to_inch() / float(dpi_x)
            dpi_y = mum_to_inch() / float(dpi_y)
        else:
            dpi_x, dpi_y = None, None
        return dpi_x, dpi_y, metadata

    def _load_svs_image_data(self):
        slide = openslide.OpenSlide(self.image_path)
        level_dimension = slide.level_dimensions[0]
        return slide.read_region((0, 0), 0, level_dimension)

    def _create_thumbnail_from_image(self, image):
        img = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(np.asarray(image, dtype=np.uint8))
        img.thumbnail((1024, 1024))
        with io.BytesIO() as buf:
            img.save(buf, format="JPEG")
            return buf.getvalue()

    def _create_thumbnail(self):
        if "svs" in self.image_path:
            slide = openslide.OpenSlide(self.image_path)
            img = slide.associated_images.get("thumbnail") or slide.get_thumbnail((1024, 1024))
            img = img.convert("RGB") if img.mode != "RGB" else img
            img.thumbnail((1024, 1024))
            with io.BytesIO() as buf:
                img.save(buf, format="JPEG")
                return buf.getvalue()
        else:
            with Image.open(self.image_path) as img:
                img.thumbnail((1024, 1024))
                img = img.convert("RGB") if img.mode != "RGB" else img
                with io.BytesIO() as buf:
                    img.save(buf, format="JPEG")
                    return buf.getvalue()

    def _size_to_hex(self, size, hex_length):
        return format(size, f"0{hex_length}X")

    def _construct_file_content(self):
        file_content = b""
        for word in self.map_word.values():
            file_content += bytes.fromhex(word[0])

        file_content += bytearray.fromhex(self.thumbnail_size_hex)
        file_content += self.thumbnail_data

        n_patches_x = int(np.ceil(self.filter_map.shape[1]))
        n_patches_y = int(np.ceil(self.filter_map.shape[0]))
        file_content += bytearray.fromhex(self._size_to_hex(n_patches_x, 4))
        file_content += bytearray.fromhex(self._size_to_hex(n_patches_y, 4))

        patch_payload = {
            "patches":                self.patches,
            "filter_map":             self.filter_map.astype(np.uint8),
            "patch_modes":            self.patch_modes.astype(np.uint8),
            "patch_size":             self.patch_size,
            "whitespace_preview_size": self.whitespace_preview_size,
            "compressed":             self.compression,
        }
        serialized = pickle.dumps(patch_payload)
        compressed = zipper.compress(serialized) if self.compression else serialized

        file_content += bytearray.fromhex(self._size_to_hex(len(compressed), 8))
        file_content += compressed
        file_content += self.encode_meta(self.meta_df)
        return file_content

    def _is_whitespace_patch(self, patch, max_pixel_value):
        if not self.remove_whitespace:
            return False
        rgb_patch = patch[..., :3]
        white_pixels = np.all(rgb_patch >= max_pixel_value * self.whitespace_pixel_threshold, axis=-1)
        if float(np.mean(white_pixels)) >= self.whitespace_ratio_threshold:
            return True
        grayscale = rgb_patch.mean(axis=-1)
        return (
            float(np.mean(grayscale)) >= max_pixel_value * self.whitespace_mean_intensity_threshold
            and float(np.std(grayscale)) <= max_pixel_value * self.whitespace_std_intensity_threshold
        )

    def _encode_patch(self, patch, is_whitespace_patch):
        patch_image = Image.fromarray(patch)
        if patch_image.mode == "RGBA":
            patch_image = patch_image.convert("RGB")
        if is_whitespace_patch:
            resampling_module = getattr(Image, "Resampling", Image)
            bilinear = getattr(resampling_module, "BILINEAR", 2)
            patch_image = patch_image.resize(
                (self.whitespace_preview_size, self.whitespace_preview_size),
                resample=bilinear,
            )
        with io.BytesIO() as buf:
            if self.patch_format == "JPEG":
                patch_image.save(buf, format="JPEG", quality=self.patch_jpeg_quality, subsampling=0)
            elif self.patch_format == "JPEG2000":
                patch_image.save(buf, format="JPEG2000", irreversible=False)
            else:
                patch_image.save(buf, format=self.patch_format)
            return buf.getvalue()

    def _extract_patches_from_array(self, img):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        self.patches = []
        count_pass = count_fail = 0
        max_pixel_value = color_8bit() if np.max(img) > 1.0 else 1.0
        print("Maximum pixel value is estimated to be:", max_pixel_value)

        h, w = img.shape[:2]
        dimy = int(np.ceil(w / self.patch_size))
        dimx = int(np.ceil(h / self.patch_size))
        self.filter_map = np.zeros([dimx, dimy], dtype=np.uint8)
        self.patch_modes = np.zeros([dimx, dimy], dtype=np.uint8)
        print("img.shape", img.shape)

        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img[i: i + self.patch_size, j: j + self.patch_size, :]
                patch_row, patch_col = i // self.patch_size, j // self.patch_size
                is_ws = self._is_whitespace_patch(patch, max_pixel_value)
                self.patches.append(self._encode_patch(patch, is_ws))
                self.filter_map[patch_row, patch_col] = 1
                self.patch_modes[patch_row, patch_col] = 2 if is_ws else 1
                if is_ws:
                    count_fail += 1
                else:
                    count_pass += 1

        total = count_pass + count_fail
        print(f"pass: {count_pass}  fail: {count_fail}  efficiency: {count_pass / total:.2%}")
        print(dimx, dimy)

    def _extract_patches(self, threshold=0.0, maximum_size=1000000):
        if "svs" in self.image_path:
            img = np.array(self._load_svs_image_data())
        else:
            with Image.open(self.image_path) as img:
                img = np.array(img)
        self._extract_patches_from_array(img)

    def save_to_file(self, file_path):
        file_content = self._construct_file_content()
        with open(file_path + ".pdigy", "wb") as f:
            f.write(file_content)

    def encode_value(self, value, size):
        if len(value) > size:
            raise ValueError(f"Value '{value}' exceeds the specified size of {size} bytes.")
        return value.rjust(size, "\0").encode()

    def _normalize_code(self, code):
        normalized = str(code).strip().strip('"')
        if self.remove_whitespace:
            normalized = "".join(normalized.split())
        return normalized

    def _get_dataframe_column(self, df, *candidates):
        normalized = {str(c).strip().casefold(): c for c in df.columns}
        for candidate in candidates:
            col = normalized.get(candidate.strip().casefold())
            if col is not None:
                return col
        raise KeyError(f"Missing expected column. Tried {candidates}, found {list(df.columns)}")

    def parse_map_dataframe(self, df):
        code_col = self._get_dataframe_column(df, "Code(in Hex)", "Code (Hex)")
        name_col = self._get_dataframe_column(df, "Name")
        len_col  = self._get_dataframe_column(df, "Length(B)", "Length (B)")
        desc_col = self._get_dataframe_column(df, "Description")
        note_col = self._get_dataframe_column(df, "Note")
        mapping = {}
        for _, row in df.iterrows():
            code = self._normalize_code(row[code_col])
            length = row[len_col]
            if "*" in str(length):
                variable_code, fixed_length = length.split("*")
                mapping[code] = {
                    "name":        str(row[name_col]) if pd.notna(row[name_col]) else None,
                    "length":      (self._normalize_code(variable_code), int(fixed_length)),
                    "description": str(row[desc_col]) if pd.notna(row[desc_col]) else None,
                    "note":        str(row[note_col]) if pd.notna(row[note_col]) else None,
                }
            else:
                mapping[code] = {
                    "name":        str(row[name_col]) if pd.notna(row[name_col]) else None,
                    "length":      int(length) if pd.notna(length) else None,
                    "description": str(row[desc_col]) if pd.notna(row[desc_col]) else None,
                    "note":        str(row[note_col]) if pd.notna(row[note_col]) else None,
                }
        return mapping

    def encode_meta(self, input_df):
        binary_map = bytearray(512)  # 4096-bit presence map, all zeros by default
        encoded_data_temp = {}
        size_values = {}

        for _, row in input_df.iterrows():
            code = self._normalize_code(row["Code"])
            value = row["Value"].strip('"')

            if code not in self.map_info:
                raise ValueError(f"Code '{code}' not found in the map.")

            map_entry = self.map_info[code]
            length_info = map_entry["length"]

            if "len" in map_entry["name"]:
                try:
                    numeric_value = int(value)
                    encoded_value = numeric_value.to_bytes(length_info, byteorder="big")
                except ValueError:
                    raise ValueError(f"Code '{code}' requires a numerical value.")
                size_values[code] = numeric_value
            else:
                if isinstance(length_info, tuple):
                    variable_code, fixed_length = length_info
                    if variable_code not in size_values:
                        continue
                    size = size_values[variable_code] * fixed_length
                else:
                    size = length_info
                encoded_value = self.encode_value(value, size)

            encoded_data_temp[code] = encoded_value
            bit_position = int(code, hex_to_dec())
            binary_map[bit_position // byte_to_bit()] |= 1 << (bit_position % byte_to_bit())

        encoded_full_data = binary_map
        for code in encoded_data_temp:
            encoded_full_data.extend(encoded_data_temp[code])
        return encoded_full_data

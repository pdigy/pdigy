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
        remove_whitespace=False,
        patch_size=1024,
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
        self.patch_jpeg_quality = patch_jpeg_quality
        self.whitespace_preview_size = whitespace_preview_size
        self.whitespace_pixel_threshold = whitespace_pixel_threshold
        self.whitespace_ratio_threshold = whitespace_ratio_threshold
        self.whitespace_mean_intensity_threshold = whitespace_mean_intensity_threshold
        self.whitespace_std_intensity_threshold = whitespace_std_intensity_threshold
        if self.image_path:
            # Check for unsupported formats before processing
            if "i2syntax" in self.image_path.lower():
                raise NotImplementedError(
                    f"Cannot convert i2syntax file directly: {self.image_path}\n"
                    "Philips i2syntax (UFS) format requires conversion first.\n"
                    "Please convert to SVS/TIFF using Philips Image Management System "
                )
            
            self.full_image_data = self._load_image_data()
            self.thumbnail_data = self._create_thumbnail()
            self.full_image_size_hex = self._size_to_hex(
                len(self.full_image_data), 10)
            self.thumbnail_size_hex = self._size_to_hex(
                len(self.thumbnail_data), 6)
            self.n_patches = 0
            self.patches = None
            self.filter_map = None
            self._extract_patches()
        self.compression = True
        map_path = os.path.join(os.path.dirname(__file__), "PDigyMap.xlsx")
        map_df = pd.read_excel(map_path)
        self.map_info = self.parse_map_dataframe(map_df)
        if meta_path:
            # Read the new Excel sheets again after the reset
            self.meta_df = pd.read_excel(meta_path)
        
    def _load_image_data(self):
        with open(self.image_path, "rb") as image_file:
            return image_file.read()

    def _read_svs_metadata(self, svs_file_path):
        slide = openslide.OpenSlide(svs_file_path)
        metadata = slide.properties
        # Read DPI information
        # micrometres per pixel in X
        dpi_x = metadata.get(openslide.PROPERTY_NAME_MPP_X)
        # micrometres per pixel in Y
        dpi_y = metadata.get(openslide.PROPERTY_NAME_MPP_Y)
        # Convert micrometres per pixel to DPI 
        if dpi_x and dpi_y:
            dpi_x = mum_to_inch() / float(dpi_x) 
            dpi_y = mum_to_inch() / float(dpi_y)
        else:
            dpi_x, dpi_y = None, None
        return dpi_x, dpi_y, metadata

    def _load_svs_image_data(self):
        # Open the SVS file
        slide = openslide.OpenSlide(self.image_path)

        # Get the level count and select the highest resolution level (level 0)
        level = 0  # highest resolution
        level_dimension = slide.level_dimensions[level]

        # Read the image at the highest resolution
        highest_res_image = slide.read_region((0, 0), level, level_dimension)
        return highest_res_image

    def _create_thumbnail(self):
        thumbnail_max_size = (1024, 1024)
        if "svs" in self.image_path:
            slide = openslide.OpenSlide(self.image_path)

            if "thumbnail" in slide.associated_images:
                img = slide.associated_images["thumbnail"]
            else:
                img = slide.get_thumbnail(thumbnail_max_size)

            if img.mode != "RGB":
                img = img.convert("RGB")

            img.thumbnail(thumbnail_max_size)
            with io.BytesIO() as thumb_io:
                img.save(thumb_io, format="JPEG")
                return thumb_io.getvalue()
        else:
            with Image.open(self.image_path) as img:
                img.thumbnail(thumbnail_max_size)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                with io.BytesIO() as thumb_io:
                    img.save(thumb_io, format="JPEG")
                    return thumb_io.getvalue()
   
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

        # Convert to 16-bit hex (2 bytes each)
        n_patches_x_hex = self._size_to_hex(
            n_patches_x, 4)  # 4 hex digits for 16 bits
        n_patches_y_hex = self._size_to_hex(
            n_patches_y, 4)  # 4 hex digits for 16 bits

        # Append patch dimensions to file content
        file_content += bytearray.fromhex(n_patches_x_hex)
        file_content += bytearray.fromhex(n_patches_y_hex)
        # Serialize and compress patches
        patch_payload = {
            "patches": self.patches,
            "filter_map": self.filter_map.astype(np.uint8),
            "patch_modes": self.patch_modes.astype(np.uint8),
            "patch_size": self.patch_size,
            "whitespace_preview_size": self.whitespace_preview_size,
        }
        serialized_patches = pickle.dumps(patch_payload)
        if self.compression == True:
            compressed_patches = zipper.compress(serialized_patches)
        else:
            compressed_patches = serialized_patches

        # Include the size of compressed patches (in hexadecimal format)
        compressed_patches_size_hex = self._size_to_hex(
            len(compressed_patches), 8
        )  # 8 hex digits for 32 bits
        file_content += bytearray.fromhex(compressed_patches_size_hex)
        encoded_metadata = self.encode_meta(self.meta_df)
        # Append compressed patch data
        file_content += compressed_patches
        file_content += encoded_metadata
        print(self.map_info)

        return file_content

    def _is_whitespace_patch(self, patch, max_pixel_value):
        if not self.remove_whitespace:
            return False

        rgb_patch = patch[..., :3]
        white_threshold = max_pixel_value * self.whitespace_pixel_threshold
        white_pixels = np.all(rgb_patch >= white_threshold, axis=-1)
        if float(np.mean(white_pixels)) >= self.whitespace_ratio_threshold:
            return True

        grayscale_patch = rgb_patch.mean(axis=-1)
        mean_intensity = float(np.mean(grayscale_patch))
        std_intensity = float(np.std(grayscale_patch))
        return (
            mean_intensity >= max_pixel_value * self.whitespace_mean_intensity_threshold
            and std_intensity <= max_pixel_value * self.whitespace_std_intensity_threshold
        )

    def _encode_patch(self, patch, is_whitespace_patch):
        patch_image = Image.fromarray(patch)
        if patch_image.mode == "RGBA":
            patch_image = patch_image.convert("RGB")

        if is_whitespace_patch:
            resampling_module = getattr(Image, "Resampling", Image)
            bilinear_resample = getattr(resampling_module, "BILINEAR", 2)
            patch_image = patch_image.resize(
                (self.whitespace_preview_size, self.whitespace_preview_size),
                resample=bilinear_resample,
            )

        with io.BytesIO() as patch_io:
            patch_image.save(patch_io, format="JPEG", quality=self.patch_jpeg_quality, subsampling=0)
            return patch_io.getvalue()

    def _extract_patches(self, threshold=0.0, maximum_size=1000000):
        if "svs" in self.image_path:
            img = np.array(self._load_svs_image_data())
        else:
            with Image.open(self.image_path) as img:
                img = np.array(img)

        self.patches = []
        count_pass = 0
        count_fail = 0
        max_pixel_value = color_8bit() if np.max(img) > 1.0 else 1.0
        print("Maximum pixel value is estimated to be: ", max_pixel_value)

        h, w, c = img.shape
        dimy = int(np.ceil(w / self.patch_size))
        dimx = int(np.ceil(h / self.patch_size))
        self.filter_map = np.zeros([dimx, dimy], dtype=np.uint8)
        self.patch_modes = np.zeros([dimx, dimy], dtype=np.uint8)
        print("img.shape", img.shape)

        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img[i: i + self.patch_size, j: j + self.patch_size, :]
                patch_row = i // self.patch_size
                patch_col = j // self.patch_size
                is_whitespace_patch = self._is_whitespace_patch(patch, max_pixel_value)
                patch_data = self._encode_patch(patch, is_whitespace_patch)
                self.patches.append(patch_data)
                self.filter_map[patch_row, patch_col] = 1
                self.patch_modes[patch_row, patch_col] = 2 if is_whitespace_patch else 1

                if is_whitespace_patch:
                    count_fail += 1
                else:
                    count_pass += 1

        print(
            "pass: ",
            count_pass,
            " fail: ",
            count_fail,
            " efficiency: ",
            count_pass / (count_pass + count_fail),
        )
        print(dimx, dimy)

    def save_to_file(self, file_path):
        file_content = self._construct_file_content()
        with open(file_path + ".pdigy", "wb") as file:
            file.write(file_content)

    def encode_value(self, value, size):
        if len(value) > size:
            raise ValueError(
                f"Value '{value}' exceeds the specified size of {size} bytes."
            )

        # Padding the value with zeros if it's shorter than the specified size
        padded_value = value.rjust(size, "\0")
        return padded_value.encode()

    def _normalize_code(self, code):
        normalized_code = str(code).strip().strip('"')
        if self.remove_whitespace:
            normalized_code = "".join(normalized_code.split())
        return normalized_code

    def _get_dataframe_column(self, df, *candidates):
        normalized_columns = {
            str(column).strip().casefold(): column for column in df.columns
        }
        for candidate in candidates:
            column = normalized_columns.get(candidate.strip().casefold())
            if column is not None:
                return column
        raise KeyError(
            f"Missing expected column. Tried {candidates}, found {list(df.columns)}"
        )

    # Function to parse the map dataframe and extract necessary information
    def parse_map_dataframe(self, df):
        code_column = self._get_dataframe_column(df, "Code(in Hex)", "Code (Hex)")
        name_column = self._get_dataframe_column(df, "Name")
        length_column = self._get_dataframe_column(df, "Length(B)", "Length (B)")
        description_column = self._get_dataframe_column(df, "Description")
        note_column = self._get_dataframe_column(df, "Note")
        mapping = {}
        for _, row in df.iterrows():
            # Ensure that the 'Code' is a string and strip any quotation marks
            code = self._normalize_code(row[code_column])

            # Extract other information, handling potential format issues
            name = str(row[name_column]) if pd.notna(row[name_column]) else None
            length = row[length_column]
            description = (
                str(row[description_column]) if pd.notna(
                    row[description_column]) else None
            )
            note = str(row[note_column]) if pd.notna(row[note_column]) else None

            if "*" in str(length):
                variable_code, fixed_length = length.split("*")
                variable_code = self._normalize_code(variable_code)
                fixed_length = int(fixed_length)
                mapping[code] = {
                    "name": name,
                    "length": (variable_code, fixed_length),
                    "description": description,
                    "note": note,
                }
            else:
                length = int(length) if pd.notna(length) else None
                mapping[code] = {
                    "name": name,
                    "length": length,
                    "description": description,
                    "note": note,
                }
        return mapping

    def encode_meta(self, input_df):
        # Initialize a binary map of 4096 bits with all zeros
        binary_map = bytearray(512)
        encoded_data_temp = {}  # Dictionary to hold encoded data temporarily
        # Temporary dictionary to store values of codes which determine sizes of other codes
        size_values = {}

        for _, row in input_df.iterrows():
            code = self._normalize_code(row["Code"])
            value = row["Value"].strip('"')
            #print("values", code, value)

            if code in self.map_info:
                map_entry = self.map_info[code]
                #print("Map entry: ", map_entry, " Code: ", code)
                length_info = map_entry["length"]

                # Special handling for 'Calibration' data
                # Handle numerical values correctly for size determining codes
                if "len" in map_entry["name"]:
                    try:
                        # Convert value to int and then to a binary string, padded to the correct length
                        numeric_value = int(value)
                        encoded_value = numeric_value.to_bytes(
                            length_info, byteorder="big"
                        )
                        #print("encoded val", encoded_value)
                    except ValueError:
                        raise ValueError(
                            f"Code '{code}' requires a numerical value.")
                    size_values[code] = numeric_value
                else:
                    # Check if length is a variable size
                    if isinstance(length_info, tuple):
                        variable_code, fixed_length = length_info
                        if variable_code in size_values:
                            size = size_values[variable_code] * fixed_length
                        else:
                            continue  # Skip processing this code until the variable code is processed
                    else:
                        size = length_info
                    encoded_value = self.encode_value(value, size)
                encoded_data_temp[code] = encoded_value
                # Update the binary map: setting the bit corresponding to the code to 1
                bit_position = int(code, hex_to_dec())
                byte_index = bit_position // byte_to_bit()
                bit_index = bit_position % byte_to_bit()
                binary_map[byte_index] |= 1 << bit_index
                #print(binary_map)
                #print(np.shape(binary_map))
            else:
                raise ValueError(f"Code '{code}' not found in the map.")
        # Concatenate the binary map and the encoded data
        encoded_full_data = binary_map
        for code in encoded_data_temp:
            #print("samples: ", encoded_data_temp)
            #print(np.shape(encoded_data_temp))
            encoded_full_data.extend(encoded_data_temp[code])
            #print(encoded_full_data)
        return encoded_full_data

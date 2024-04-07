import io
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import logging
import openslide
import lzma as zipper
from constants import *


class pdigy:
    """
    Pdigy encoder class for processing and encoding medical image data with
    metadata handling.

    Attributes:
        image_path (str): Path to the image file.
        meta_path (str): Path to the metadata file.
        map_word (dict): Mapping of word headers.
        full_image_data (bytes): Raw image data.
        thumbnail_data (bytes): Compressed thumbnail data.
        full_image_size_hex (str): Hexadecimal string representing the size of
    the full image data.
        thumbnail_size_hex (str): Hexadecimal string representing the size of
    the thumbnail data.
        n_patches (int): Number of image patches.
        patches (list): List of image patches.
        filter_map (numpy.ndarray): Filter map for patch extraction.
        compression (bool): Flag indicating if compression is applied.
        map_info (dict): Information mapping from a dataframe.
        meta_df (pandas.DataFrame): DataFrame containing metadata.
    """
    def __init__(self, image_path=None, meta_path=None):
        """
        Initializes the Pdigy object with image and meta data paths.

        Parameters:
            image_path (str, optional): Path to the image file. Defaults to None.
            meta_path (str, optional): Path to the metadata Excel file. Defaults to None.
        """
        self.map_word = header()
        self.image_path = image_path
        self.full_image_data = self._load_image_data() if image_path else None
        self.thumbnail_data = self._create_thumbnail() if image_path else None
        self.full_image_size_hex = self._size_to_hex(len(self.full_image_data), 10) if image_path else None
        print(self.full_image_size_hex)
        self.thumbnail_size_hex = self._size_to_hex(len(self.thumbnail_data), 6) if image_path else None
        self.n_patches = 0
        self.patches = []
        self.filter_map = None
        self.compression = True
        self.map_info = self.parse_map_dataframe(pd.read_excel(map_path()))
        if meta_path:
            self.meta_df = pd.read_excel(meta_path)
            self._extract_patches()  # Extract patches if image path is provided and valid.

    def _load_image_data(self):
        """
        Loads image data from the file specified by the image path.

        Returns:
            bytes: The raw image data.
        """
        try:
            with open(self.image_path, "rb") as image_file:
                return image_file.read()
        except IOError as e:
            logging.error(f"Failed to load image data: {e}")
            return None

    def _create_thumbnail(self):
        """
        Creates a thumbnail of the loaded image.

        Returns:
            bytes: The compressed thumbnail data.
        """
        try:
            if "svs" in self.image_path:
                img = self._load_svs_image_data()
                dpi = img.info.get("dpi", (72, 72))
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.thumbnail((img.size[0] // 10, img.size[1] // 10), Image.LANCZOS)
            else:
                with Image.open(self.image_path) as img:
                    img.thumbnail((img.size[0] // 10, img.size[1] // 10), Image.LANCZOS)

            with io.BytesIO() as thumb_io:
                img.save(thumb_io, format="JPEG")
                return thumb_io.getvalue()
        except Exception as e:
            logging.error(f"Failed to create thumbnail: {e}")
            return None

    def _size_to_hex(self, size, hex_length):
        """
        Converts a size value to a hexadecimal string of specified length.

        Parameters:
            size (int): The size value to convert.
            hex_length (int): The desired length of the hexadecimal string.

        Returns:
            str: The hexadecimal representation of the size value, padded to `hex_length`.
        """
        return format(size, f"0{hex_length}X")

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
        serialized_patches = pickle.dumps(self.patches)
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

    def _extract_patches(self, patch_size=1024, threshold=0.0, maximum_size=1000000):
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
        dimy = int(np.ceil(w / patch_size))
        dimx = int(np.ceil(h / patch_size))
        self.filter_map = np.zeros([dimx, dimy])
        print("img.shape", img.shape)

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img[i: i + patch_size, j: j + patch_size, :]
                # Remove fat cells and white spaces
                if np.mean(patch) < max_pixel_value * (1 - threshold):
                    # Serialize each patch as JPEG
                    with io.BytesIO() as patch_io:
                        patch_image = Image.fromarray(patch)
                        if patch_image.mode == "RGBA":
                            patch_image = patch_image.convert("RGB")
                        patch_image.save(patch_io, format="JPEG", quality=65)
                        patch_data = patch_io.getvalue()
                        self.patches.append(patch_data)

                    count_pass += 1
                else:
                    count_fail += 1

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

    def _encode_value(self, value, size):
        """
        Encodes a value with padding to fit the specified size.

        Parameters:
            value (str): The value to encode.
            size (int): The size to pad the value to.

        Returns:
            bytes: The encoded value as bytes.
        """
        if len(value) > size:
            logging.error(f"Value '{value}' exceeds the specified size of {size} bytes.")
            return None
        return value.ljust(size, "\0").encode()

    def parse_map_dataframe(self, df):
        """
        Parses the DataFrame containing mapping information.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the map information.

        Returns:
            dict: A dictionary representation of the map information.
        """
        mapping = {}
        for _, row in df.iterrows():
            # Ensure that the 'Code' is a string and strip any quotation marks
            code = str(row["Code"]).strip('"')

            # Extract other information, handling potential format issues
            name = str(row["Name"]) if pd.notna(row["Name"]) else None
            length = row["Length(B)"]
            description = (
                str(row["Description"]) if pd.notna(
                    row["Description"]) else None
            )
            note = str(row["Note"]) if pd.notna(row["Note"]) else None

            if "*" in str(length):
                variable_code, fixed_length = length.split("*")
                variable_code = variable_code.strip('"')
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

    def _update_binary_map(self, binary_map, code):
        """
        Updates the binary map to indicate the presence of a code.

        Parameters:
            binary_map (bytearray): The binary map to be updated.
            code (str): The code to update in the binary map.
        """
        bit_position = int(code, hex_to_dec())
        byte_index = bit_position // byte_to_bit()
        bit_index = bit_position % byte_to_bit()
        binary_map[byte_index] |= 1 << bit_index

    def _encode_value_based_on_length(self, value, length_info, size_values):
        """
        Encodes a value based on its specified length information.

        Parameters:
            value (str): The value to be encoded.
            length_info (tuple or int): Length information for the value, either as an integer or a tuple for variable lengths.
            size_values (dict): Dictionary storing values of codes that determine the sizes of other codes.

        Returns:
            bytes: The encoded value as bytes, or None if the value cannot be encoded.
        """
        try:
            if isinstance(length_info, tuple):
                variable_code, fixed_length = length_info
                size = size_values.get(variable_code, 0) * fixed_length
            else:
                size = length_info
            return self._encode_value(value, size)
        except Exception as e:
            logging.error(f"Error encoding value '{value}': {e}")
            return None

    def _concatenate_encoded_data(self, binary_map, encoded_data_temp):
        """
        Concatenates the binary map with encoded data into a single byte array.

        Parameters:
            binary_map (bytearray): The binary map indicating which codes are present.
            encoded_data_temp (dict): The temporary dictionary holding encoded values.

        Returns:
            bytes: The concatenated binary map and encoded data as a single byte array.
        """
        encoded_full_data = binary_map
        for code, data in encoded_data_temp.items():
            encoded_full_data.extend(data)
        return bytes(encoded_full_data)

    def encode_meta(self, input_df):
        """
        Encodes metadata into a binary format based on the mapping information.

        Parameters:
            input_df (pandas.DataFrame): The DataFrame containing metadata to be encoded.

        Returns:
            bytes: Encoded metadata as a byte array.
        """
        binary_map = bytearray(512)
        encoded_data_temp = {}  # Dictionary to hold encoded data temporarily
        # Temporary dictionary to store values of codes which determine sizes of other codes
        size_values = {}

        for _, row in input_df.iterrows():
            code = row["Code"].strip('"')
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
                    encoded_value = self._encode_value(value, size)
                encoded_data_temp[code] = encoded_value
                self._update_binary_map(binary_map, code)
            else:
                raise ValueError(f"Code '{code}' not found in the map.")
        return self._concatenate_encoded_data(binary_map, encoded_data_temp)
from PIL import Image
import io, os
import numpy as np
import pickle
from constants import *
import lzma as zipper
import pandas as pd
import struct
import openslide

class pdigy:
    
    def get_word_map(self):
        return self.map_word        
    def __init__(self, image_path):
        self.map_word = header()
        self.image_path = image_path
        self.full_image_data = self._load_image_data()
        self.thumbnail_data = self._create_thumbnail()
        self.full_image_size_hex = self._size_to_hex(len(self.full_image_data), 10)
        self.thumbnail_size_hex = self._size_to_hex(len(self.thumbnail_data), 6)
        self.n_patches = 0
        self.patches = None
        self.filter_map = None
        self._extract_patches()
        self.compression = True
    
    def _load_image_data(self):
        with open(self.image_path, 'rb') as image_file:
            return image_file.read()

    def _read_svs_metadata(self, svs_file_path):
        slide = openslide.OpenSlide(svs_file_path)
        metadata = slide.properties
        # Read DPI information
        dpi_x = metadata.get(openslide.PROPERTY_NAME_MPP_X)  # micrometres per pixel in X
        dpi_y = metadata.get(openslide.PROPERTY_NAME_MPP_Y)  # micrometres per pixel in Y
    
        # Convert micrometres per pixel to DPI (if possible)
        if dpi_x and dpi_y:
            dpi_x = 25400 / float(dpi_x)  # 25400 micrometres per inch
            dpi_y = 25400 / float(dpi_y)
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
        if( 'svs' in self.image_path):
            img = self._load_svs_image_data()
            dpi = img.info.get('dpi', (72, 72))  # Default to 72 DPI if not found - this seems a bit inefficient, we will need to find a better solution 
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_format = img.format
            print(dpi, image_format)
            img.thumbnail((np.shape(img)[0]//10, np.shape(img)[1]//10))
            with io.BytesIO() as thumb_io:
                img.save(thumb_io, format='JPEG')
                return thumb_io.getvalue()
        else:
            with Image.open(self.image_path) as img:
                img.thumbnail((np.shape(img)[0]//10, np.shape(img)[1]//10))
                with io.BytesIO() as thumb_io:
                    img.save(thumb_io, format='JPEG')
                    return thumb_io.getvalue()

    def _size_to_hex(self, size, hex_length):
        return format(size, f'0{hex_length}X')

    def _construct_file_content(self):
        file_content = b''
        for word in self.map_word.values():
            file_content += bytes.fromhex(word[0])
        
        file_content += bytearray.fromhex(self.thumbnail_size_hex)
        file_content += self.thumbnail_data
        
        n_patches_x = int(np.ceil(self.filter_map.shape[1]))
        n_patches_y = int(np.ceil(self.filter_map.shape[0]))

        # Convert to 16-bit hex (2 bytes each)
        n_patches_x_hex = self._size_to_hex(n_patches_x, 4)  # 4 hex digits for 16 bits
        n_patches_y_hex = self._size_to_hex(n_patches_y, 4)  # 4 hex digits for 16 bits

        # Append patch dimensions to file content
        file_content += bytearray.fromhex(n_patches_x_hex)
        file_content += bytearray.fromhex(n_patches_y_hex)
        # Serialize and compress patches
        serialized_patches = pickle.dumps(self.patches)
        if(self.compression == True):
            compressed_patches = zipper.compress(serialized_patches)
        else: 
            compressed_patches = serialized_patches

        # Include the size of compressed patches (in hexadecimal format)
        compressed_patches_size_hex = self._size_to_hex(len(compressed_patches), 8)  # 8 hex digits for 32 bits
        file_content += bytearray.fromhex(compressed_patches_size_hex)
        
        # Append compressed patch data
        file_content += compressed_patches
        return file_content

    def _extract_patches(self, patch_size=1024, threshold=0.0, maximum_size=1000000):
        if( 'svs' in self.image_path):
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
                patch = img[i:i + patch_size, j:j + patch_size, :]
                if np.mean(patch) < max_pixel_value * (1 - threshold):  # Remove fat cells and white spaces
                    # Serialize each patch as JPEG
                    with io.BytesIO() as patch_io:
                        patch_image = Image.fromarray(patch)
                        if patch_image.mode == 'RGBA':
                            patch_image = patch_image.convert('RGB')
                        patch_image.save(patch_io, format='JPEG', quality=65)
                        patch_data = patch_io.getvalue()
                        self.patches.append(patch_data)

                    count_pass += 1
                else:
                    count_fail += 1

        print("pass: ", count_pass, " fail: ", count_fail, " efficiency: ", count_pass / (count_pass + count_fail))
        print(dimx, dimy)    
    def save_to_file(self, file_path):
        file_content = self._construct_file_content()
        with open(file_path+".pdigy", 'wb') as file:
            file.write(file_content)
    """ EXPERIMENTAL FEATURES BELOW - WILL BE INTEGRATED IN v0.6 """
    
    def encode_value(self, value, size):
        if len(value) > size:
            raise ValueError(f"Value '{value}' exceeds the specified size of {size} bytes.")
    
        # Padding the value with zeros if it's shorter than the specified size
        padded_value = value.rjust(size, '\0')
        return padded_value.encode()
    
    # Function to parse the map dataframe and extract necessary information
    def parse_map_dataframe(self, df):
        mapping = {}
        for _, row in df.iterrows():
            # Ensure that the 'Code' is a string and strip any quotation marks
            code = str(row['Code']).strip('"')
    
            # Extract other information, handling potential format issues
            name = str(row['Name']) if pd.notna(row['Name']) else None
            length = row['Length(B)']
            description = str(row['Description']) if pd.notna(row['Description']) else None
            note = str(row['Note']) if pd.notna(row['Note']) else None
    
            if '*' in str(length):
                variable_code, fixed_length = length.split('*')
                variable_code = variable_code.strip('"')
                fixed_length = int(fixed_length)
                mapping[code] = {'name': name, 'length': (variable_code, fixed_length), 'description': description, 'note': note}
            else:
                length = int(length) if pd.notna(length) else None
                mapping[code] = {'name': name, 'length': length, 'description': description, 'note': note}
        return mapping
    def encode_meta(self, example_df, map_info):
        binary_map = bytearray(512)  # Initialize a binary map of 4096 bits with all zeros
        encoded_data_temp = {}       # Dictionary to hold encoded data temporarily
        size_values = {}             # Temporary dictionary to store values of codes which determine sizes of other codes
    
        for _, row in example_df.iterrows():
            code = row['Code'].strip('"')
            value = row['Value'].strip('"')
            print("values", code, value)
            
            if code in map_info:
                map_entry = map_info[code]
                length_info = map_entry['length']
    
                # Special handling for 'Calibration' data
                if code == '010':
                    # Remove curly braces and split the value into individual words - these may look like json but they are just tuples 
                    words = value.split(',')
                    print(words)
                    encoded_words = []
                    for word in words:
                        word = word.strip().strip('{').strip('}').strip('"')  # Removing extra spaces and quotes
                        padded_word = encode_value(word, 32)  # Assuming each word should be padded to 32 bytes
                        encoded_words.append(padded_word)
                    encoded_value = b''.join(encoded_words)
                    print(encoded_value)
                else:
                    # Handle numerical values correctly for size determining codes
                    if 'len' in map_entry['name']:
                        try:
                            # Convert value to int and then to a binary string, padded to the correct length
                            numeric_value = int(value)
                            encoded_value = numeric_value.to_bytes(length_info, byteorder='big')
                            print("encoded val", encoded_value)
                        except ValueError:
                            raise ValueError(f"Code '{code}' requires a numerical value.")
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
                        encoded_value = encode_value(value, size)
    
                encoded_data_temp[code] = encoded_value
    
                # Update the binary map: setting the bit corresponding to the code to 1
                bit_position = int(code, 16)
                byte_index = bit_position // 8
                bit_index = bit_position % 8
                binary_map[byte_index] |= 1 << bit_index
                print(binary_map)
                print(np.shape(binary_map))
            else:
                raise ValueError(f"Code '{code}' not found in the map.")
    
        # Concatenate the binary map and the encoded data
        encoded_full_data = binary_map
        for code in encoded_data_temp:
            print("samples: ", encoded_data_temp)
            print(np.shape(encoded_data_temp))
            encoded_full_data.extend(encoded_data_temp[code])
            print(encoded_full_data)
    
        return encoded_full_data


![Alt text](docs/pdigy_animated.gif?raw=true "Title")

# pdigy

A pathology digital image compression and storage format for whole slide images (WSI).

<!-- ## Installation

Install pdigy using pip:

```bash
pip install pdigy
``` -->

### Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/pdigy/pdigy.git
cd pdigy
pip install -e .
```

To install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Prerequisites

- Python >= 3.7
- PIL (Pillow)
- numpy
- pandas
- openslide-python (for SVS file support)
- openpyxl
- jupyter-notebook (for running examples, optional)

## Usage

```python
from pdigy import pdigy, pdigyDecoder

# Encode an image
image_path = 'path/to/image.svs'
example_path = "PDigyExample.xlsx"
output_path = 'output'

pdigy_file = pdigy(image_path, example_path)
pdigy_file.save_to_file(output_path)

# Decode a pdigy file
file_path = 'output.pdigy'
decoder = pdigyDecoder(file_path)
decoder.save_images('thumbnail.JPEG')
```

## Example

See `PDigyTest.ipynb` for a complete example using TCGA database files.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

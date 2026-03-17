"""
pdigy - A pathology digital image compression and storage format
"""

from .encoder import pdigy
from .decoder import pdigyDecoder
from .constants import *

__version__ = "1.0a"
__all__ = ["pdigy", "pdigyDecoder"]

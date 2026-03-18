"""
test_helpers.py — all the plumbing so PDigyTest.ipynb stays readable.
I put everything here so the notebook can just think about science.
"""
import io
from datetime import datetime, timezone

import numpy as np
import tifffile
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    JPEGBaseline8Bit,
    JPEG2000Lossless,
    PYDICOM_IMPLEMENTATION_UID,
    VLWholeSlideMicroscopyImageStorage,
    generate_uid,
)


def ensure_rgb_uint8(image):
    # I refuse to deal with RGBA at 2am
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image.convert("RGB")


def crop_white_border(image, threshold=245):
    # Trim the white fat
    rgb = np.asarray(ensure_rgb_uint8(image), dtype=np.uint8)
    non_white = np.any(rgb < threshold, axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return ensure_rgb_uint8(image)
    return ensure_rgb_uint8(image).crop(
        (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    )


def save_lossless_tiff(image, file_path, tile_size=512):
    # BigTIFF: because regular TIFF couldn't handle the drip
    tifffile.imwrite(
        str(file_path),
        np.asarray(ensure_rgb_uint8(image), dtype=np.uint8),
        bigtiff=True, photometric="rgb",
        compression="zlib", predictor=True,
        tile=(tile_size, tile_size),
    )


def save_lossless_svs(image, file_path, tile_size=512):
    # SVS is just TIFF with a lab coat
    tifffile.imwrite(
        str(file_path),
        np.asarray(ensure_rgb_uint8(image), dtype=np.uint8),
        bigtiff=True, photometric="rgb",
        compression="zlib", predictor=True,
        tile=(tile_size, tile_size),
    )


def iter_padded_tiles(image, tile_size):
    # I tile, therefore I am
    rgb = ensure_rgb_uint8(image)
    width, height = rgb.size
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            tile = Image.new("RGB", (tile_size, tile_size), color="white")
            tile.paste(
                rgb.crop((left, top, min(left + tile_size, width), min(top + tile_size, height))),
                (0, 0),
            )
            yield tile


def encode_jpeg_tile(tile, quality=100):
    buf = io.BytesIO()
    tile.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def encode_j2k_lossless_tile(tile):
    # What you compressed is exactly what you get back — revolutionary
    buf = io.BytesIO()
    tile.save(buf, format="JPEG2000", irreversible=False)
    return buf.getvalue()


def _build_dicom(file_path, transfer_syntax, width, height, tile_size, frames, lossy):
    """I share all the boilerplate DICOM tags so the two save functions don't have to."""
    now = datetime.now(timezone.utc)

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.MediaStorageSOPClassUID = VLWholeSlideMicroscopyImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID              = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID           = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID         = generate_uid()
    ds.SeriesInstanceUID        = generate_uid()
    ds.FrameOfReferenceUID      = generate_uid()
    ds.Modality                 = "SM"
    ds.ImageType                = ["DERIVED", "PRIMARY", "VOLUME", "NONE"]
    ds.InstanceNumber           = 1
    ds.PatientName              = "PDigy^Notebook"
    ds.PatientID                = "PDIGYTEST"
    ds.StudyID                  = "1"
    ds.SeriesNumber             = 1
    ds.ContentDate              = now.strftime("%Y%m%d")
    ds.ContentTime              = now.strftime("%H%M%S.%f")
    ds.AcquisitionDateTime      = now.strftime("%Y%m%d%H%M%S.%f")
    ds.Rows                     = tile_size
    ds.Columns                  = tile_size
    ds.TotalPixelMatrixRows        = height
    ds.TotalPixelMatrixColumns     = width
    ds.TotalPixelMatrixFocalPlanes = 1
    ds.NumberOfOpticalPaths     = 1
    ds.NumberOfFrames           = len(frames)
    ds.DimensionOrganizationType = "TILED_FULL"
    ds.SamplesPerPixel          = 3
    ds.PlanarConfiguration      = 0
    ds.BitsAllocated            = 8
    ds.BitsStored               = 8
    ds.HighBit                  = 7
    ds.PixelRepresentation      = 0
    ds.BurnedInAnnotation       = "NO"
    ds.SpecimenLabelInImage     = "NO"
    ds.ImageOrientationSlide    = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    if lossy:
        compressed_size = max(sum(len(f) for f in frames), 1)
        ds.PhotometricInterpretation  = "YBR_FULL_422"
        ds.LossyImageCompression      = "01"
        ds.LossyImageCompressionMethod = "ISO_10918_1"
        ds.LossyImageCompressionRatio  = f"{width * height * 3 / compressed_size:.4f}"
    else:
        ds.PhotometricInterpretation = "RGB"
        ds.LossyImageCompression     = "00"

    dim_org = Dataset()
    dim_org.DimensionOrganizationUID = generate_uid()
    ds.DimensionOrganizationSequence = [dim_org]

    optical = Dataset()
    optical.OpticalPathIdentifier = "1"
    ds.OpticalPathSequence = [optical]

    origin = Dataset()
    origin.XOffsetInSlideCoordinateSystem = 0.0
    origin.YOffsetInSlideCoordinateSystem = 0.0
    origin.ZOffsetInSlideCoordinateSystem = 0.0
    ds.TotalPixelMatrixOriginSequence = [origin]

    px = Dataset()
    px.PixelSpacing            = [1.0, 1.0]
    px.SliceThickness          = 1.0
    px.SpacingBetweenSlices    = 1.0
    sfg = Dataset()
    sfg.PixelMeasuresSequence  = [px]
    ds.SharedFunctionalGroupsSequence = [sfg]

    ds.PixelData = encapsulate(frames)
    ds["PixelData"].is_undefined_length = True
    ds.save_as(str(file_path), write_like_original=False)


def save_tiled_dicom(image, file_path, tile_size=512, jpeg_quality=100):
    rgb = ensure_rgb_uint8(image)
    frames = [encode_jpeg_tile(t, quality=jpeg_quality) for t in iter_padded_tiles(rgb, tile_size)]
    _build_dicom(file_path, JPEGBaseline8Bit, rgb.size[0], rgb.size[1], tile_size, frames, lossy=True)


def save_lossless_dicom(image, file_path, tile_size=512):
    rgb = ensure_rgb_uint8(image)
    frames = [encode_j2k_lossless_tile(t) for t in iter_padded_tiles(rgb, tile_size)]
    _build_dicom(file_path, JPEG2000Lossless, rgb.size[0], rgb.size[1], tile_size, frames, lossy=False)


def reconstruct_tissue_only(dec):
    """Show only what was actually stored — whitespace gets the void it deserves."""
    pw = dec.patch_size
    dimy, dimx = dec.patch_modes.shape
    canvas = Image.new("RGB", (dimx * pw, dimy * pw), color=(255, 255, 255))
    patch_index = 0
    for i in range(dimy):
        for j in range(dimx):
            if dec.filter_map is not None and not dec.filter_map[i, j]:
                continue
            if dec.patch_modes[i, j] == 1:
                canvas.paste(
                    Image.open(io.BytesIO(dec.patches[patch_index])).convert("RGB"),
                    (j * pw, i * pw),
                )
            patch_index += 1
    return canvas


# ── Benchmark helpers ────────────────────────────────────────────────────────

import time as _time


def time_fn(fn, repeats=5):
    """I clock the same function N times and return the best run — racing, not averaging."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = _time.perf_counter()
        result = fn()
        times.append(_time.perf_counter() - t0)
    return min(times), result


def make_bench_tasks(tiff_path, svs_path, dicom_path, dicom_lossless_path,
                     patch_size, generated_files):
    """I build all benchmark callables so the notebook can stay blissfully function-free."""
    import openslide
    import pydicom
    from pydicom.encaps import decode_data_sequence
    from pdigy import pdigyDecoder as _Dec

    def _patches_tiff():
        with Image.open(str(tiff_path)) as img:
            w, h = img.size
            return [img.crop((l, t, min(l + patch_size, w), min(t + patch_size, h)))
                    for t in range(0, h, patch_size) for l in range(0, w, patch_size)]

    def _thumb_tiff():
        with Image.open(str(tiff_path)) as img:
            thumb = img.copy(); thumb.thumbnail((256, 256)); return thumb

    def _patches_svs():
        slide = openslide.OpenSlide(str(svs_path))
        w, h = slide.dimensions
        patches = [slide.read_region((l, t), 0, (min(patch_size, w - l), min(patch_size, h - t)))
                   for t in range(0, h, patch_size) for l in range(0, w, patch_size)]
        slide.close(); return patches

    def _thumb_svs():
        slide = openslide.OpenSlide(str(svs_path))
        thumb = slide.get_thumbnail((256, 256)); slide.close(); return thumb

    def _patches_dicom():
        ds = pydicom.dcmread(str(dicom_path))
        return [Image.open(io.BytesIO(f)) for f in decode_data_sequence(ds.PixelData)]

    def _thumb_dicom():
        ds = pydicom.dcmread(str(dicom_path))
        thumb = Image.open(io.BytesIO(next(iter(decode_data_sequence(ds.PixelData)))))
        thumb.thumbnail((256, 256)); return thumb

    def _patches_dicom_lossless():
        ds = pydicom.dcmread(str(dicom_lossless_path))
        return [Image.open(io.BytesIO(f)) for f in decode_data_sequence(ds.PixelData)]

    def _thumb_dicom_lossless():
        ds = pydicom.dcmread(str(dicom_lossless_path))
        thumb = Image.open(io.BytesIO(next(iter(decode_data_sequence(ds.PixelData)))))
        thumb.thumbnail((256, 256)); return thumb

    def _pdigy_patches(label):
        return lambda: _Dec(generated_files[label]).get_extracted_patches()

    def _pdigy_thumb(label):
        return lambda: _Dec(generated_files[label]).thumbnail_image

    return [
        ("TIFF",                              _patches_tiff,                          _thumb_tiff),
        ("SVS",                               _patches_svs,                           _thumb_svs),
        ("DICOM lossy",                       _patches_dicom,                         _thumb_dicom),
        ("DICOM lossless",                    _patches_dicom_lossless,                _thumb_dicom_lossless),
        ("PDIGY",                             _pdigy_patches("with_whitespace"),      _pdigy_thumb("with_whitespace")),
        ("PDIGY (zero compressed)",           _pdigy_patches("without_whitespace"),   _pdigy_thumb("without_whitespace")),
        ("PDIGY lossless",                    _pdigy_patches("lossless"),             _pdigy_thumb("lossless")),
        ("PDIGY lossless (zero compressed)",  _pdigy_patches("lossless_zero_compressed"), _pdigy_thumb("lossless_zero_compressed")),
    ]


# ── Analysis cell helpers ────────────────────────────────────────────────────

def _mb(b):
    """Bytes → megabytes. I keep it SI-free for the vibe."""
    return b / 1024 ** 2


def _row(df, fmt, variant):
    """Pull a single row from the format comparison dataframe."""
    return df[(df["format"] == fmt) & (df["variant"] == variant)].iloc[0]


def _bench(df, label):
    """Pull a single benchmark row by format label."""
    return df[df["format"] == label].iloc[0]

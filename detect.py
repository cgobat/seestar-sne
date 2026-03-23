import logging
import numpy as np
import astroalign as aa

from astropy import wcs
from astropy.io import fits
from scipy.ndimage import label
from typing import Optional, Tuple
from reproject import reproject_interp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


# CONFIG <START>
# --------------------------

# Images used as reference sample
REFERENCE_IMAGE = "reference.fit"
# Images used for analysis (this one should contain SNe)
ANALYSIS_IMAGE = "science.fit"
# Use detect("reference_image_string", "analysis_image_string") directly if building a pipeline

# Output image with found difference
DIF_IMG = "found_difference.fit"
# Align images based on WCS data from image headers (Seestar usually has this).
# If False, try to align images based on geometry with AstroAlign.
USE_WCS = True
# Use Seestar-specific luminance weights when True; otherwise use standard perceptual weights.
USE_SEESTAR_LUMINANCE = True
# Detection threshold
NSIGMA = 1

# --------------------------
# CONFIG <END>


def open_fit(fl: str) -> Optional[Tuple[fits.Header, np.ndarray]]:
    """
    Opens a FITS (.fit or .fits) file and returns its primary header and data.

    Parameters
    ----------
    fl : str
        Path to the FITS file.

    Returns
    -------
    Optional[Tuple[Header, np.ndarray]]
        A tuple containing:
        - The FITS header (astropy.io.fits.Header)
        - The FITS image data as a NumPy array

        Returns None if the file could not be opened or an error occurs.
    """
    try:
        with fits.open(fl) as hdul:
            return (hdul[0].header, hdul[0].data)
    except Exception as e:
        logging.error(e)
        return None



def check_fit(pth: str) -> bool:
    """
    Checks whether a FITS file contains valid 3D image data with celestial WCS coordinates.

    Parameters
    ----------
    pth : str
        Path to the FITS file.

    Returns
    -------
    bool
        True if the FITS file contains:
        - A 3D data array
        - A valid celestial WCS in the header
        False otherwise.
    """
    try:
        hdul = open_fit(pth)
        if not hdul:
            return False

        header: Header = hdul[0]
        data: np.ndarray = hdul[1]

        # Ensure data is 3D (e.g., RGB or stacked exposures)
        if data.ndim != 3:
            logging.error(f"❌ Expected a 3D array, but got {data.ndim} dimensions.")
            return False

        # Validate WCS
        w = wcs.WCS(header, naxis=2)
        return w.is_celestial

    except Exception as e:
        logging.error(e)
        return False



def align_with_astro_align(reference_array: np.ndarray, source_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Aligns a source image (NumPy array) to a reference image (NumPy array) using AstroAlign.

    This function does not require or preserve any FITS header metadata — it only
    works with the raw image arrays.

    Parameters
    ----------
    reference_array : np.ndarray
        Reference image array (target alignment).
    source_array : np.ndarray
        Image array to be aligned (will be transformed to match the reference).

    Returns
    -------
    np.ndarray or None
        The aligned source image array if alignment succeeds, otherwise None.
    """
    try:
        if not isinstance(reference_array, np.ndarray):
            logging.error("reference_array must be a NumPy array.")
            return None
        if not isinstance(source_array, np.ndarray):
            logging.error("source_array must be a NumPy array.")
            return None

        logging.info("🚀 Aligning images... (This may take a moment)")

        aligned_data, footprint = aa.register(
            source=source_array.astype('float32'),
            target=reference_array.astype('float32')
        )

        return aligned_data

    except Exception as e:
        logging.error(f"❌ Error during alignment: {e}")
        if isinstance(e, aa.MaxIterError):
            logging.error("This can happen if there are not enough stars or if the images are too different.")
        return None



def align_with_wcs(reference_path: str, source_path: str) -> Optional[np.ndarray]:
    """
    Aligns a source FITS image to a reference FITS image using their WCS (World Coordinate System).

    This method reprojects the source image so that its celestial coordinate grid
    matches the reference image's grid.

    Parameters
    ----------
    reference_path : str
        Path to the reference FITS file (target alignment).  
        This file's header defines the output WCS and pixel grid.
    source_path : str
        Path to the source FITS file (to be aligned).

    Returns
    -------
    np.ndarray or None
        The aligned source image array if reprojection succeeds, otherwise None.
    """
    try:
        # Load reference FITS
        ref = open_fit(reference_path)
        if not ref:
            return None
        reference_header = ref[0]

        # Load source FITS
        srcp = open_fit(source_path)
        if not srcp:
            return None
        source_header = srcp[0]
        source_data = srcp[1]

        if not isinstance(source_data, np.ndarray):
            logging.error("Source FITS data must be a NumPy array.")
            return None

        logging.info(f"🚀 Reprojecting '{source_path}' onto the WCS of '{reference_path}'...")

        aligned_array, footprint = reproject_interp(
            (source_data, source_header),
            reference_header
        )

        return aligned_array

    except Exception as e:
        logging.error(f"❌ An error occurred during WCS reprojection: {e}")
        return None



def find_difference_arrays(reference_array: np.ndarray, aligned_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Perform difference imaging between a reference image and an aligned image
    to detect transient sources such as supernovae.

    The function subtracts the reference array from the aligned array, 
    applies an n-sigma threshold to find bright transient candidates, 
    removes very small detections, and writes the result to a FITS file.

    Parameters
    ----------
    reference_array : np.ndarray
        The reference image array.
    aligned_array : np.ndarray
        The aligned image array to compare against the reference.

    Returns
    -------
    np.ndarray or None
        A NumPy array containing only significant positive residuals 
        (likely new sources), or None if no detection is made.

    Notes
    -----
    - Saves the difference image to the path stored in the global `DIF_IMG` variable.
    - Uses the global `NSIGMA` variable to set the detection threshold.
    - Connected-component labeling is used to remove noise detections
      smaller than `min_size` pixels.
    """
    if reference_array.shape != aligned_array.shape:
        raise ValueError("Images must have the same dimensions for subtraction.")

    logging.info("🔎 Starting difference imaging...")

    # Image subtraction in float32 for precision
    diff = aligned_array.astype(np.float32) - reference_array.astype(np.float32)

    # Estimate background noise and set threshold
    sigma = np.nanstd(diff)
    threshold = NSIGMA * sigma
    logging.info(f"📈 Using {NSIGMA}σ threshold: {threshold:.2f}")

    # Mask significant bright pixels
    bright_mask = diff > threshold

    # Label connected components
    labeled, num_features = label(bright_mask)
    min_size = 3  # Ignore detections smaller than this size in pixels
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_size:
            bright_mask[labeled == i] = False

    # Create output transient data
    transient_data = np.zeros_like(diff)
    transient_data[bright_mask] = diff[bright_mask]

    num_pixels = np.sum(bright_mask)
    if num_pixels > 0:
        logging.info(f"✅ Found {num_pixels} significant pixels ({np.max(labeled)} objects).")
        fits.writeto(DIF_IMG, transient_data, overwrite=True)
        return transient_data
    else:
        logging.info("✅ No significant new objects found above threshold.")
        return None



def to_luminance(rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert an RGB image cube into a single-channel luminance image.

    Accepts either a channel-first (3, H, W) or channel-last (H, W, 3) NumPy array
    and returns a 2D float32 array of luminance values.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image data with exactly 3 channels, shape (3, H, W) or (H, W, 3).

    Returns
    -------
    np.ndarray or None
        2D array (H, W) of float32 luminance values,
        or None if dymenesions doesn't match

    Notes
    -----
    - Uses standard perceptual weights from the sRGB / Rec.709 standard:
      L = 0.2126*R + 0.7152*G + 0.0722*B
    - These weights are derived from the CIE XYZ color space to approximate
      perceived brightness for an average human observer.
    - If used camera sensor has a different spectral response, consider using
      a sensor-specific version (e.g., `to_luminance_s50_lp` for Seestar S50).
    """
    if rgb.ndim != 3 or 3 not in rgb.shape:
        logging.error("Expected an RGB array with 3 channels.")
        return None

    # Reorder to (H, W, 3) if needed
    if rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)  # (H, W, 3)

    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def to_luminance_s50_lp(rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert an RGB image cube to a luminance image using heuristic weights
    optimized for the Seestar S50 with a light-pollution (duo-band) filter.

    Accepts either (3, H, W) or (H, W, 3) array and returns a 2D float32 array.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image data with exactly 3 channels, shape (3, H, W) or (H, W, 3).

    Returns
    -------
    np.ndarray or None
        2D array (H, W) of float32 luminance values,
        or None if dymenesions doesn't match

    Notes
    -----
    - The weights are tuned for the spectral transmission of the Seestar S50 sensor 
      and its duo-band LP filter:
        L = 0.60*R + 0.35*G + 0.05*B
    - The red channel is emphasized because duo-band filters typically isolate Hα
      and OIII emission lines, making R more valuable for deep-sky targets.
    """
    if rgb.ndim != 3 or 3 not in rgb.shape:
        logging.error("Expected RGB array with 3 channels.")
        return None

    # Handle channel-first vs channel-last formats
    if rgb.shape[0] == 3:
        R, G, B = rgb[0].astype(np.float32), rgb[1].astype(np.float32), rgb[2].astype(np.float32)
    else:
        R = rgb[..., 0].astype(np.float32)
        G = rgb[..., 1].astype(np.float32)
        B = rgb[..., 2].astype(np.float32)

    return 0.60 * R + 0.35 * G + 0.05 * B




def detect(ref_img, ana_img) -> bool:
    """
    Created as function for purpose of pipeline building

    Parameters
    ----------
    ref_img: str
        Reference Image

    ana_img: str
        Image for Analyzing

    Returns
    -------
    bool
        True: processing was succesful
        False: Some kind of error occured during the process
    """

    # Check if .fit files are usable in this pipeline
    if check_fit(ref_img) and check_fit(ana_img):
        aligned_data = None
        luminance_channel_ref = None
        luminance_channel_sci = None

        # Get data and header files from the .fit files
        # *data already checked in "check_fit" for None
        ref = open_fit(ref_img)
        ref_header = ref[0]
        ref_data = ref[1]

        sci = open_fit(ana_img)
        sci_header = sci[0]
        sci_data = sci[1]

        # Convert RGB cube to luminance
        if USE_SEESTAR_LUMINANCE:
            luminance_channel_ref = to_luminance_s50_lp(ref_data)
            luminance_channel_sci = to_luminance_s50_lp(sci_data)
        else:
            luminance_channel_ref = to_luminance(ref_data)
            luminance_channel_sci = to_luminance(sci_data)

        if luminance_channel_ref is None or luminance_channel_sci is None:
            return False

        # Align the data
        if USE_WCS:
            lum_c_ref = "luminance_channel_ref.fit"
            lum_c_sci = "luminance_channel_sci.fit"
            fits.writeto(lum_c_ref, luminance_channel_ref, ref_header, overwrite=True)
            fits.writeto(lum_c_sci, luminance_channel_sci, sci_header, overwrite=True)

            aligned_data = align_with_wcs(lum_c_ref, lum_c_sci)
            os.remove(lum_c_ref)
            os.remove(lum_c_sci)

        else:
            aligned_data = align_with_astro_align(luminance_channel_ref, luminance_channel_sci)

        if not aligned_data is None:
            # Find differences in bright pixels
            datections = find_difference_arrays(luminance_channel_ref, aligned_data)
            if detect is None:
                return False
            else:
                return True
        else: 
            return False

    return False


if __name__ == "__main__":

    detect(REFERENCE_IMAGE, ANALYSIS_IMAGE)

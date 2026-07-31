"""
Qgis Geomorphometric Analysis Toolkit v1.0.0

An Open-Source QGIS Tool for Multi-Scale Terrain Analysis
using Digital Elevation Models (DEMs)
"""

from qgis.core import *
from qgis.gui import *
from qgis.PyQt.QtWidgets import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter, gaussian_filter, zoom
import scipy
import math
import os
import platform
import tempfile
import traceback
import gc
import logging
import time
from threading import Lock
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from contextlib import contextmanager

__version__ = "1.0.1"

logger = logging.getLogger('GeomorphologyTool')
logger.setLevel(logging.DEBUG)

class QgisLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        level_map = {
            logging.DEBUG: Qgis.Info,
            logging.INFO: Qgis.Info,
            logging.WARNING: Qgis.Warning,
            logging.ERROR: Qgis.Critical,
            logging.CRITICAL: Qgis.Critical
        }
        QgsMessageLog.logMessage(msg, "Geomorph", level_map.get(record.levelno, Qgis.Info))

handler = QgisLogHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

@contextmanager
def timer(algorithm_name: str):
    start = time.perf_counter()
    logger.info(f"Starting {algorithm_name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{algorithm_name} completed in {elapsed:.1f}s")

class GeomorphologyError(Exception):
    pass

class ValidationError(GeomorphologyError):
    pass

class ComputationError(GeomorphologyError):
    pass

class MemoryError(GeomorphologyError):
    pass

MAX_CELLS_HARD = 200_000_000
MAX_CELLS_WARN = 50_000_000
MAX_CELLS_LARGE = 30_000_000

WINDOW_FACTORS: Dict[str, int] = {
    "default": 9,
    "hypsometric": 15,
    "relief": 100,
    "tpi": 9,
    "tri": 5,
    "lrm": 25,
    "lbl": 25,
    "openness": 5
}

DEFAULT_SCALES_M = "25,100,250"
DEFAULT_NAN_FILL_WINDOW = 5
MIN_SIGMA_PX = 0.5
Z_LIMIT = 3.0
OUTPUT_SCALE = 100.0
DEFAULT_CURVATURE_SMOOTH = 0.0
MIN_VALID_TRIANGLES_ROUGHNESS = 4
CURVATURE_DENOMINATOR_EPSILON = 1e-8

DEFAULT_OPENNESS_DIRECTIONS = 8
DEFAULT_OPENNESS_RADIUS = 10
MAX_OPENNESS_RADIUS_WARN = 20
OPENNESS_ZENITH_LIMIT = 90

LRM_GAUSSIAN_FACTOR = 3.0
LRM_NAN_FILL_WINDOW = 5

DEFAULT_LACUNARITY_WINDOW = 65
DEFAULT_LACUNARITY_STEP = 10
DEFAULT_LACUNARITY_BOXES = "5,15,30"
DEFAULT_LACUNARITY_MIN_VALID = 0.75

NO_DATA_VALUE = -9999
EPSILON = 1e-10
DEFAULT_NAME_MAX_LENGTH = 50
GEOTIFF_CREATION_OPTIONS = ['COMPRESS=LZW', 'PREDICTOR=3', 'TILED=YES']

INDEX_SHORT_NAMES: Dict[str, str] = {
    "tpi": "TPI",
    "tri_classic": "TRI",
    "roughness": "Roughness",
    "curvature": "Curvature",
    "relief": "Relief",
    "hypsometric_integral": "MWHI",
    "multiscale_tpi": "GMTPI",
    "lrm": "LRM",
    "positive_openness": "PosOpen",
    "negative_openness": "NegOpen",
    "lbl": "LBL",
    "lacunarity": "LAC",
}

SCIENTIFIC_DOCS: Dict[str, Dict[str, Any]] = {
    "tpi": {
        "title": "1. TOPOGRAPHIC POSITION INDEX (TPI)",
        "algorithm_type": "Original implementation",
        "author": "Weiss, A. (2001)",
        "citation": "Weiss, A. (2001). Topographic Position and Landforms Analysis. ESRI User Conference.",
        "description": "Elevation difference between the central cell and the mean of the surrounding neighborhood.",
        "formula": "TPI = z₀ - z̄",
        "interpretation": "Positive: Ridges | Near zero: Flat/mid-slope | Negative: Valleys",
        "output_range": "±50 m typical",
        "units": "Meters (m)",
        "typical_scale": "100-500 m local, 1-5 km regional",
        "applications": "Landform classification, soil mapping, habitat modeling",
        "advantages": "Simple, scale-flexible",
        "limitations": "Sensitive to window size",
        "practical_use": "Small windows for local features, large for regional mapping",
        "recommended_color_ramp": "Blue (valleys) → White → Red (ridges)",
        "scientific_notes": "NaN-aware moving-window with masked mean.",
        "implementation_note": "Moving window with masked mean.",
        "how_to_interpret": ["High positive: Ridges/peaks", "Low negative: Valleys", "Near zero: Flat or uniform slopes"]
    },
    "tri_classic": {
        "title": "2. TERRAIN RUGGEDNESS INDEX (TRI)",
        "algorithm_type": "Original implementation",
        "author": "Riley et al. (1999)",
        "citation": "Riley, S.J. et al. (1999). A Terrain Ruggedness Index. Intermountain Journal of Sciences.",
        "description": "Root of summed squared elevation differences to neighbors in a moving window.",
        "formula": "TRI = √(Σ(zᵢ - z₀)² / n_valid)",
        "interpretation": "0-50 Smooth | 50-200 Moderate | 200-500 High | >500 Extreme",
        "output_range": "0 to >1000 m",
        "units": "Meters (m)",
        "typical_scale": "3×3 to 5×5 cells",
        "applications": "Terrain heterogeneity, landslide susceptibility, habitat complexity",
        "advantages": "Widely recognized, NaN-aware, simple interpretation",
        "limitations": "Window-size dependent, sensitive to DEM resolution",
        "practical_use": "Standard for ecological studies. Use 3×3 for micro-heterogeneity, larger for landscape scale.",
        "recommended_color_ramp": "Green → Yellow → Red",
        "scientific_notes": "Classic Riley TRI with valid cell normalization.",
        "implementation_note": "Moving window with uniform filter and valid cell count.",
        "how_to_interpret": ["High: Rugged terrain, cliffs, canyons", "Low: Smooth plains, plateaus", "Normalize by window size for cross-scale comparison"]
    },
    "roughness": {
        "title": "3. SURFACE ROUGHNESS",
        "algorithm_type": "Original implementation",
        "author": "Jenness (2004)",
        "citation": "Jenness, J.S. (2004). Calculating landscape surface area from DEMs.",
        "description": "Ratio of 3D surface area to planar area using triangular facets.",
        "formula": "Roughness = Σ|v₁ × v₂| / (n · dx · dy)",
        "interpretation": "1.00-1.02 Smooth | 1.02-1.10 Undulating | >1.10 Rough",
        "output_range": "1.0 to >3.0",
        "units": "Dimensionless",
        "typical_scale": "3×3 cells",
        "applications": "Microtopography, erosion potential, surface texture",
        "advantages": "True 3D measure, physically meaningful",
        "limitations": "Very local, sensitive to noise and DEM artifacts",
        "practical_use": "Best with high-resolution LiDAR. Requires ≥4 valid triangles.",
        "recommended_color_ramp": "White (1.0) → Dark red",
        "scientific_notes": "8-triangle facet method from Jenness (2004).",
        "implementation_note": "Requires ≥4 valid triangles for reliable results.",
        "how_to_interpret": ["Near 1.0: Smooth, planar surface", "1.02-1.10: Undulating terrain", ">1.3: Highly complex, dissected terrain"]
    },
    "curvature": {
        "title": "4. TERRAIN CURVATURE",
        "algorithm_type": "Original implementation",
        "author": "Zevenbergen & Thorne (1987)",
        "citation": "Zevenbergen & Thorne (1987). Quantitative Analysis of Land Surface Topography.",
        "description": "Profile and plan curvature from second-order finite differences.",
        "formula": "k_prof, k_plan from 2nd derivatives",
        "interpretation": "Profile+: Convex | Profile-: Concave | Plan+: Convergent | 0: Flat",
        "output_range": "±1 m⁻¹ typical",
        "units": "1/meters",
        "typical_scale": "3×3 cells",
        "applications": "Hydrology, erosion-deposition, soil moisture mapping",
        "advantages": "Physically meaningful, flat = 0, no NaN on flat terrain",
        "limitations": "Sensitive to DEM noise, may require pre-smoothing",
        "practical_use": "Start with σ=0, increase smoothing if results are noisy.",
        "recommended_color_ramp": "Blue → White → Red",
        "scientific_notes": "Classic Zevenbergen-Thorne finite difference method.",
        "implementation_note": "Flat terrain returns exactly 0, not NaN.",
        "how_to_interpret": ["Profile+ Plan+: Ridges, convex divergent", "Profile- Plan-: Valleys, concave convergent", "Zero: Flat or planar slopes"]
    },
    "relief": {
        "title": "5. RELATIVE RELIEF",
        "algorithm_type": "Original implementation",
        "author": "Smith, G.H. (1935)",
        "citation": "Smith, G.H. (1935). The Relative Relief of Ohio.",
        "description": "Local elevation range (max - min) in moving window.",
        "formula": "RR = z_max - z_min",
        "interpretation": "<10 m Flat | 10-50 Undulating | 50-200 Dissected | >200 Mountainous",
        "output_range": "0 to >1000 m",
        "units": "Meters (m)",
        "typical_scale": "100-500 m",
        "applications": "Basin characterization, tectonic geomorphology, landscape energy",
        "advantages": "Simple, robust, easy to interpret",
        "limitations": "Window dependent, can miss nested features",
        "practical_use": "First-order terrain characterization before detailed analysis.",
        "recommended_color_ramp": "Green → Yellow → Red → Purple",
        "scientific_notes": "One of the oldest geomorphometric measures (Smith, 1935).",
        "implementation_note": "Max/min filters with NaN-safe implementation.",
        "how_to_interpret": ["Low (<10m): Floodplains, plateaus", "Medium (10-50m): Rolling hills", "High (>200m): Mountainous terrain"]
    },
    "hypsometric_integral": {
        "title": "6. MOVING WINDOW HYPSOMETRIC INTEGRAL",
        "algorithm_type": "Moving-window adaptation",
        "author": "Strahler (1952) – adaptation",
        "citation": "Strahler, A.N. (1952). Hypsometric analysis of erosional topography.",
        "description": "Normalized elevation distribution in local neighborhood.",
        "formula": "MWHI = (z̄ - z_min) / (z_max - z_min)",
        "interpretation": ">0.6 Youthful | 0.35-0.6 Mature | <0.35 Old",
        "output_range": "0.0 – 1.0",
        "units": "Dimensionless",
        "typical_scale": "Watershed to regional",
        "applications": "Tectonic geomorphology, basin evolution, erosion stage",
        "advantages": "Simple physical meaning, dimensionless",
        "limitations": "Moving-window differs from basin HI, window-dependent",
        "practical_use": "Moving-window adaptation of Strahler's basin HI for continuous mapping.",
        "recommended_color_ramp": "Red (>0.6) → Yellow → Blue (<0.35)",
        "scientific_notes": "Moving-window adaptation of Strahler's classic hypsometric integral.",
        "implementation_note": "Protected against zero relief division.",
        "how_to_interpret": [">0.6: Youthful landscape (recent uplift)", "0.35-0.6: Mature (equilibrium)", "<0.35: Old (peneplain, low relief)"]
    },
    "multiscale_tpi": {
        "title": "7. GAUSSIAN MULTISCALE TPI",
        "algorithm_type": "Modified implementation",
        "author": "Extended from Weiss (2001)",
        "citation": "Weiss, A. (2001). Topographic Position and Landforms Analysis.",
        "description": "TPI across multiple Gaussian scales, normalized and fused.",
        "formula": "TPI_s = z₀ - G(z, σ_s), then fused",
        "interpretation": "Positive: Persistent ridges | Negative: Persistent valleys",
        "output_range": "-100 to +100 (normalized)",
        "units": "Dimensionless",
        "typical_scale": "25 / 100 / 250 m",
        "applications": "Multi-resolution landform classification, nested feature detection",
        "advantages": "Captures nested features across scales",
        "limitations": "Scale selection subjective, computationally intensive",
        "practical_use": "Default scales 25,100,250 m capture micro to landscape features.",
        "recommended_color_ramp": "Blue (valleys) → White → Red (ridges)",
        "scientific_notes": "Gaussian multiscale extension with z-score normalization.",
        "implementation_note": "Per-scale z-score normalization + multi-method fusion.",
        "how_to_interpret": ["Strong positive: Persistent ridges across scales", "Strong negative: Persistent valleys", "Near zero: Scale-dependent or flat"]
    },
    "lrm": {
        "title": "8. LOCAL RELIEF MODEL (LRM)",
        "algorithm_type": "Original implementation",
        "author": "Hesse, R. (2010)",
        "citation": "Hesse, R. (2010). LiDAR-derived Local Relief Models.",
        "description": "Original DEM minus Gaussian-smoothed surface.",
        "formula": "LRM = z₀ - G(z, σ)  σ = window/3",
        "interpretation": "Positive: Local highs | Negative: Local lows",
        "output_range": "±10 m typical (LiDAR)",
        "units": "Meters (m)",
        "typical_scale": "50-500 m",
        "applications": "Archaeological prospection, micro-topography, feature extraction",
        "advantages": "Reveals subtle features, removes regional trend",
        "limitations": "Window size critical, may remove real features if window is small",
        "practical_use": "Window ≈ 10× feature size for optimal detection.",
        "recommended_color_ramp": "Blue (depressions) → White → Red (elevations)",
        "scientific_notes": "Developed for LiDAR archaeological prospection (Hesse, 2010).",
        "implementation_note": "Gaussian σ = window/3 + NaN fill for edge handling.",
        "how_to_interpret": ["Strong positive: Mounds, buildings, ridges", "Negative: Ditches, channels, pits", "Near zero: Trend surface"]
    },
    "positive_openness": {
        "title": "9. POSITIVE OPENNESS",
        "algorithm_type": "Approximation",
        "author": "Based on Yokoyama et al. (2002)",
        "citation": "Yokoyama et al. (2002). Visualizing topography by openness.",
        "description": "Maximum upward viewing angle in multiple directions.",
        "formula": "Φ₊ ≈ 180° - max(arctan((z_r - z₀)/r))",
        "interpretation": "High (>130°): Exposed | Low (<90°): Sheltered",
        "output_range": "0–180°",
        "units": "Degrees (°)",
        "typical_scale": "Radius 5–30 cells",
        "applications": "Visibility analysis, microclimate, habitat exposure",
        "advantages": "Efficient for large DEMs, intuitive interpretation",
        "limitations": "Directional approximation, not continuous horizon",
        "practical_use": "Directional horizon approximation. More directions = more accurate but slower.",
        "recommended_color_ramp": "Dark blue (sheltered) → Light (exposed)",
        "scientific_notes": "Efficient approximation of Yokoyama openness.",
        "implementation_note": "Multi-directional sampling with safe_shift.",
        "how_to_interpret": [">140°: Exposed peaks, ridges", "90-130°: Moderate slopes", "<80°: Sheltered valleys, depressions"]
    },
    "negative_openness": {
        "title": "10. NEGATIVE OPENNESS",
        "algorithm_type": "Approximation",
        "author": "Based on Yokoyama et al. (2002)",
        "citation": "Yokoyama et al. (2002). Visualizing topography by openness.",
        "description": "Maximum downward viewing angle in multiple directions.",
        "formula": "Φ₋ ≈ 180° - max(arctan((z₀ - z_r)/r))",
        "interpretation": "High: Deep valleys | Low: Flat/ridges",
        "output_range": "0–180°",
        "units": "Degrees (°)",
        "typical_scale": "Radius 5–30 cells",
        "applications": "Valley mapping, depression detection, drainage analysis",
        "advantages": "Complementary to positive openness, reveals hidden valleys",
        "limitations": "Directional approximation, computationally intensive",
        "practical_use": "Combine with positive openness for complete terrain characterization.",
        "recommended_color_ramp": "Light (flat) → Dark blue (deep valleys)",
        "scientific_notes": "Efficient approximation of Yokoyama negative openness.",
        "implementation_note": "Multi-directional sampling with safe_shift.",
        "how_to_interpret": [">140°: Deep valleys, gorges", "90-130°: Moderate valleys", "<80°: Flat terrain or ridges"]
    },
    "lbl": {
        "title": "11. LOCAL BASE LEVEL APPROXIMATION",
        "algorithm_type": "Approximation",
        "author": "Adapted from base-level concepts",
        "citation": "Filosofov (1960); Golts & Rosenthal (1992); Jaboyedoff et al. (2004)",
        "description": "Minimum elevation in moving window (simplified base level).",
        "formula": "LBL ≈ min(z) within window",
        "interpretation": "Low: Valley bottoms | Higher: Elevated local minima",
        "output_range": "Same as DEM",
        "units": "Meters (m)",
        "typical_scale": "100-500 m",
        "applications": "Relative elevation mapping, valley floor identification",
        "advantages": "Simple, robust, fast computation",
        "limitations": "Not full SLBL, simplified local minimum",
        "practical_use": "Calculate DEM − LBL for residual topography above local base level.",
        "recommended_color_ramp": "Blue (low) → Yellow → Red (high)",
        "scientific_notes": "Simplified local minimum filter, not iterative SLBL algorithm.",
        "implementation_note": "Minimum filter with NaN-safe implementation.",
        "how_to_interpret": ["Lowest values: Active valley floors", "Higher values: Elevated terraces, plateaus", "Best used as: DEM − LBL for relative elevation"]
    },
    "lacunarity": {
        "title": "12. MOMENT-BASED LACUNARITY",
        "algorithm_type": "Approximation",
        "author": "Moment estimator from Plotnick et al. (1996)",
        "citation": "Plotnick et al. (1996). Lacunarity analysis. Physical Review E.",
        "description": "Spatial heterogeneity from local first and second moments.",
        "formula": "L(r) ≈ <Z²> / <Z>²",
        "interpretation": "≈1.0 Homogeneous | >1.5 Heterogeneous | >2.0 Highly heterogeneous",
        "output_range": "1.0 – 3.0 typical",
        "units": "Dimensionless",
        "typical_scale": "Window 65 px, boxes 5-30",
        "applications": "Terrain texture, habitat complexity, landscape heterogeneity",
        "advantages": "Fully vectorized, suitable for large DEMs",
        "limitations": "Moment approximation, not full gliding-box method",
        "practical_use": "Efficient for large rasters. Multiple box sizes capture multi-scale texture.",
        "recommended_color_ramp": "Blue (homogeneous) → Yellow → Red (heterogeneous)",
        "scientific_notes": "Moment estimator of Plotnick lacunarity - efficient vectorized implementation.",
        "implementation_note": "Vectorized uniform_filter with multi-scale averaging.",
        "how_to_interpret": ["≈1.0: Homogeneous terrain (plains, plateaus)", "1.5-2.0: Heterogeneous (mixed terrain)", ">2.0: Highly heterogeneous (complex topography)"]
    },
}

def validate_dem_array(arr: np.ndarray, min_size: int = 3) -> None:
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(arr)}")
    if arr.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {arr.ndim}D")
    if arr.shape[0] < min_size or arr.shape[1] < min_size:
        raise ValidationError(f"Array too small: {arr.shape}. Minimum required: {min_size}x{min_size}")
    if np.all(np.isnan(arr)):
        raise ValidationError("Array contains only NaN values")

def validate_cell_size(dx: float, dy: float) -> None:
    if dx <= 0 or dy <= 0:
        raise ValidationError(f"Invalid cell sizes: dx={dx}, dy={dy}. Must be positive.")
    if np.isinf(dx) or np.isinf(dy):
        raise ValidationError(f"Invalid cell sizes: dx={dx}, dy={dy}. Cannot be infinite.")

def validate_window_size(window_px: int, array_shape: Tuple[int, int]) -> None:
    if window_px < 3:
        raise ValidationError(f"Window size {window_px} too small. Minimum: 3")
    if window_px > min(array_shape):
        raise ValidationError(f"Window size {window_px} exceeds array dimension {min(array_shape)}")

def get_window_size(wx: int, wy: int) -> int:
    size = max(wx, wy)
    return size + 1 if size % 2 == 0 else size

def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b > EPSILON, a / b, fill_value)
    return np.where(np.isinf(result), fill_value, result)

def masked_mean(data: np.ndarray, size: int, vmask: np.ndarray) -> np.ndarray:
    size_odd = get_window_size(size, size)
    sv = uniform_filter(data, size=size_odd, mode='constant', cval=0)
    ct = uniform_filter(vmask, size=size_odd, mode='constant', cval=0)
    return safe_divide(sv, ct)

def nan_safe_max_filter(arr: np.ndarray, size: int) -> np.ndarray:
    return maximum_filter(
        np.where(np.isnan(arr), -np.inf, arr),
        size=size,
        mode='constant',
        cval=-np.inf
    )

def nan_safe_min_filter(arr: np.ndarray, size: int) -> np.ndarray:
    return minimum_filter(
        np.where(np.isnan(arr), np.inf, arr),
        size=size,
        mode='constant',
        cval=np.inf
    )

def compute_stats(arr: np.ndarray, name: str) -> Dict[str, Any]:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        logger.warning(f"No valid values for {name}")
        return {}
    
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    
    return {
        'name': name,
        'mean': mean,
        'median': float(np.median(valid)),
        'std': std,
        'cv': float(std / abs(mean)) if abs(mean) > EPSILON else 0,
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'count': len(valid)
    }

def compute_derivatives(Z: np.ndarray, dx: float, dy: float, smooth_sigma: Optional[float] = None) -> Dict[str, np.ndarray]:
    validate_dem_array(Z, min_size=3)
    validate_cell_size(dx, dy)
    
    if smooth_sigma and smooth_sigma > 0:
        Z_smooth = gaussian_filter(Z, sigma=smooth_sigma, mode='nearest')
    else:
        Z_smooth = Z
    
    fx = (Z_smooth[1:-1, 2:] - Z_smooth[1:-1, :-2]) / (2 * dx)
    fy = (Z_smooth[2:, 1:-1] - Z_smooth[:-2, 1:-1]) / (2 * dy)
    fxx = (Z_smooth[1:-1, 2:] + Z_smooth[1:-1, :-2] - 2 * Z_smooth[1:-1, 1:-1]) / (dx ** 2)
    fyy = (Z_smooth[2:, 1:-1] + Z_smooth[:-2, 1:-1] - 2 * Z_smooth[1:-1, 1:-1]) / (dy ** 2)
    fxy = (Z_smooth[2:, 2:] - Z_smooth[2:, :-2] - Z_smooth[:-2, 2:] + Z_smooth[:-2, :-2]) / (4 * dx * dy)
    
    return {'fx': fx, 'fy': fy, 'fxx': fxx, 'fyy': fyy, 'fxy': fxy}

def fill_nan_with_local_mean(dem: np.ndarray, mask: np.ndarray, window_size: int = DEFAULT_NAN_FILL_WINDOW) -> np.ndarray:
    nan_mask = np.isnan(dem)
    if not np.any(nan_mask):
        return dem
    
    filled = dem.copy()
    valid_dem = np.where(mask, dem, 0)
    valid_count = uniform_filter(mask.astype(np.float32), size=window_size, mode='constant', cval=0)
    valid_sum = uniform_filter(valid_dem, size=window_size, mode='constant', cval=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = np.where(valid_count > 0, valid_sum / valid_count, 0)
    
    filled[nan_mask] = local_mean[nan_mask]
    return filled

def safe_shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    rows, cols = arr.shape
    out = np.full_like(arr, np.nan)
    
    y_src = slice(max(0, -dy), min(rows, rows - dy))
    x_src = slice(max(0, -dx), min(cols, cols - dx))
    y_dst = slice(max(0, dy), min(rows, rows + dy))
    x_dst = slice(max(0, dx), min(cols, cols + dx))
    
    if y_src.stop > y_src.start and x_src.stop > x_src.start:
        out[y_dst, x_dst] = arr[y_src, x_src]
    
    return out

def parse_scales(scales_str: str) -> List[float]:
    try:
        scales = [float(s.strip()) for s in scales_str.split(',') if s.strip()]
    except ValueError as e:
        raise ValidationError(f"Invalid scales format: '{scales_str}'. Use comma-separated numbers, e.g., '25,100,250'") from e
    
    if len(scales) < 2:
        raise ValidationError("Need at least 2 scales for multiscale analysis")
    if any(s <= 0 for s in scales):
        raise ValidationError("All scales must be positive values")
    
    return sorted(scales)

def create_metadata(dk: str, doc: Dict[str, str], params_dict: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        'Algorithm': doc.get('title', dk),
        'Algorithm_Key': dk,
        'Algorithm_Type': doc.get('algorithm_type', 'Unknown'),
        'Citation': doc.get('citation', ''),
        'Formula': doc.get('formula', ''),
        'Units': doc.get('units', ''),
        'Version': __version__,
        'Author': doc.get('author', ''),
        'Timestamp': datetime.now().isoformat()
    }
    meta.update(params_dict)
    return meta

def make_output_name(dk: str, base_name: str, window_m: Optional[float] = None, extra: str = "") -> str:
    parts = [INDEX_SHORT_NAMES.get(dk, dk.upper())]
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)[:20].strip("_")
    
    if clean:
        parts.append(clean)
    if window_m and window_m > 0:
        parts.append(f"{int(window_m)}m")
    if extra:
        parts.append(str(extra))
    
    return "_".join(parts)[:DEFAULT_NAME_MAX_LENGTH]

def calc_tpi(w: 'Worker') -> Dict[str, Any]:
    with timer("TPI calculation"):
        w.status.emit("Calculating TPI...")
        w.progress.emit(10)
        
        dem, mask, df = w.sd.get()
        validate_dem_array(dem)
        
        tpi = dem - masked_mean(df, w.window_size(), w.sd.vmask())
        tpi[~mask] = np.nan
        
        w.progress.emit(90)
        return {"array": tpi, "name": w.name, "stats": compute_stats(tpi, "TPI")}

def calc_tri_classic(w: 'Worker') -> Dict[str, Any]:
    with timer("TRI calculation"):
        w.status.emit("Calculating TRI...")
        w.progress.emit(10)
        
        dem, mask, df = w.sd.get()
        validate_dem_array(dem)
        
        size = w.window_size()
        v = w.sd.vmask()
        
        m = uniform_filter(df, size=size, mode='constant', cval=0)
        m2 = uniform_filter(df ** 2, size=size, mode='constant', cval=0)
        ct = uniform_filter(v, size=size, mode='constant', cval=0)
        
        variance = np.maximum(ct * m2 - 2 * dem * ct * m + ct * dem ** 2, 0)
        tri = np.where((ct > 0) & mask, np.sqrt(variance / ct), np.nan)
        
        w.progress.emit(90)
        return {"array": tri, "name": w.name, "stats": compute_stats(tri, "TRI")}

def calc_roughness(w: 'Worker') -> Dict[str, Any]:
    with timer("Roughness calculation"):
        w.status.emit("Calculating Surface Roughness...")
        w.progress.emit(5)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        dx, dy = w.sd.csx, w.sd.csy
        validate_cell_size(dx, dy)
        
        Z = w.padded()
        rows, cols = dem.shape
        tri_planar = dx * dy / 8
        min_valid = w.metadata.get('min_valid_triangles', MIN_VALID_TRIANGLES_ROUGHNESS)
        
        shifts = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        total_3d = np.zeros_like(dem)
        valid_count = np.zeros_like(dem, dtype=np.int32)
        
        for k, (di, dj) in enumerate(shifts):
            z_k = Z[1+di:rows+1+di, 1+dj:cols+1+dj]
            dk1 = shifts[(k + 1) % 8]
            z_k1 = Z[1+dk1[0]:rows+1+dk1[0], 1+dk1[1]:cols+1+dk1[1]]
            
            valid = mask & ~np.isnan(z_k) & ~np.isnan(z_k1)
            
            v1x, v1y, v1z = dj * dx, -di * dy, z_k - dem
            v2x, v2y, v2z = dk1[1] * dx, -dk1[0] * dy, z_k1 - dem
            
            cx = v1y * v2z - v1z * v2y
            cy = v1z * v2x - v1x * v2z
            cz = v1x * v2y - v1y * v2x
            
            tri_area = 0.5 * np.sqrt(np.maximum(cx**2 + cy**2 + cz**2, 0))
            
            total_3d[valid] += tri_area[valid]
            valid_count[valid] += 1
            w.check()
        
        roughness = np.full_like(dem, np.nan)
        good = valid_count >= min_valid
        roughness[good] = total_3d[good] / (valid_count[good] * tri_planar)
        
        w.progress.emit(90)
        return {"array": roughness, "name": w.name, "stats": compute_stats(roughness, "Roughness")}

def calc_curvature(w: 'Worker') -> Dict[str, Any]:
    with timer("Curvature calculation"):
        w.status.emit("Calculating Curvature...")
        w.progress.emit(10)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        dx, dy = w.sd.csx, w.sd.csy
        validate_cell_size(dx, dy)
        
        Z = w.padded()
        smooth_sigma = w.metadata.get('smooth_sigma', DEFAULT_CURVATURE_SMOOTH)
        
        derivs = compute_derivatives(Z, dx, dy, smooth_sigma if smooth_sigma > 0 else None)
        w.progress.emit(40)
        
        denom = derivs['fx']**2 + derivs['fy']**2
        
        prof = np.zeros_like(dem)
        plan = np.zeros_like(dem)
        
        valid = (denom > CURVATURE_DENOMINATOR_EPSILON) & mask
        flat = (denom <= CURVATURE_DENOMINATOR_EPSILON) & mask
        
        if np.any(valid):
            fx, fy = derivs['fx'], derivs['fy']
            fxx, fyy, fxy = derivs['fxx'], derivs['fyy'], derivs['fxy']
            
            prof[valid] = -(
                fxx[valid] * fx[valid]**2 +
                2 * fxy[valid] * fx[valid] * fy[valid] +
                fyy[valid] * fy[valid]**2
            ) / (denom[valid] * (1 + denom[valid])**1.5)
            
            plan[valid] = -(
                fyy[valid] * fx[valid]**2 -
                2 * fxy[valid] * fx[valid] * fy[valid] +
                fxx[valid] * fy[valid]**2
            ) / (denom[valid]**1.5)
        
        prof[flat] = 0.0
        plan[flat] = 0.0
        prof[~mask] = np.nan
        plan[~mask] = np.nan
        
        w.progress.emit(70)
        
        return {
            "arrays": {"profile": prof, "plan": plan},
            "name": w.name,
            "stats": {
                "profile": compute_stats(prof, "Profile_Curvature"),
                "plan": compute_stats(plan, "Plan_Curvature")
            }
        }

def calc_relief(w: 'Worker') -> Dict[str, Any]:
    with timer("Relief calculation"):
        w.status.emit("Calculating Relative Relief...")
        w.progress.emit(20)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        size = w.window_size()
        validate_window_size(size, dem.shape)
        
        mx = nan_safe_max_filter(dem, size)
        mn = nan_safe_min_filter(dem, size)
        
        relief = mx - mn
        relief[~mask] = np.nan
        
        w.progress.emit(90)
        return {"array": relief, "name": w.name, "stats": compute_stats(relief, "Relief")}

def calc_hypsometric_integral(w: 'Worker') -> Dict[str, Any]:
    with timer("Hypsometric Integral calculation"):
        w.status.emit("Calculating Moving Window Hypsometric Integral...")
        w.progress.emit(10)
        
        dem, mask, df = w.sd.get()
        validate_dem_array(dem)
        
        size = w.window_size()
        validate_window_size(size, dem.shape)
        
        z_min = nan_safe_min_filter(dem, size)
        z_max = nan_safe_max_filter(dem, size)
        z_mean = masked_mean(df, size, w.sd.vmask())
        
        relief = z_max - z_min
        hi = np.full_like(dem, np.nan)
        valid = relief > EPSILON
        hi[valid] = (z_mean[valid] - z_min[valid]) / relief[valid]
        hi = np.clip(hi, 0.0, 1.0)
        hi[~mask] = np.nan
        
        w.progress.emit(90)
        return {"array": hi, "name": w.name, "stats": compute_stats(hi, "MWHI")}

def calc_multiscale_tpi(w: 'Worker') -> Dict[str, Any]:
    with timer("Multiscale TPI calculation"):
        w.status.emit("Calculating Gaussian Multiscale TPI...")
        w.progress.emit(5)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        pixel_size = min(w.sd.csx, w.sd.csy)
        scales_m = parse_scales(w.metadata.get('scales', DEFAULT_SCALES_M))
        scales_px = [max(s / pixel_size, MIN_SIGMA_PX) for s in scales_m]
        
        dem_filled = fill_nan_with_local_mean(dem, mask, w.metadata.get('nan_fill_window', DEFAULT_NAN_FILL_WINDOW))
        
        tpi_results = []
        for i, (scale_m, sigma_px) in enumerate(zip(scales_m, scales_px)):
            w.check()
            progress = 10 + (i * 80 // len(scales_m))
            w.progress.emit(progress)
            
            smoothed = gaussian_filter(dem_filled, sigma=sigma_px, mode='nearest')
            tpi = dem - smoothed
            tpi[~mask] = np.nan
            
            valid = tpi[mask]
            std_val = np.std(valid)
            
            if std_val > EPSILON:
                z_scores = np.clip((tpi - np.mean(valid)) / std_val, -Z_LIMIT, Z_LIMIT)
                normalized = (z_scores / Z_LIMIT) * OUTPUT_SCALE
            else:
                normalized = np.zeros_like(tpi)
            
            normalized[~mask] = np.nan
            tpi_results.append(normalized)
        
        tpi_stack = np.stack(tpi_results, axis=0)
        fusion = w.metadata.get('fusion', 'mean')
        
        if fusion == 'median':
            result = np.nanmedian(tpi_stack, axis=0)
        elif fusion == 'max_abs':
            idx = np.nanargmax(np.abs(tpi_stack), axis=0)
            rows, cols = np.indices(idx.shape)
            result = tpi_stack[idx, rows, cols]
        elif fusion == 'weighted':
            weights = np.array([1.0 / s for s in scales_m])
            weights /= weights.sum()
            result = np.nansum(tpi_stack * weights.reshape(-1, 1, 1), axis=0)
        else:
            result = np.nanmean(tpi_stack, axis=0)
        
        result[~mask] = np.nan
        
        w.progress.emit(95)
        return {"array": result, "name": w.name, "stats": compute_stats(result, "GMTPI")}

def calc_lrm(w: 'Worker') -> Dict[str, Any]:
    with timer("LRM calculation"):
        w.status.emit("Calculating Local Relief Model...")
        w.progress.emit(10)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        size = w.window_size()
        sigma = size / LRM_GAUSSIAN_FACTOR
        
        dem_filled = fill_nan_with_local_mean(dem, mask, LRM_NAN_FILL_WINDOW)
        smoothed = gaussian_filter(dem_filled, sigma=sigma, mode='nearest')
        
        lrm = dem - smoothed
        lrm[~mask] = np.nan
        
        w.progress.emit(90)
        return {"array": lrm, "name": w.name, "stats": compute_stats(lrm, "LRM")}

def calc_openness(w: 'Worker', positive: bool = True) -> Dict[str, Any]:
    label = "Positive" if positive else "Negative"
    
    with timer(f"{label} Openness calculation"):
        w.status.emit(f"Calculating {label} Openness...")
        w.progress.emit(5)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        radius = w.metadata.get('Radius_cells', DEFAULT_OPENNESS_RADIUS)
        directions = w.metadata.get('Num_Directions', DEFAULT_OPENNESS_DIRECTIONS)
        pixel_size = min(w.sd.csx, w.sd.csy)
        
        max_angle = np.full_like(dem, -np.inf, dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, directions, endpoint=False)
        
        for d_idx, angle in enumerate(angles):
            w.check()
            
            if d_idx % max(1, directions // 4) == 0:
                progress = 15 + (d_idx * 70 // directions)
                w.progress.emit(progress)
            
            dx_dir, dy_dir = np.cos(angle), np.sin(angle)
            max_dir = np.full_like(dem, -np.inf, dtype=np.float32)
            
            for r in range(1, radius + 1):
                sx, sy = int(round(dx_dir * r)), int(round(dy_dir * r))
                if sx == 0 and sy == 0:
                    continue
                
                shifted = safe_shift(dem, sy, sx)
                horiz = r * pixel_size
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    if positive:
                        ang = np.degrees(np.arctan2(dem - shifted, horiz))
                    else:
                        ang = np.degrees(np.arctan2(shifted - dem, horiz))
                
                ang = np.clip(ang, -OPENNESS_ZENITH_LIMIT, OPENNESS_ZENITH_LIMIT)
                valid = mask & ~np.isnan(ang) & ~np.isnan(shifted)
                max_dir[valid] = np.maximum(max_dir[valid], ang[valid])
            
            valid_dir = np.isfinite(max_dir)
            max_angle[valid_dir] = np.maximum(max_angle[valid_dir], max_dir[valid_dir])
        
        openness = np.full_like(dem, np.nan, dtype=np.float32)
        valid = np.isfinite(max_angle)
        openness[valid] = 180.0 - max_angle[valid]
        openness = np.clip(openness, 0, 180)
        openness[~mask] = np.nan
        
        w.progress.emit(95)
        name = "PosOpen" if positive else "NegOpen"
        return {"array": openness, "name": w.name, "stats": compute_stats(openness, name)}

def calc_positive_openness(w: 'Worker') -> Dict[str, Any]:
    return calc_openness(w, positive=True)

def calc_negative_openness(w: 'Worker') -> Dict[str, Any]:
    return calc_openness(w, positive=False)

def calc_local_base_level(w: 'Worker') -> Dict[str, Any]:
    with timer("Local Base Level calculation"):
        w.status.emit("Calculating Local Base Level Approximation...")
        w.progress.emit(20)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        size = w.window_size()
        validate_window_size(size, dem.shape)
        
        lbl = nan_safe_min_filter(dem, size)
        lbl[~mask] = np.nan
        lbl[np.isinf(lbl)] = np.nan
        
        w.progress.emit(90)
        return {"array": lbl, "name": w.name, "stats": compute_stats(lbl, "LBL_Approx")}

def calc_lacunarity(w: 'Worker') -> Dict[str, Any]:
    with timer("Lacunarity calculation"):
        w.status.emit("Calculating Moment-based Lacunarity...")
        w.progress.emit(5)
        
        dem, mask, _ = w.sd.get()
        validate_dem_array(dem)
        
        h, w_dem = dem.shape
        
        window = w.metadata.get('lac_window', DEFAULT_LACUNARITY_WINDOW)
        step = w.metadata.get('lac_step', DEFAULT_LACUNARITY_STEP)
        box_sizes_str = w.metadata.get('lac_boxes', DEFAULT_LACUNARITY_BOXES)
        min_valid_frac = w.metadata.get('lac_min_valid', DEFAULT_LACUNARITY_MIN_VALID)
        
        try:
            box_sizes = [int(s.strip()) for s in box_sizes_str.split(',') if s.strip() and int(s.strip()) > 0]
        except (ValueError, TypeError):
            box_sizes = [5, 15, 30]
        
        if not box_sizes:
            box_sizes = [5, 15, 30]
        
        if window >= min(h, w_dem):
            window = min(h, w_dem) // 4
            step = max(3, window // 8)
        
        box_sizes = [b for b in box_sizes if b < window]
        if not box_sizes:
            box_sizes = [min(5, window - 1)]
        
        data = np.where(mask, dem, 0.0).astype(np.float32)
        valid_f = mask.astype(np.float32)
        
        out_h = (h - window) // step + 1
        out_w = (w_dem - window) // step + 1
        
        if out_h <= 0 or out_w <= 0:
            raise ValidationError(f"Window {window} too large for DEM {h}x{w_dem}")
        
        lac_sum = np.zeros((out_h, out_w), dtype=np.float64)
        lac_count = np.zeros((out_h, out_w), dtype=np.int32)
        half = window // 2
        min_valid_px = box_sizes[0] ** 2 * min_valid_frac
        
        for idx, box in enumerate(box_sizes):
            w.check()
            progress = 20 + (idx * 60 // len(box_sizes))
            w.progress.emit(progress)
            
            box_area = box * box
            
            local_sum = uniform_filter(data, size=box, mode='constant', cval=0) * box_area
            local_sq = uniform_filter(data**2, size=box, mode='constant', cval=0) * box_area
            local_cnt = uniform_filter(valid_f, size=box, mode='constant', cval=0) * box_area
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = np.where(local_cnt >= min_valid_px, local_sum / local_cnt, np.nan)
                mean_sq = np.where(local_cnt >= min_valid_px, local_sq / local_cnt, np.nan)
                pixel_lac = np.where(
                    (local_cnt >= min_valid_px) & (mean > EPSILON),
                    mean_sq / mean**2,
                    np.nan
                )
            
            local_win = window - box + 1
            valid_lac = ~np.isnan(pixel_lac)
            
            lac_avg = uniform_filter(np.where(valid_lac, pixel_lac, 0.0), size=local_win, mode='constant', cval=0) * local_win**2
            lac_cnt = uniform_filter(valid_lac.astype(np.float32), size=local_win, mode='constant', cval=0) * local_win**2
            
            with np.errstate(divide='ignore', invalid='ignore'):
                averaged = np.where(lac_cnt >= 10, lac_avg / lac_cnt, np.nan)
            
            sampled = averaged[half:h-half:step, half:w_dem-half:step]
            
            if sampled.shape[0] < out_h or sampled.shape[1] < out_w:
                pad_h = max(0, out_h - sampled.shape[0])
                pad_w = max(0, out_w - sampled.shape[1])
                sampled = np.pad(sampled, ((0, pad_h), (0, pad_w)), constant_values=np.nan)
            
            sampled = sampled[:out_h, :out_w]
            
            valid_s = ~np.isnan(sampled)
            lac_sum[valid_s] += sampled[valid_s]
            lac_count[valid_s] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            down = np.where(lac_count > 0, lac_sum / lac_count, np.nan)
        
        scale_y, scale_x = h / down.shape[0], w_dem / down.shape[1]
        
        if scale_y != 1.0 or scale_x != 1.0:
            result = zoom(down, (scale_y, scale_x), order=1)
        else:
            result = down
        
        result = result[:h, :w_dem]
        result[~mask] = np.nan
        
        w.progress.emit(95)
        return {"array": result, "name": w.name, "stats": compute_stats(result, "LAC_Moment")}

ALGORITHMS: Dict[str, callable] = {
    "tpi": calc_tpi,
    "tri_classic": calc_tri_classic,
    "roughness": calc_roughness,
    "curvature": calc_curvature,
    "relief": calc_relief,
    "hypsometric_integral": calc_hypsometric_integral,
    "multiscale_tpi": calc_multiscale_tpi,
    "lrm": calc_lrm,
    "positive_openness": calc_positive_openness,
    "negative_openness": calc_negative_openness,
    "lbl": calc_local_base_level,
    "lacunarity": calc_lacunarity,
}

class SharedData:
    def __init__(self):
        self._lock = Lock()
        self.dem: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.dem_filled: Optional[np.ndarray] = None
        self.csx: Optional[float] = None
        self.csy: Optional[float] = None
        self.ext = None
        self.crs = None
        self._init = False

    def init(self, dem: np.ndarray, mask: np.ndarray, csx: float, csy: float, ext, crs) -> None:
        with self._lock:
            self.dem = dem.copy()
            self.mask = mask.copy()
            self.dem_filled = np.where(np.isnan(dem), 0, dem).astype(np.float32)
            self.csx = csx
            self.csy = csy
            self.ext = ext
            self.crs = crs
            self._init = True

    def ready(self) -> bool:
        with self._lock:
            return self._init and self.dem is not None

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if not self._init:
                raise RuntimeError("SharedData not initialized")
            return self.dem.copy(), self.mask.copy(), self.dem_filled.copy()

    def vmask(self) -> np.ndarray:
        with self._lock:
            return (~np.isnan(self.dem)).astype(np.float32)

    def stats(self) -> Tuple[int, int, int, float, float]:
        with self._lock:
            if not self._init:
                return 0, 0, 0, 0.0, 0.0
            return (
                self.dem.shape[1],
                self.dem.shape[0],
                int(np.sum(self.mask)),
                float(np.nanmin(self.dem)),
                float(np.nanmax(self.dem))
            )

class Worker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn: callable, dk: str, sd: SharedData, cols: int, rows: int, wm: float, name: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.fn = fn
        self.dk = dk
        self.sd = sd
        self.cols = cols
        self.rows = rows
        self.wm = wm
        self.name = name
        self.metadata = metadata or {}
        self._cancelled = False
        self._padded = None
        self._pad_lock = Lock()
        
        if wm > 0:
            self.wx = max(3, int(wm / sd.csx))
            self.wy = max(3, int(wm / sd.csy))
        else:
            self.wx = 3
            self.wy = 3
        
        if self.wx % 2 == 0:
            self.wx += 1
        if self.wy % 2 == 0:
            self.wy += 1

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise InterruptedError("Cancelled by user")

    def padded(self) -> np.ndarray:
        self.check()
        
        if self._padded is None:
            with self._pad_lock:
                if self._padded is None:
                    dem, _, _ = self.sd.get()
                    self._padded = np.pad(dem, 1, mode='reflect')
        
        return self._padded

    def window_size(self) -> int:
        return get_window_size(self.wx, self.wy)

    def cleanup(self) -> None:
        self._padded = None
        gc.collect()

    def run(self) -> None:
        try:
            self.check()
            r = self.fn(self)
            r["doc"] = SCIENTIFIC_DOCS.get(self.dk, {})
            r["metadata"] = self.metadata
            self.finished.emit(r)
        except InterruptedError:
            self.status.emit("Cancelled")
        except (ValidationError, ComputationError, MemoryError) as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {str(e)}")
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {str(e)}")
                QgsMessageLog.logMessage(traceback.format_exc(), "Geomorph", Qgis.Critical)
        finally:
            self.cleanup()

class GeomorphologyTool:
    def __init__(self, iface):
        self.iface = iface
        self.raster: Optional[QgsRasterLayer] = None
        self.sd = SharedData()
        self.rows = 0
        self.cols = 0
        self.crs = None
        self.w: Optional[Worker] = None
        self.settings = QSettings("GeomorphologyTool", "Settings")

    def _save_setting(self, dk: str, key: str, value: Any) -> None:
        self.settings.setValue(f"{dk}/{key}", value)

    def load(self) -> bool:
        layer = self.iface.activeLayer()
        
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.critical(None, "Error", "Please select a DEM raster layer!")
            return False
        
        self.raster = layer
        self.crs = layer.crs()
        
        if self.crs.isGeographic():
            r = QMessageBox.question(None, "Geographic CRS Detected",
                "DEM uses geographic coordinates. Reproject to UTM?\n\nYes = Auto UTM\nNo = Keep CRS\nCancel = Abort",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            
            if r == QMessageBox.Yes:
                if not self._reproj():
                    return False
            elif r == QMessageBox.Cancel:
                return False
        
        ext = layer.extent()
        self.cols = layer.width()
        self.rows = layer.height()
        
        if self.cols == 0 or self.rows == 0 or ext.isEmpty():
            QMessageBox.critical(None, "Error", "Invalid layer dimensions or extent!")
            return False
        
        csx = ext.width() / self.cols
        csy = ext.height() / self.rows
        total = self.cols * self.rows
        
        if total > MAX_CELLS_HARD:
            QMessageBox.critical(None, "Too Large", f"DEM has {total:,} cells (max {MAX_CELLS_HARD:,}).")
            return False
        
        if total > MAX_CELLS_WARN:
            if QMessageBox.warning(None, "Large DEM", f"DEM has {total:,} cells. Processing may be slow.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return False
        
        try:
            from osgeo import gdal
            path = self.raster.source()
            
            if path:
                ds = gdal.Open(path)
                if ds:
                    band = ds.GetRasterBand(1)
                    dem = band.ReadAsArray().astype(np.float32)
                    nd = band.GetNoDataValue()
                    if nd is not None:
                        dem[dem == nd] = np.nan
                    ds = None
                else:
                    raise Exception("GDAL could not open raster")
            else:
                raise Exception("No raster file path available")
        except Exception as e:
            logger.warning(f"GDAL read failed, using block reading: {e}")
            provider = layer.dataProvider()
            block = provider.block(1, ext, self.cols, self.rows)
            dem = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
            nd = provider.sourceNoDataValue(1)
            
            for i in range(self.rows):
                for j in range(self.cols):
                    v = block.value(i, j)
                    if v is not None and v != nd and not math.isnan(v):
                        dem[i, j] = float(v)
        
        if np.all(np.isnan(dem)):
            QMessageBox.critical(None, "Error", "All cells are NoData!")
            return False
        
        mask = ~np.isnan(dem)
        self.sd.init(dem, mask, csx, csy, ext, self.crs)
        return True

    def _reproj(self) -> bool:
        try:
            from qgis import processing
            
            center = self.raster.extent().center()
            zone = int((center.x() + 180) / 6) + 1
            hemisphere = 'N' if center.y() >= 0 else 'S'
            epsg = f"EPSG:{32600 + zone if hemisphere == 'N' else 32700 + zone}"
            
            result = processing.run("gdal:warpreproject", {
                'INPUT': self.raster,
                'TARGET_CRS': QgsCoordinateReferenceSystem(epsg),
                'RESAMPLING': 1,
                'OUTPUT': 'memory:'
            })
            
            reprojected = result['OUTPUT']
            if not reprojected.isValid():
                raise Exception("Invalid reprojection result")
            
            reprojected.setName(f"{self.raster.name()}_UTM")
            QgsProject.instance().addMapLayer(reprojected)
            self.iface.setActiveLayer(reprojected)
            self.raster = reprojected
            return True
        except Exception as e:
            QMessageBox.critical(None, "Reprojection Error", str(e))
            return False

    def save(self, arr: np.ndarray, name: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            from osgeo import gdal
            
            temp_dir = tempfile.gettempdir()
            if not os.access(temp_dir, os.W_OK):
                raise PermissionError(f"Cannot write to temporary directory: {temp_dir}")
            
            path = os.path.join(temp_dir, f"geomorph_{name}_{np.random.randint(0, 10000)}.tif")
            
            src = gdal.Open(self.raster.source()) if self.raster.source() else None
            if src:
                gt = src.GetGeoTransform()
                pr = src.GetProjection()
                src = None
            else:
                gt = [self.sd.ext.xMinimum(), self.sd.csx, 0, self.sd.ext.yMaximum(), 0, -self.sd.csy]
                pr = self.sd.crs.toWkt()
            
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(path, self.cols, self.rows, 1, gdal.GDT_Float32, options=GEOTIFF_CREATION_OPTIONS)
            
            if ds is None:
                raise Exception("Failed to create GeoTIFF")
            
            ds.SetGeoTransform(gt)
            ds.SetProjection(pr)
            
            base_meta = {
                'TIFFTAG_SOFTWARE': f'Geomorphology Analysis Tool v{__version__}',
                'TIFFTAG_DATETIME': datetime.now().isoformat(),
                'ALGORITHM_VERSION': __version__,
                'DEM_SOURCE': self.raster.source(),
                'DEM_RESOLUTION_X': f"{self.sd.csx:.4f}",
                'DEM_RESOLUTION_Y': f"{self.sd.csy:.4f}",
                'CRS': self.sd.crs.authid(),
                'COMPRESSION': 'LZW'
            }
            base_meta.update(metadata or {})
            
            for key, value in base_meta.items():
                ds.SetMetadataItem(key, str(value))
            
            band = ds.GetRasterBand(1)
            band.SetNoDataValue(NO_DATA_VALUE)
            save_arr = np.where(np.isnan(arr), NO_DATA_VALUE, arr).astype(np.float32)
            band.WriteArray(save_arr)
            
            ds.FlushCache()
            ds = None
            
            layer = QgsRasterLayer(path, name)
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                return layer
            else:
                raise Exception("Invalid raster layer created")
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Failed to save raster: {str(e)}")
            raise

    def show(self) -> None:
        if not self.sd.ready() and not self.load():
            return
        
        if not self.sd.ready():
            QMessageBox.critical(None, "Error", "Initialization failed!")
            return
        
        cols, rows, valid, dmin, dmax = self.sd.stats()
        dx, dy = self.sd.csx, self.sd.csy
        
        dlg = QDialog()
        dlg.setWindowTitle(f"Geomorphology v{__version__}")
        dlg.setMinimumSize(650, 720)
        layout = QVBoxLayout(dlg)
        
        hdr = QLabel(f"""
╔══════════════════════════════════════════════╗
║  GEOMORPHOLOGY v{__version__} – 12 INDICES          ║
╚══════════════════════════════════════════════╝
📊 {self.raster.name()} | {cols:,}×{rows:,} | {dx:.2f}×{dy:.2f} m
🗺️ {self.crs.authid()} | ✅ {valid:,} valid | ⛰️ {dmin:.1f} – {dmax:.1f} m
""")
        hdr.setStyleSheet("font-family:monospace;font-size:9px;padding:10px;background:#f8f9fa;border:1px solid #ddd;border-radius:5px;")
        layout.addWidget(hdr)
        
        scroll = QScrollArea()
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setSpacing(3)
        
        indices = [
            ("1. Topographic Position Index [Original]", "tpi"),
            ("2. Terrain Ruggedness Index [Original]", "tri_classic"),
            ("3. Surface Roughness [Original]", "roughness"),
            ("4. Terrain Curvature [Original]", "curvature"),
            ("5. Relative Relief [Original]", "relief"),
            ("6. Moving Window Hypsometric Integral [Adaptation]", "hypsometric_integral"),
            ("7. Gaussian Multiscale TPI [Modified]", "multiscale_tpi"),
            ("8. Local Relief Model [Original]", "lrm"),
            ("9. Positive Openness [Approximation]", "positive_openness"),
            ("10. Negative Openness [Approximation]", "negative_openness"),
            ("11. Local Base Level Approximation [Approximation]", "lbl"),
            ("12. Moment-based Lacunarity [Approximation]", "lacunarity"),
        ]
        
        btn_style = """
            QPushButton { text-align: left; padding: 10px 15px; margin: 1px 5px;
            border: 1px solid #ddd; border-radius: 4px; font-size: 11px; background: #fafafa; }
            QPushButton:hover { background: #e3f2fd; border-color: #2196f3; }
        """
        
        for display, dk in indices:
            btn = QPushButton(f"  {display}")
            btn.setStyleSheet(btn_style)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, d=dk, n=display: self._show_algorithm_dialog(d, n, dlg))
            sl.addWidget(btn)
        
        scroll.setWidget(sw)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        ft = QLabel(f"v{__version__} | 12 indices | Type-safe & validated | LZW compression")
        ft.setStyleSheet("font-size:10px;color:#666;padding:10px;background:#f0f0f0;border-top:1px solid #ddd;")
        ft.setAlignment(Qt.AlignCenter)
        layout.addWidget(ft)
        
        cb = QPushButton("Close")
        cb.setStyleSheet("padding:12px;background:#6c757d;color:white;font-weight:bold;border-radius:5px;")
        cb.clicked.connect(dlg.close)
        layout.addWidget(cb)
        
        dlg.exec_()

    def _create_settings_widget(self, dk: str, doc: Dict[str, Any], form: QFormLayout) -> Dict[str, QWidget]:
        widgets = {}
        cs = min(self.sd.csx, self.sd.csy)
        
        def add_window_spin(label, default_factor, min_f=3, max_f=201, step_f=1):
            sp = QSpinBox()
            sp.setRange(int(cs * min_f), int(cs * max_f))
            sp.setValue(int(cs * default_factor))
            sp.setSingleStep(int(cs * step_f))
            sp.setSuffix(" m")
            info = QLabel(f"(≈ {int(sp.value()/self.sd.csx)}×{int(sp.value()/self.sd.csy)} cells)")
            info.setStyleSheet("color:#666;font-size:10px;")
            sp.valueChanged.connect(lambda v, lbl=info: lbl.setText(f"(≈ {int(v/self.sd.csx)}×{int(v/self.sd.csy)} cells)"))
            form.addRow(label, sp)
            form.addRow("", info)
            return sp
        
        if dk == "multiscale_tpi":
            scales = QLineEdit(self.settings.value(f"{dk}/scales", DEFAULT_SCALES_M))
            form.addRow("Analysis Scales (m):", scales)
            fusion = QComboBox()
            fusion.addItems(["mean", "median", "max_abs", "weighted"])
            fusion.setCurrentText(self.settings.value(f"{dk}/fusion", "mean"))
            form.addRow("Fusion Method:", fusion)
            widgets.update(scales=scales, fusion=fusion)
        elif dk == "curvature":
            wm = add_window_spin("Window Size:", WINDOW_FACTORS["default"])
            smooth = QDoubleSpinBox()
            smooth.setRange(0, 5.0)
            smooth.setValue(self.settings.value(f"{dk}/smooth_sigma", DEFAULT_CURVATURE_SMOOTH, type=float))
            smooth.setSingleStep(0.1)
            form.addRow("Pre-smoothing σ (px):", smooth)
            widgets.update(wm=wm, smooth=smooth)
        elif dk in ("lrm", "lbl"):
            factor = WINDOW_FACTORS.get(dk, 25)
            wm = add_window_spin("Trend Window:" if dk == "lrm" else "Window Size:", factor, 10, 1000, 5)
            tip = "Gaussian trend (σ = window/3)" if dk == "lrm" else "Local minimum. Use DEM − LBL for residual height"
            info = QLabel(f"💡 {tip}")
            info.setStyleSheet("color:#666;font-size:10px;font-style:italic;")
            form.addRow("", info)
            widgets["wm"] = wm
        elif dk in ("positive_openness", "negative_openness"):
            rad = QSpinBox()
            rad.setRange(3, 50)
            rad.setValue(min(self.settings.value(f"{dk}/openness_radius", DEFAULT_OPENNESS_RADIUS, type=int), 50))
            form.addRow("Search Radius (cells):", rad)
            
            dirs = QSpinBox()
            dirs.setRange(4, 32)
            dirs.setSingleStep(4)
            dirs.setValue(self.settings.value(f"{dk}/openness_directions", DEFAULT_OPENNESS_DIRECTIONS, type=int))
            form.addRow("Number of Directions:", dirs)
            
            info = QLabel(f"Effective radius: {rad.value()*cs:.0f} m | Directions: {dirs.value()}")
            info.setStyleSheet("color:#666;font-size:10px;")
            rad.valueChanged.connect(lambda v: info.setText(f"Effective radius: {v*cs:.0f} m | Directions: {dirs.value()}"))
            dirs.valueChanged.connect(lambda v: info.setText(f"Effective radius: {rad.value()*cs:.0f} m | Directions: {v}"))
            form.addRow("", info)
            
            note = QLabel("💡 Directional horizon approximation")
            note.setStyleSheet("color:#e67e22;font-size:10px;font-style:italic;")
            form.addRow("", note)
            widgets.update(rad=rad, dirs=dirs)
        elif dk == "lacunarity":
            win = QSpinBox()
            win.setRange(15, 501)
            win.setValue(self.settings.value(f"{dk}/lac_window", DEFAULT_LACUNARITY_WINDOW, type=int))
            win.setSingleStep(10)
            win.setSuffix(" px")
            form.addRow("Moving Window Size:", win)
            
            step = QSpinBox()
            step.setRange(3, 100)
            step.setValue(self.settings.value(f"{dk}/lac_step", DEFAULT_LACUNARITY_STEP, type=int))
            step.setSingleStep(5)
            step.setSuffix(" px")
            form.addRow("Sampling Step:", step)
            
            boxes = QLineEdit(self.settings.value(f"{dk}/lac_boxes", DEFAULT_LACUNARITY_BOXES))
            form.addRow("Box Sizes:", boxes)
            
            minv = QDoubleSpinBox()
            minv.setRange(0.1, 1.0)
            minv.setValue(self.settings.value(f"{dk}/lac_min_valid", DEFAULT_LACUNARITY_MIN_VALID, type=float))
            minv.setSingleStep(0.05)
            form.addRow("Min Valid Fraction:", minv)
            widgets.update(win=win, step=step, boxes=boxes, minv=minv)
        else:
            factor = WINDOW_FACTORS.get(dk if dk in WINDOW_FACTORS else "hypsometric" if dk == "hypsometric_integral" else "default", 9)
            wm = add_window_spin("Window Size:", factor)
            widgets["wm"] = wm
        
        return widgets

    def _create_info_tabs(self, doc: Dict[str, Any]) -> QTabWidget:
        tabs = QTabWidget()
        
        qtab = QWidget()
        ql = QVBoxLayout(qtab)
        ql.addWidget(QLabel("<h3>📋 Quick Interpretation Guide</h3>"))
        for point in doc.get('how_to_interpret', []):
            lab = QLabel(f"• {point}")
            lab.setWordWrap(True)
            lab.setStyleSheet("padding:5px;font-size:11px;")
            ql.addWidget(lab)
        ql.addStretch()
        tabs.addTab(qtab, "📋 Quick Guide")
        
        stab = QWidget()
        sl = QVBoxLayout(stab)
        scroll = QScrollArea()
        content = QWidget()
        cl = QVBoxLayout(content)
        
        for label, key in [("📖 Description", "description"), ("🔢 Formula", "formula"), ("📊 Interpretation", "interpretation")]:
            if key in doc:
                cl.addWidget(QLabel(f"<b style='color:#e67e22;'>{label}</b>"))
                t = QLabel(doc[key])
                t.setWordWrap(True)
                t.setStyleSheet("padding:5px;margin-bottom:8px;")
                cl.addWidget(t)
        
        grid = QGridLayout()
        if 'units' in doc:
            grid.addWidget(QLabel("<b>Units:</b>"), 0, 0)
            grid.addWidget(QLabel(doc['units']), 0, 1)
        if 'output_range' in doc:
            grid.addWidget(QLabel("<b>Typical Range:</b>"), 1, 0)
            grid.addWidget(QLabel(doc['output_range']), 1, 1)
        cl.addLayout(grid)
        cl.addStretch()
        
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        sl.addWidget(scroll)
        tabs.addTab(stab, "📖 Science")
        
        atab = QWidget()
        al = QVBoxLayout(atab)
        for label, key in [("🎯 Typical Applications", "applications"), ("💡 Practical Use", "practical_use")]:
            if key in doc:
                al.addWidget(QLabel(f"<b style='color:#27ae60;'>{label}</b>"))
                t = QLabel(doc[key])
                t.setWordWrap(True)
                t.setStyleSheet("padding:5px;margin-bottom:8px;")
                al.addWidget(t)
        al.addStretch()
        tabs.addTab(atab, "🎯 Applications")
        
        ptab = QWidget()
        pl = QVBoxLayout(ptab)
        for label, key, color in [("✅ Advantages", "advantages", "#2980b9"), ("⚠️ Limitations", "limitations", "#e74c3c"), ("📏 Typical Scale", "typical_scale", "#2980b9")]:
            if key in doc:
                pl.addWidget(QLabel(f"<b style='color:{color};'>{label}</b>"))
                t = QLabel(doc[key])
                t.setWordWrap(True)
                t.setStyleSheet("padding:5px;margin-bottom:8px;")
                pl.addWidget(t)
        pl.addStretch()
        tabs.addTab(ptab, "⚖️ Pros/Cons")
        
        rtab = QWidget()
        rl = QVBoxLayout(rtab)
        for label, key in [("📚 Citation", "citation"), ("📝 Scientific Notes", "scientific_notes"), ("🔧 Implementation", "implementation_note"), ("🎨 Recommended Color Ramp", "recommended_color_ramp")]:
            if key in doc:
                rl.addWidget(QLabel(f"<b style='color:#8e44ad;'>{label}</b>"))
                text = doc[key]
                if key == "recommended_color_ramp":
                    text += "<br><i style='color:#666;font-size:10px;'>Note: Tool does not apply style automatically – set manually in Layer Properties → Symbology.</i>"
                t = QLabel(text)
                t.setWordWrap(True)
                style = "padding:5px;margin-bottom:8px;"
                if key == "scientific_notes":
                    style += "background:#fdf2e9;border:1px solid #f5cba7;border-radius:3px;"
                elif key == "recommended_color_ramp":
                    style += "background:#e8f4fd;border:1px solid #a8d4f0;border-radius:3px;"
                t.setStyleSheet(style)
                rl.addWidget(t)
        rl.addStretch()
        tabs.addTab(rtab, "📚 Reference")
        
        return tabs

    def _format_statistics(self, stats: Dict[str, Any], doc: Dict[str, Any]) -> str:
        if isinstance(stats, dict) and "mean" in stats:
            units = doc.get("units", "")
            return (f"📊 <b>Statistics</b> [{units}]<br>"
                    f"├─ Mean: <b>{stats.get('mean', 0):.4f}</b><br>"
                    f"├─ Median: <b>{stats.get('median', 0):.4f}</b><br>"
                    f"├─ Std: <b>{stats.get('std', 0):.4f}</b><br>"
                    f"├─ Min: <b>{stats.get('min', 0):.4f}</b><br>"
                    f"├─ Max: <b>{stats.get('max', 0):.4f}</b><br>"
                    f"└─ Valid: <b>{stats.get('count', 0):,}</b>")
        elif isinstance(stats, dict):
            txt = "📊 <b>Statistics</b><br>"
            for key, ss in stats.items():
                if isinstance(ss, dict) and "mean" in ss:
                    txt += (f"<br><b>{key}:</b><br>"
                           f"├─ Mean: <b>{ss.get('mean', 0):.4f}</b><br>"
                           f"├─ Std: <b>{ss.get('std', 0):.4f}</b><br>"
                           f"└─ Valid: <b>{ss.get('count', 0):,}</b>")
            return txt
        return ""

    def _show_algorithm_dialog(self, dk: str, name: str, parent: QDialog) -> None:
        self._cancel()
        
        doc = SCIENTIFIC_DOCS.get(dk, {})
        func = ALGORITHMS.get(dk)
        
        if func is None:
            QMessageBox.critical(parent, "Error", f"Algorithm not found: {dk}")
            return
        
        dlg = QDialog(parent)
        dlg.setWindowTitle(f"📊 {name}")
        dlg.setMinimumSize(850, 800)
        layout = QVBoxLayout(dlg)
        
        layout.addWidget(QLabel(f"<h2>{name}</h2>"))
        
        type_colors = {'Original implementation': '#27ae60', 'Modified implementation': '#2980b9', 'Moving-window adaptation': '#2980b9', 'Approximation': '#e67e22'}
        algo_type = doc.get('algorithm_type', 'Unknown')
        layout.addWidget(QLabel(f"<span style='color:{type_colors.get(algo_type, '#666')};font-weight:bold;'>[{algo_type}]</span>"))
        
        tabs = self._create_info_tabs(doc)
        layout.addWidget(tabs)
        
        calc_group = QGroupBox("⚙️ Calculation Settings")
        form = QFormLayout(calc_group)
        widgets = self._create_settings_widget(dk, doc, form)
        
        base = os.path.splitext(os.path.basename(self.raster.source() or "dem"))[0]
        default_name = make_output_name(dk, base)
        
        if "wm" in widgets:
            default_name = make_output_name(dk, base, window_m=widgets["wm"].value())
        elif dk == "lacunarity":
            default_name = make_output_name(dk, base, extra=f"W{widgets['win'].value()}")
        elif dk in ("positive_openness", "negative_openness"):
            default_name = make_output_name(dk, base, extra=f"R{widgets['rad'].value()}")
        
        name_edit = QLineEdit(default_name[:DEFAULT_NAME_MAX_LENGTH])
        form.addRow("Output Name:", name_edit)
        widgets["name"] = name_edit
        
        layout.addWidget(calc_group)
        
        pb = QProgressBar()
        layout.addWidget(pb)
        
        status = QLabel("Ready")
        status.setStyleSheet("color:#666;font-style:italic;")
        layout.addWidget(status)
        
        stats_lab = QLabel("")
        stats_lab.setStyleSheet("background:#e8f5e9;padding:10px;border-radius:4px;font-family:monospace;border:1px solid #a5d6a7;")
        stats_lab.setVisible(False)
        stats_lab.setWordWrap(True)
        layout.addWidget(stats_lab)
        
        btn_row = QHBoxLayout()
        
        calc_btn = QPushButton("▶ Calculate")
        calc_btn.setStyleSheet("background:#28a745;color:white;padding:12px 30px;font-weight:bold;border-radius:5px;")
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding:12px 20px;background:#95a5a6;color:white;border-radius:5px;")
        cancel_btn.clicked.connect(lambda: (self._cancel(), dlg.close()))
        
        btn_row.addWidget(calc_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        
        def start_calculation():
            if not self.sd.ready():
                QMessageBox.critical(dlg, "Error", "DEM not loaded!")
                return
            
            calc_btn.setEnabled(False)
            pb.setValue(0)
            status.setText("⏳ Calculating...")
            stats_lab.setVisible(False)
            
            params = {}
            meta_extra = {}
            wm_val = 0
            cs = min(self.sd.csx, self.sd.csy)
            
            if dk == "multiscale_tpi":
                self._save_setting(dk, "scales", widgets["scales"].text())
                self._save_setting(dk, "fusion", widgets["fusion"].currentText())
                params = {'Scales_m': widgets["scales"].text(), 'Fusion_Method': widgets["fusion"].currentText()}
                meta_extra = params
            elif dk == "curvature":
                self._save_setting(dk, "smooth_sigma", widgets["smooth"].value())
                wm_val = widgets["wm"].value()
                params = {'Window_m': wm_val, 'Smooth_sigma': widgets["smooth"].value()}
                meta_extra = params
            elif dk in ("lrm", "lbl"):
                wm_val = widgets["wm"].value()
                params = {'Window_m': wm_val}
                if dk == "lrm":
                    params['Sigma'] = f"{wm_val / LRM_GAUSSIAN_FACTOR:.1f}"
                meta_extra = params
            elif dk in ("positive_openness", "negative_openness"):
                r, d = widgets["rad"].value(), widgets["dirs"].value()
                self._save_setting(dk, "openness_radius", r)
                self._save_setting(dk, "openness_directions", d)
                params = {'Radius_cells': r, 'Num_Directions': d, 'Method': 'Directional horizon approximation'}
                wm_val = r * int(cs)
                meta_extra = params
            elif dk == "lacunarity":
                self._save_setting(dk, "lac_window", widgets["win"].value())
                self._save_setting(dk, "lac_step", widgets["step"].value())
                self._save_setting(dk, "lac_boxes", widgets["boxes"].text())
                self._save_setting(dk, "lac_min_valid", widgets["minv"].value())
                params = {'Window_px': widgets["win"].value(), 'Step_px': widgets["step"].value(), 'Box_Sizes': widgets["boxes"].text(), 'Min_Valid_Fraction': widgets["minv"].value()}
                meta_extra = {'lac_window': widgets["win"].value(), 'lac_step': widgets["step"].value(), 'lac_boxes': widgets["boxes"].text(), 'lac_min_valid': widgets["minv"].value()}
                wm_val = widgets["win"].value() * int(cs)
            elif "wm" in widgets:
                wm_val = widgets["wm"].value()
                params = {'Window_m': wm_val}
                meta_extra = params
            
            metadata = create_metadata(dk, doc, params)
            metadata.update(meta_extra)
            
            try:
                self.w = Worker(func, dk, self.sd, self.cols, self.rows, wm_val, widgets["name"].text(), metadata)
            except Exception as e:
                QMessageBox.critical(dlg, "Error", f"Failed to create worker: {e}")
                calc_btn.setEnabled(True)
                return
            
            self.w.progress.connect(pb.setValue)
            self.w.status.connect(status.setText)
            
            def on_finished(r):
                calc_btn.setEnabled(True)
                pb.setValue(100)
                status.setText("✅ Complete!")
                
                try:
                    if "array" in r:
                        self.save(r["array"], r["name"], r.get("metadata", {}))
                    elif "arrays" in r:
                        for k, arr in r["arrays"].items():
                            self.save(arr, f"{r['name']}_{k}", r.get("metadata", {}))
                    
                    if "stats" in r:
                        txt = self._format_statistics(r["stats"], doc)
                        if txt:
                            stats_lab.setText(txt)
                            stats_lab.setVisible(True)
                    
                    QMessageBox.information(dlg, "Success", f"✅ {r['name']}\n\nAdded to map layers.\nCheck layer properties for metadata.")
                except Exception as e:
                    QMessageBox.critical(dlg, "Save Error", f"Calculation succeeded but save failed: {e}")
                
                self.w = None
            
            def on_error(e):
                calc_btn.setEnabled(True)
                status.setText("❌ Error!")
                QMessageBox.critical(dlg, "Calculation Error", f"Error: {e}\n\nCheck DEM integrity and memory.\nSee QGIS Log for details.")
                self.w = None
            
            self.w.finished.connect(on_finished)
            self.w.error.connect(on_error)
            self.w.start()
        
        calc_btn.clicked.connect(start_calculation)
        dlg.exec_()
        self._cancel()

    def _cancel(self) -> None:
        if self.w is not None:
            if self.w.isRunning():
                self.w.cancel()
                self.w.wait(3000)
            self.w.cleanup()
            self.w = None
            gc.collect()

try:
    tool = GeomorphologyTool(iface)
    tool.show()
except NameError:
    QMessageBox.critical(None, "QGIS Required",
        "This tool must be run from within QGIS Python Console.\n\n"
        "Please open QGIS, select a DEM layer, and run this script\n"
        "from the Python Console (Plugins → Python Console).")
except Exception as e:
    QMessageBox.critical(None, "Startup Error", f"Failed to initialize Geomorphology Tool:\n{str(e)}")
    QgsMessageLog.logMessage(f"Startup error: {e}", "Geomorph", Qgis.Critical)
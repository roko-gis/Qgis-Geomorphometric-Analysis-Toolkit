# Qgis Geomorphometric Analysis Toolkit v1.0.0

## An Open-Source QGIS Tool for Multi-Scale Terrain Analysis Using Digital Elevation Models (DEMs)

---

# Overview

**Qgis Geomorphometric Analysis Toolkit** is an open-source QGIS application for advanced terrain analysis using Digital Elevation Models (DEMs).

The project provides a curated collection of **12 scientifically documented geomorphometric indices**, selected to extend conventional terrain analysis workflows with less commonly available but highly informative measures of terrain structure, landform organization, surface complexity, and landscape evolution.

Unlike many standard GIS terrain analysis tools that focus primarily on widely used derivatives such as slope, aspect, curvature, and basic terrain measures, this toolkit emphasizes a selection of more specialized geomorphometric indices that can reveal subtle spatial patterns, multi-scale terrain organization, and geomorphic processes.

The toolkit combines established geomorphometric approaches with optimized numerical implementations using **Python, NumPy, SciPy, GDAL, and QGIS APIs**, providing researchers and GIS practitioners with reproducible methods for advanced DEM-based landscape analysis.
---

# Project Goals

The main objectives of this project are:

* Provide accessible geomorphometric analysis tools inside QGIS
* Implement scientifically documented terrain indices
* Support reproducible DEM analysis workflows
* Provide transparent algorithms with formulas and references
* Enable researchers and GIS professionals to analyse terrain morphology without proprietary software

---

# Main Applications

The toolkit can support research and applications in:

* Geomorphology
* Landscape evolution studies
* Digital terrain modelling
* Remote sensing
* LiDAR analysis
* Hydrological modelling
* Landslide susceptibility assessment
* Archaeological landscape analysis
* Environmental monitoring


---

# Features

## Multi-scale Terrain Analysis

The toolkit supports analysis of terrain characteristics at multiple spatial scales:

* Micro-topography
* Local landforms
* Hillslope morphology
* Regional terrain organization

Analysis windows are defined in metric units and automatically converted into DEM pixel dimensions.

---

## Scientific Transparency

Each algorithm includes:

* Scientific background
* Mathematical formulation
* Interpretation guidance
* Typical value ranges
* Applications
* Advantages
* Limitations
* Recommended visualization approach

Scientific information is available directly inside the QGIS interface.

---

# Implemented Terrain Indices

The current release contains:

| No. | Index                              | Type                    |
| --- | ---------------------------------- | ----------------------- |
| 1   | Topographic Position Index (TPI)   | Original implementation |
| 2   | Terrain Ruggedness Index (TRI)     | Original implementation |
| 3   | Surface Roughness                  | Original implementation |
| 4   | Terrain Curvature                  | Original implementation |
| 5   | Relative Relief                    | Original implementation |
| 6   | Moving Window Hypsometric Integral | Adaptation              |
| 7   | Gaussian Multiscale TPI            | Modified implementation |
| 8   | Local Relief Model (LRM)           | Original implementation |
| 9   | Positive Openness                  | Approximation           |
| 10  | Negative Openness                  | Approximation           |
| 11  | Local Base Level Approximation     | Approximation           |
| 12  | Moment-Based Lacunarity            | Approximation           |

---

---

# Software Architecture

The toolkit is developed using:

* Python
* QGIS Python API
* PyQt interface framework
* NumPy numerical processing
* SciPy spatial filtering
* GDAL raster handling

Main components:

```
Geomorphometric Toolkit

├── DEM loading and validation
├── Shared raster memory management
├── Terrain analysis algorithms
├── Background processing workers
├── GeoTIFF export
├── Metadata generation
└── QGIS user interface
```

---

# Output Products

The toolkit generates GeoTIFF raster products containing:

* Spatial reference information
* Processing parameters
* Algorithm metadata
* Version information
* DEM information

Output files use:

```
GeoTIFF
LZW compression
Floating point format
NoData handling
```

---

# Data Requirements

Recommended input:

* Digital Elevation Models
* Projected coordinate systems
* Metric map units
* Valid NoData definitions

Supported raster formats:

* GeoTIFF
* Cloud Optimized GeoTIFF
* GDAL-compatible raster formats

---

# Performance and Validation

The toolkit includes:

* DEM validation
* NaN-safe calculations
* Memory protection
* Progress monitoring
* Background computation
* Error handling
* Metadata preservation

Large raster protection:

```
Maximum supported cells:
200,000,000
```

Actual performance depends on:

* DEM resolution
* Analysis window size
* Available RAM
* CPU performance

---

# Author and Contact

## Author

**Rosen Iliev**

## Contact

Email:

```
ilievrosen88@abv.bg
```
---


# Satellite Geospatial Deep Learning

Implementation of the satellite-image preprocessing workflow described in:

**"Understanding Satellite Image For Geo-spatial Deep Learning"**  
by Fractal AI.

## Project Overview

This project implements a geospatial computer-vision preprocessing pipeline for building-footprint segmentation from high-resolution satellite imagery.

The workflow takes a GeoTIFF satellite image and corresponding GeoJSON building annotations and prepares aligned image/mask tiles suitable for deep-learning workflows.

## Workflow

1. Load and inspect a high-resolution GeoTIFF using Rasterio.
2. Reduce image resolution from approximately 0.07748 m/pixel to 0.1 m/pixel using GDAL.
3. Visualize RGB satellite imagery.
4. Load building-footprint annotations from GeoJSON.
5. Reproject building geometries to the satellite image CRS.
6. Rasterize building polygons into a pixel-level building mask.
7. Save the generated mask as a GeoTIFF.
8. Split the satellite image into 1024 × 1024 tiles.
9. Split the corresponding building mask into matching 1024 × 1024 tiles.
10. Visualize corresponding satellite-image and building-mask tile pairs.

## Dataset

The project uses the `33cae6` scene from Tier 1 of the DrivenData Open Cities AI Challenge dataset.

Input data:

- High-resolution GeoTIFF satellite imagery
- GeoJSON building-footprint annotations
- 4,439 annotated building footprints

The original image is:

- 37,113 × 34,306 pixels
- 4 bands
- CRS: EPSG:32737
- Resolution: approximately 0.07748 m/pixel

## Technologies

- Python
- Rasterio
- GeoPandas
- Shapely
- NumPy
- Matplotlib
- GDAL
- Jupyter Notebook
- Git / GitHub

## Repository Structure

```text
satellite-geospatial-dl/
├── notebooks/
│   └── 01_satellite_image_exploration.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   ├── images/
│   └── masks/
├── outputs/
├── src/
├── .gitignore
└── README.md

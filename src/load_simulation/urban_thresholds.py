"""
Classify market locations in Nigeria into Urban / Peri-urban / Rural
using GHSL built-up surface data.

For each market, computes built-up intensity statistics within a 3 km buffer
and classifies markets using quantile-based thresholds.
"""

import pandas as pd
import geopandas as gpd
import rasterio  # type: ignore
from rasterio.mask import mask  # type: ignore
from shapely.geometry import Point
import numpy as np
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# File paths (relative to project root)
MARKETS_CSV = Path(os.environ.get("PAPER2_RAW_MARKETS", "markets_filtered_q1_density_cap.csv"))
RASTER_DIR = Path(os.environ.get("PAPER2_GHSL_RASTER_DIR", "nigeria_tif"))
OUTPUT_CSV = Path(os.environ.get("PAPER2_PREPARED_MARKETS_OUT", "markets_with_urban_class_3km.csv"))

# Buffer size in meters
BUFFER_SIZE_M = 3000

# Quantile thresholds for classification
URBAN_QUANTILE = 0.80  # 80th percentile
PERI_URBAN_QUANTILE = 0.40  # 40th percentile

# Pixel area in square meters (100m × 100m)
PIXEL_AREA_M2 = 10000

# Test mode: limit number of markets to process (set to None to process all)
MAX_MARKETS = None  # Set to None for full processing


# ============================================================================
# Helper Functions
# ============================================================================

def load_markets(csv_path):
    """
    Load market CSV and create GeoDataFrame with point geometries.
    
    Parameters
    ----------
    csv_path : Path
        Path to the markets CSV file
        
    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame with point geometries in EPSG:4326
    """
    print(f"Loading markets from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean data: remove rows with invalid coordinates
    initial_count = len(df)
    df = df.dropna(subset=['x', 'y'])
    
    # Filter out invalid coordinate ranges (Nigeria is roughly 2-15°E, 4-14°N)
    df = df[(df['x'] >= 2) & (df['x'] <= 15) & (df['y'] >= 4) & (df['y'] <= 15)]
    
    if len(df) < initial_count:
        print(f"  Removed {initial_count - len(df)} rows with invalid coordinates")
    
    # Create point geometries from x (longitude) and y (latitude)
    geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    print(f"Loaded {len(gdf)} markets")
    return gdf


def get_raster_crs(raster_dir):
    """
    Read CRS from the first raster file found.
    
    Parameters
    ----------
    raster_dir : Path
        Directory containing raster files
        
    Returns
    -------
    crs : CRS
        Coordinate reference system of the rasters
    """
    raster_files = list(raster_dir.glob("*.tif"))
    if not raster_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {raster_dir}")
    
    # Read CRS from first raster
    with rasterio.open(raster_files[0]) as src:
        crs = src.crs
        print(f"Raster CRS: {crs}")
    return crs


def get_intersecting_rasters(buffer_geom, raster_dir, raster_crs):
    """
    Find raster files that intersect with the buffer geometry.
    
    Parameters
    ----------
    buffer_geom : shapely.geometry
        Buffer geometry in raster CRS
    raster_dir : Path
        Directory containing raster files
    raster_crs : CRS
        CRS of the raster files
        
    Returns
    -------
    intersecting_rasters : list
        List of Path objects for intersecting raster files
    """
    intersecting_rasters = []
    
    for raster_path in raster_dir.glob("*.tif"):
        with rasterio.open(raster_path) as src:
            # Get raster bounds
            raster_bounds = src.bounds
            
            # Create a box from raster bounds
            from shapely.geometry import box
            raster_box = box(raster_bounds.left, raster_bounds.bottom,
                           raster_bounds.right, raster_bounds.top)
            
            # Check if buffer intersects raster bounds
            if buffer_geom.intersects(raster_box):
                intersecting_rasters.append(raster_path)
    
    return intersecting_rasters


def extract_builtup_values(buffer_geom, raster_paths, raster_crs):
    """
    Extract built-up values from all intersecting raster tiles.
    
    Parameters
    ----------
    buffer_geom : shapely.geometry
        Buffer geometry in raster CRS
    raster_paths : list
        List of Path objects for raster files
    raster_crs : CRS
        CRS of the raster files
        
    Returns
    -------
    values : numpy.ndarray
        Flattened array of built-up values (excluding NoData and zeros)
    """
    all_values = []
    
    for raster_path in raster_paths:
        try:
            with rasterio.open(raster_path) as src:
                # Mask raster to buffer geometry
                out_image, out_transform = mask(src, [buffer_geom], crop=True)
                
                # Flatten the array and convert to 1D
                values = out_image.flatten()
                
                # Remove NoData values (typically NaN or a specific nodata value)
                nodata = src.nodata
                if nodata is not None:
                    values = values[values != nodata]
                
                # Remove NaN values
                values = values[~np.isnan(values)]
                
                # Remove zero values (as per requirements)
                values = values[values > 0]
                
                all_values.extend(values.tolist())
        except Exception as e:
            # If masking fails (e.g., no overlap), skip this raster
            print(f"  Warning: Could not extract from {raster_path.name}: {e}")
            continue
    
    return np.array(all_values) if all_values else np.array([])


def compute_statistics(values):
    """
    Compute built-up statistics from extracted values.
    
    Parameters
    ----------
    values : numpy.ndarray
        Array of built-up values
        
    Returns
    -------
    stats : dict
        Dictionary with mean, p90, and fraction
    """
    if len(values) == 0:
        return {
            'builtup_mean_3km': 0.0,
            'builtup_p90_3km': 0.0,
            'builtup_fraction_3km': 0.0
        }
    
    mean_val = np.mean(values)
    p90_val = np.percentile(values, 90)
    fraction = mean_val / PIXEL_AREA_M2
    
    return {
        'builtup_mean_3km': mean_val,
        'builtup_p90_3km': p90_val,
        'builtup_fraction_3km': fraction
    }


def classify_markets(gdf, urban_threshold, peri_urban_threshold):
    """
    Classify markets into Urban / Peri-urban / Rural based on thresholds.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with builtup_mean_3km column
    urban_threshold : float
        Threshold for Urban classification (80th percentile)
    peri_urban_threshold : float
        Threshold for Peri-urban classification (40th percentile)
        
    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame with urban_class column added
    """
    def assign_class(value):
        if value >= urban_threshold:
            return 'Urban'
        elif value >= peri_urban_threshold:
            return 'Peri-urban'
        else:
            return 'Rural'
    
    gdf['urban_class'] = gdf['builtup_mean_3km'].apply(assign_class)
    return gdf


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main processing function."""
    
    # Step 1: Load markets and create point geometries
    markets_gdf = load_markets(MARKETS_CSV)
    
    # Limit to test sample if MAX_MARKETS is set
    output_csv = OUTPUT_CSV
    if MAX_MARKETS is not None:
        print(f"\n⚠️  TEST MODE: Processing only first {MAX_MARKETS} markets")
        markets_gdf = markets_gdf.head(MAX_MARKETS).copy()
        # Update output filename for test mode
        output_csv = SCRIPT_DIR / f"markets_with_urban_class_3km_test_{MAX_MARKETS}.csv"
    
    # Step 2: Get raster CRS from first raster file
    raster_crs = get_raster_crs(RASTER_DIR)
    
    # Step 3: Reproject markets to raster CRS
    print(f"\nReprojecting markets to raster CRS...")
    
    # Check for invalid geometries
    invalid = ~markets_gdf.geometry.is_valid
    if invalid.sum() > 0:
        print(f"  Found {invalid.sum()} invalid geometries, fixing...")
        markets_gdf.loc[invalid, 'geometry'] = markets_gdf.loc[invalid].geometry.buffer(0)
    
    # Try reprojection with different methods
    try:
        # Method 1: Use CRS object directly
        markets_proj = markets_gdf.to_crs(raster_crs)
        print(f"  Successfully reprojected using CRS object")
    except Exception as e:
        print(f"  Method 1 failed: {e}")
        try:
            # Method 2: Use pyproj Transformer directly
            from pyproj import Transformer
            transformer = Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)
            
            # Transform coordinates manually
            coords = [(geom.x, geom.y) for geom in markets_gdf.geometry]
            transformed_coords = transformer.transform([c[0] for c in coords], [c[1] for c in coords])
            
            # Create new geometries
            from shapely.geometry import Point
            new_geometry = [Point(xy) for xy in zip(transformed_coords[0], transformed_coords[1])]
            markets_proj = markets_gdf.copy()
            markets_proj.geometry = new_geometry
            markets_proj.crs = raster_crs
            print(f"  Successfully reprojected using pyproj Transformer")
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
            raise RuntimeError(f"Could not reproject markets. Error: {e2}")
    
    # Step 4: Process each market
    print(f"\nProcessing {len(markets_proj)} markets with {BUFFER_SIZE_M}m buffer...")
    
    results = []
    total_markets = len(markets_proj)
    progress_interval = max(1, total_markets // 10)  # Show progress every 10%
    
    for idx, row in markets_proj.iterrows():
        market_num = len(results) + 1
        if market_num % progress_interval == 0 or market_num == total_markets:
            print(f"  Processed {market_num}/{total_markets} markets...")
        
        # Create buffer around market point
        buffer_geom = row.geometry.buffer(BUFFER_SIZE_M)
        
        # Find intersecting raster files
        raster_paths = get_intersecting_rasters(buffer_geom, RASTER_DIR, raster_crs)
        
        if not raster_paths:
            # No intersecting rasters found
            stats = {
                'builtup_mean_3km': 0.0,
                'builtup_p90_3km': 0.0,
                'builtup_fraction_3km': 0.0
            }
        else:
            # Extract built-up values from all intersecting rasters
            values = extract_builtup_values(buffer_geom, raster_paths, raster_crs)
            
            # Compute statistics
            stats = compute_statistics(values)
        
        # Store results (using original index to match with original dataframe)
        results.append({
            'index': idx,
            **stats
        })
    
    # Step 5: Add statistics to GeoDataFrame
    results_df = pd.DataFrame(results).set_index('index')
    markets_proj = markets_proj.join(results_df)
    
    # Step 6: Compute national quantile thresholds
    print(f"\nComputing national quantile thresholds...")
    urban_threshold = markets_proj['builtup_mean_3km'].quantile(URBAN_QUANTILE)
    peri_urban_threshold = markets_proj['builtup_mean_3km'].quantile(PERI_URBAN_QUANTILE)
    
    print(f"  40th percentile (Peri-urban threshold): {peri_urban_threshold:.2f}")
    print(f"  80th percentile (Urban threshold): {urban_threshold:.2f}")
    
    # Step 7: Classify markets
    print(f"\nClassifying markets...")
    markets_proj = classify_markets(markets_proj, urban_threshold, peri_urban_threshold)
    
    # Print classification summary
    class_counts = markets_proj['urban_class'].value_counts()
    print(f"\nClassification summary:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} markets ({count/len(markets_proj)*100:.1f}%)")
    
    # Step 8: Convert back to original CRS and prepare output
    print(f"\nConverting back to EPSG:4326...")
    try:
        markets_output = markets_proj.to_crs('EPSG:4326')
    except Exception as e:
        print(f"  Warning: Direct conversion failed: {e}")
        print(f"  Using pyproj Transformer...")
        from pyproj import Transformer
        transformer = Transformer.from_crs(raster_crs, 'EPSG:4326', always_xy=True)
        
        # Transform coordinates manually
        coords = [(geom.x, geom.y) for geom in markets_proj.geometry]
        transformed_coords = transformer.transform([c[0] for c in coords], [c[1] for c in coords])
        
        # Create new geometries
        from shapely.geometry import Point
        new_geometry = [Point(xy) for xy in zip(transformed_coords[0], transformed_coords[1])]
        markets_output = markets_proj.copy()
        markets_output.geometry = new_geometry
        markets_output.crs = 'EPSG:4326'
    
    # Drop geometry column for CSV output (keep x, y columns)
    output_df = markets_output.drop(columns=['geometry'])
    
    # Step 9: Save output CSV
    print(f"\nSaving results to {output_csv}...")
    output_df.to_csv(output_csv, index=False)
    print(f"Done! Output saved to {output_csv}")
    
    return output_df


if __name__ == "__main__":
    result_df = main()

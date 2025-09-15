# Service Points Location and Capacity Problem

Implementation of the mixed-integer linear programming (MILP) model from:
> Raviv, T. (2023). "The service points' location and capacity problem".  
> Transportation Research Part E, 176, 103216.

## Overview

This project implements an optimization model for locating and sizing automatic parcel lockers (APLs) in urban delivery networks, considering stochastic demand and pickup processes.

## Installation

### Git LFS Required

This repository uses Git LFS for large OSM data files. Install it before cloning:

**Windows:** `winget install GitHub.GitLFS`  
**macOS:** `brew install git-lfs`  
**Linux:** `sudo apt-get install git-lfs`

### Clone and Run

```bash
# Initialize Git LFS 
git lfs install

# Clone the repository (automatically downloads OSM files)
git clone https://github.com/francescacraievich/service-points-location-capacity.git
cd service-points-location-capacity

# Install dependencies
pip install -r requirements.txt

# Run 
python test.py          
python scalability.py   
```

## Key Files Description

### test.py
Main entry point for testing the Service Point Location and Capacity model. Includes:
- Basic model validation
- Comparison between PWL, deterministic, and scenario-based models
- Small instance tests (4×4 grid)

### scalability.py
Comprehensive scalability analysis using real data from Austrian cities:
- Tests with Vienna, Graz, and Linz data from OpenStreetMap
- Computational complexity analysis
- Performance benchmarking for different instance sizes
- Generates detailed reports and visualizations

### src/models/
Core optimization models implementing Raviv (2023):
- `sp_model.py`: Main stochastic model with piecewise linear approximation
- `deterministic_model.py`: Benchmark deterministic model
- `scenarios_model.py`: Two-stage stochastic model with demand scenarios
- `rejection_function.py`: M/M/k queueing-based rejection calculations

### src/utils/
Supporting utilities:
- `data_generator.py`: Create synthetic and realistic test instances
- `osm_austria.py`: Extract and process real city data from OpenStreetMap
- `visualization.py`: Generate network plots and analysis charts
- `extract_cities.py`: Extract individual city files from full Austria OSM dataset

### data/output/
Results organized by test type:
- `test/`: Basic model validation results
- `scalability/`: Performance analysis with timestamped runs
- `osm/`: OpenStreetMap data files for Austrian cities

## Data Files

The repository includes pre-extracted OSM data for three Austrian cities:
- `data/output/osm/vienna.osm.pbf` (26MB)
- `data/output/osm/graz.osm.pbf` (7MB)  
- `data/output/osm/linz.osm.pbf` (5MB)

These files are sufficient to run all tests and examples.

### Full Austria Dataset (Optional)
For complete scalability tests with all Austrian cities:
```bash
wget https://download.geofabrik.de/europe/austria-latest.osm.pbf -O data/output/osm/austria-latest.osm.pbf

```

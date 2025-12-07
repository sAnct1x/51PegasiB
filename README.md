# 51 Pegasi b - Radial Velocity Analysis

A Python project for analyzing radial velocity data to detect exoplanets, specifically focusing on **51 Pegasi b** - the first exoplanet discovered around a Sun-like star in 1995.

## Overview

This project implements a radial velocity fitting pipeline that:
- Performs least-squares fitting on radial velocity (RV) data
- Detects exoplanets using the Doppler spectroscopy method
- Models circular orbits with a sine wave model
- Visualizes phase-folded RV curves and residuals
- Calculates planetary mass and orbital parameters

## Features

### Version 1: `51PegasiB1995.py`
- **Custom GUI Loading Screen**: Interactive progress bar and status updates
- **Automatic Data Synthesis**: Generates synthetic RV data calibrated to 1995 ELODIE precision (~13 m/s)
- **Phase-Folding Visualization**: Displays data over 2.0 orbital cycles (visual 'M' shape)
- **Enhanced Plotting**: Dark mode with light grey graphs for better visibility
- **Comprehensive Reporting**: Detailed parameter output with uncertainties and goodness-of-fit statistics
- **Mathematical Verification**: Error bar consistency checks and residual analysis

### Version 2: `51PegasiB1995-test.py`
- Enhanced version with additional features and refinements

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd "Project 3"
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Version (`51PegasiB1995.py`)
```bash
python 51PegasiB1995.py
```

This script:
- Shows a GUI loading screen with progress updates
- Generates synthetic data if `51peg_rv_data.csv` doesn't exist
- Performs least-squares optimization using scipy
- Displays comprehensive analysis plots with phase-folding
- Outputs detailed report to terminal

### Test Version (`51PegasiB1995-test.py`)
```bash
python 51PegasiB1995-test.py
```

## Project Structure

```
Project 3/
├── 51PegasiB1995.py          # Main version with GUI and enhanced features
├── 51PegasiB1995-test.py     # Test/alternative version
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # License information
├── .gitignore                # Git ignore rules
└── Changelog/
    └── CHANGELOG.md          # Development history and fixes
```

## Key Parameters

The radial velocity model fits the following parameters:

- **P** (Period): Orbital period in days (~4.23 days for 51 Peg b)
- **K** (Semi-amplitude): Maximum RV variation in m/s (~56 m/s)
- **T0** (Time offset): Reference time when phase = 0
- **γ** (Gamma): Systemic velocity of the star in m/s

## Physical Calculations

The project calculates:

1. **Planetary Mass** (minimum mass):
   ```
   M_p sin(i) ≈ (P / 2πG)^(1/3) × K × M_star^(2/3)
   ```

Where:
- `P` is the orbital period in seconds
- `K` is the RV semi-amplitude in m/s
- `M_star` is the stellar mass in kg
- `G` is the gravitational constant

## Output

The analysis produces:
- **Phase-folded RV curve**: Shows the periodic signal over 2.0 orbital cycles
- **Residual plot**: Displays fit quality (O-C residuals)
- **Parameter report**: Best-fit values with uncertainties
- **Goodness-of-fit**: Chi-squared and reduced chi-squared statistics
- **Physical properties**: Derived planetary mass

## Scientific Background

**51 Pegasi b** was discovered in 1995 by Michel Mayor and Didier Queloz using the ELODIE spectrograph. This discovery:
- Was the first confirmed exoplanet around a main-sequence star
- Revolutionized astronomy and led to the 2019 Nobel Prize in Physics
- Revealed a "Hot Jupiter" - a gas giant orbiting extremely close to its star
- Demonstrated the radial velocity method for exoplanet detection

## Dependencies

See `requirements.txt` for the complete list. Main dependencies:
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Optimization and statistical functions
- `pandas` - Data handling
- `tkinter` - GUI components (usually included with Python)

## Notes

- The script generates synthetic data if no CSV file is found
- Assumes circular orbits (sine wave model)
- Error bars are calibrated to match 1995 ELODIE spectrograph precision (~10-15 m/s)
- Phase-folding is critical for visualizing periodic RV signals
- The visualization shows 2.0 cycles to reveal the repeating 'M' pattern

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Mayor, M., & Queloz, D. (1995). "A Jupiter-mass companion to a solar-type star." *Nature*, 378(6555), 355-359.
- ELODIE spectrograph precision: ~13 m/s for bright stars

## Author

OSU Astro 1221 - Autumn 2025


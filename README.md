# Cellular Automata (CA) Project - Traffic Noise Simulation

This repository contains an optimized and refactored Cellular Automata simulation framework designed to model traffic flow and analyze its acoustics and noise pollution environmental impact.

## 🚀 Features & Improvements
* **Intersection Modeling:** Integrated intersection mechanics into the traffic simulation.
* **Acoustic Analysis:** Includes environmental noise calculation scripts (`cnossos.py`, `noise_database.py`, `noise_map_linear.py`).
* **Visualizations:** Automated generation of traffic and acoustic animations.
* **Smart Data Management:** Heavy local simulation output folders (`simulation_*`) are automatically ignored via `.gitignore` to keep the repository clean.

## 📦 Project Structure
```text
├── Main_File.ipynb       # Main Jupyter Notebook to run and analyze simulations
├── simulation.py         # Core Cellular Automata simulation logic
├── parameters.py         # Simulation parameters and configuration constants
├── animate_road_only.py  # Script for animating standard road traffic
├── animate_acoustics.py  # Script for animating noise levels and acoustics
├── cnossos.py            # Cnossos-EU standard noise emission calculations
├── noise_database.py     # Database handler for noise levels
├── noise_map_linear.py   # Linear noise mapping framework
├── update.py             # Traffic state update rules
├── requirements.txt      # Python dependencies required to run the project
└── .gitignore            # Git exclusion rules

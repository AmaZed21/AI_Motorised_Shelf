# Motorized Shelf System

This project is a modular Python simulation of a voice-activated motorized shelf system that uses offline speech recognition to detect item requests, locate the correct compartment based on its contents, and control shelf movement in real time. It includes compartment-level state management, position tracking, motor current monitoring, obstruction and overload safety logic, and CSV-based logging for system events and status data.


## Features

- Voice command detection using offline speech recognition (Vosk)
- Dynamic item lookup across compartments — no hardcoded commands
- Real-time compartment movement simulation (position, speed, direction)
- Motor current monitoring
- Obstruction and overload safety flags
- CSV logging of system state over time
- Streamlit dashboard via `stream_app.py`


## Project Structure

motorized-shelf/
├── simulator.py # All classes: Compartment, Shelf, Voice, Logger
├── stream_app.py # Streamlit dashboard
├── data/
│ └── logs.csv # Runtime CSV logs
├── models/ # Place downloaded Vosk model here (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md


## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies:
- `streamlit`
- `numpy`
- `pandas`


## Vosk Model Setup

1. Download a model from [https://alphacephei.com/vosk/models]
   - Recommended: `vosk-model-small-en-us-0.15` (40 MB, good for testing)
2. Unzip the model folder into the `models/` directory:
    models/vosk-model-small-en-us-0.15/

3. Update the model path in `simulator.py` if needed:
```python
model_path = "models/vosk-model-small-en-us-0.15"
```

## Usage

Run the Streamlit dashboard:

```bash
streamlit run stream_app.py
```


## Voice Commands

Commands follow the pattern:
bring <item>

For example:
- `"bring inhaler"` — moves the compartment containing the inhaler
- `"bring towel"` — moves the compartment containing the towel
- `"stop"` — stops all compartment movement

The system dynamically searches compartment contents — no commands are hardcoded.


## Notes

- The Vosk model folder is git-ignored due to its size. Download it manually as described above.
- `logs.csv` is saved in the `data/` folder during runtime.

# smart-mirror-ai

This folder contains the AI-related services and scripts for the Smart Mirror project.

## Structure

- `ai_service_socketio.py`: Python service for AI communication via Socket.IO.
- `exercise_counters.py`: Script for counting exercises using AI models.
- `requirements.txt`: Python dependencies for the AI services.
- `ai-venv/`: Python virtual environment for isolating dependencies.
- `__pycache__/`: Compiled Python files.

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv ai-venv
   source ai-venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Run the AI service:
  ```bash
  python ai_service_socketio.py
  ```
- Run the exercise counter:
  ```bash
  python exercise_counters.py
  ```

## Models

- Place your ONNX models in the `../models/` directory.

## Environment Variables

- See `.env.example` for required environment variables.

## Notes

- Make sure the virtual environment is activated before running any scripts.
- For development, update `requirements.txt` as needed.

## License

This project is licensed under the MIT License.

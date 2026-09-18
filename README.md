# Helmet & Plate Detection

A Flask web application for detecting helmets and vehicle number plates from uploaded images and videos. It uses Ultralytics/YOLO for detection and OCR to read plate text.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Place the required model weights in `model/` (this folder is intentionally not committed).
4. Start the application:

   ```powershell
   python app.py
   ```

Then open the local address shown by Flask in your browser.

## Git notes

Model weights, uploaded media, generated outputs/reports, databases, virtual environments, and local editor settings are ignored by Git. This keeps the repository focused on the source code and avoids uploading large or private local files.

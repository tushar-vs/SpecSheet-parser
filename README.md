📊 SpecSheet Parser: Optical Table Extraction Engine
SpecSheet Parser is a Computer Vision module designed to reconstruct tabular data from scanned engineering documents, technical specifications, and construction blueprints.
Unlike standard OCR which simply reads text as a stream, this project utilizes Morphological Layout Analysis to understand the structure of a document. It isolates grid lines, segments individual table cells, and extracts data into a structured CSV format, ensuring that row/column relationships are preserved even in non-selectable PDFs or images.
🚀 Key Features
Structure Recognition: Uses OpenCV morphological kernels to mathematically detect horizontal and vertical lines, separating the "Table" from "General Notes".
Cell Segmentation: Isolates individual data fields before passing them to OCR, preventing text merging errors common in dense schedules.
Noise Handling: Implements adaptive thresholding to handle shadows and uneven lighting in scanned documents (300 DPI).
Layout Preservation: Custom sorting algorithms ensure extracted text is mapped to the correct row and column indices.
🛠️ Tech Stack
OpenCV (cv2): Image preprocessing, kernel filtering, and contour detection.
Tesseract OCR: Text recognition engine (configured with --psm 6 for cell-level extraction).
Pandas: Data structuring and CSV export.
NumPy: Matrix operations for image array handling.
⚙️ How It Works
Input: Accepts scanned images (JPG/PNG) of technical schedules.
Binarization: Converts image to binary (black/white) using Adaptive Gaussian Thresholding to handle noise.
Morphology:
Applies a (40, 1) kernel to isolate horizontal lines.
Applies a (1, 40) kernel to isolate vertical lines.
Grid Reconstruction: Combines lines to create a mask of the table structure.
Extraction:
Finds contours (cells) in the mask.
Crops each cell.
Runs Tesseract on the cropped region.
Output: Generates a clean CSV file and a debug image showing detected cells.
📦 Installation & Usage
Install Dependencies:
pip install -r requirements.txt

Run the Parser:
python main.py


Check Output:
Results are saved in the output/ folder:
extracted_schedule.csv: The raw text data.
detected_cells.jpg: A visualization of what the computer saw.


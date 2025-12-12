import cv2
import numpy as np
import pandas as pd
import pytesseract
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_IMAGE_PATH = "input_document.jpg" 


OUTPUT_DIR = "output"



# ==========================================
# CORE LOGIC
# ==========================================

def process_spec_sheet(image_path):
    """
    detects table structure using morphological operations,
    segments cells, and extracts text using OCR.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] The file '{image_path}' was not found.")
        return []

    print(f"[INFO] Loading image: {image_path}")
    
    # 1. Preprocessing
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding: Best for scanned blueprints with shadows/uneven light
    # Block Size: 11, C: 2
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Layout Analysis (Morphological Operations)
    # Define scale based on image width to detect lines relative to size
    scale = 20
    
    # Detect Horizontal Lines
    # Kernel shape: (Width, 1) -> E.g., (40, 1)
    hori_kernel_len = np.array(img).shape[1] // scale
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_kernel_len, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hori_kernel, iterations=2)
    
    # Detect Vertical Lines
    # Kernel shape: (1, Height) -> E.g., (1, 40)
    vert_kernel_len = np.array(img).shape[0] // scale
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    
    # Combine to find the Grid Structure
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # 3. Cell Segmentation
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for all contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Filter noise: Keep boxes that are large enough to be text cells
    # Adjust these values based on your image resolution (300 DPI vs 72 DPI)
    cells = [b for b in bounding_boxes if b[2] > 20 and b[3] > 10]
    
    # Sort Cells:
    # 1. Sort by Y position (Rows) with a tolerance of 10 pixels (// 10)
    # 2. Then sort by X position (Columns)
    cells.sort(key=lambda b: (b[1] // 10, b[0]))
    
    print(f"[INFO] Detected {len(cells)} cells. Starting OCR extraction...")
    
    extracted_data = []
    vis_img = img.copy() # Copy for visualization
    
    # 4. OCR Extraction Loop
    for (x, y, w, h) in cells:
        # Crop the cell
        roi = gray[y:y+h, x:x+w]
        
        # Add white padding (border) so Tesseract doesn't miss edge characters
        roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255])
        
        # Run Tesseract
        # --psm 6: Assume a single uniform block of text
        text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        
        # Basic cleaning
        text = text.replace('\n', ' ').replace('|', '')
        
        if text:
            extracted_data.append({
                "x_pos": x,
                "y_pos": y,
                "width": w,
                "height": h,
                "content": text
            })
            
            # Draw green box on visualization image
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save visualization debug image
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_path = os.path.join(OUTPUT_DIR, "debug_detected_cells.jpg")
    cv2.imwrite(debug_path, vis_img)
    print(f"[INFO] Debug image saved to: {debug_path}")
    
    return extracted_data

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Ensure input file exists
    if not os.path.exists(INPUT_IMAGE_PATH):
        print("="*50)
        print(f"ERROR: Input file not found: '{INPUT_IMAGE_PATH}'")
        print("Please place an image file in this folder and update 'INPUT_IMAGE_PATH' in the code.")
        print("="*50)
        sys.exit(1)

    # Run the pipeline
    results = process_spec_sheet(INPUT_IMAGE_PATH)
    
    if results:
        # Save to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "extracted_table.csv")
        df.to_csv(csv_path, index=False)
        
        print("="*50)
        print("SUCCESS! Extraction Complete.")
        print(f"Data saved to: {csv_path}")
        print("="*50)
        print(df.head())
    else:
        print("[WARN] No text detected. Check if the image has a clear table structure.")

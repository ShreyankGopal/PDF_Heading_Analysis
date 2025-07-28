# PDF Heading Analysis with Random Forest Classifier

This project processes PDF documents to identify structural headings (e.g., Title, H1, H2, etc.) using a trained Random Forest classifier. It extracts and classifies text segments from PDFs based on formatting features, and produces structured JSON outlines for further downstream applications.

## Features

- PDF text extraction using PyMuPDF
- Heuristic filtering for noise and bullet points
- Feature engineering: font size, capitalization, bold/italic, layout gaps
- Machine Learning classifier to label headings (Random Forest)
- Outputs structured JSON with document outline
- Works fully offline using local files and models

## Directory Structure

```
project/
│
├── input/  # Place your PDFs here
├── output/ # JSON outputs will be saved here
├── heading_classifier_with_font_count_norm_textNorm_5.pkl  # Trained model
├── process_pdfs_3.py # Main pipeline script
├── requirements.txt
├── Dockerfile  
```

## How to Run

### Docker (Recommended)

1. Build the Docker image
   ```bash
   docker build --platform linux/amd64 -t pdf-heading-analyzer .
   ```
2. Run the container
   ```bash
   docker run --rm \
     -v $(pwd)/input:/app/input:ro \
     -v $(pwd)/output:/app/output \
     --network none \
     pdf-heading-analyzer
   ```

Ensure that the model file `heading_classifier_with_font_count_norm_textNorm_5.pkl` is in the working directory before running.

## Approach Summary

1. **Text Extraction & Preprocessing**
   - Uses PyMuPDF to extract spans from each page
   - Groups lines based on font size, bold/italic flags, and vertical spacing (Y Gap)
   - Cleans bullet points and filters out non-informative lines (e.g., page numbers)

2. **Feature Engineering**
   For each text group:
   - Font Ratio (relative to mode font size)
   - Font Size Rank
   - Text Length
   - Capitalization Ratio
   - Position Y (vertical location on page)
   - Is Bold, Is Italic
   - Starts with Numbering
   - Font Size Count
   - Is Unique Font Size
   - Y Gap Scaled (scaled gap between blocks)

3. **Heading Classification**
   A Random Forest model (trained separately) predicts one of: 'Title', 'H1', 'H2', 'H3', 'None'. Uses only numerical and categorical formatting features — no semantic analysis.

4. **Structured Output**
   First detected 'Title' is treated as the document's main title. Other headings (H1, H2, H3) are collected with associated page numbers. Output format:
   ```json
   {
     "title": "Document Title",
     "outline": [
       { "level": "H1", "text": "Introduction", "page": 1 },
       { "level": "H2", "text": "Project Goals", "page": 2 },
       ...
     ]
   }
   ```

## Notes

You can modify the `is_heading_heuristic()` function to further improve candidate heading filtering. PDFs with unusual layouts (e.g., scanned images) may not work reliably. The Random Forest model expects consistent feature preprocessing — don't change input feature names unless retraining the model.

## Dependencies

If running locally (not Docker), install:
```bash
pip install -r requirements.txt
```
Typical dependencies:
- PyMuPDF
- scikit-learn
- pandas
- joblib
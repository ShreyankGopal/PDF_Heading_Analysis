FROM python:3.10

WORKDIR /app

# Copy the processing script
COPY heading_classifier_with_font_count_norm_textNorm_5.pkl .
COPY process_pdfs_3.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

# Run the script
CMD ["python", "process_pdfs_3.py"] 
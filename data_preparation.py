import os
from pathlib import Path
import PyPDF2
from docx import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_pdf(file_path):
    """Extract text from PDF files"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {str(e)}")
        return ''

def read_docx(file_path):
    """Extract text from DOCX files"""
    try:
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
        return ''

def process_documents(input_folder, output_file):
    """Process all PDF and DOCX files in the input folder and combine into one text file"""
    combined_text = []
    input_path = Path(input_folder)
    
    # Supported file extensions
    supported_extensions = {'.pdf', '.docx'}
    
    # Process all files in the input folder
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in supported_extensions:
            logger.info(f"Processing file: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                text = read_pdf(file_path)
            else:  # .docx
                text = read_docx(file_path)
                
            if text:
                # Add a separator between documents
                combined_text.append(f"### Document: {file_path.name} ###")
                combined_text.append(text)
                combined_text.append("\n" + "="*50 + "\n")  # Separator between documents
    
    # Write the combined text to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_text))
        logger.info(f"Successfully created combined text file: {output_file}")
    except Exception as e:
        logger.error(f"Error writing to output file: {str(e)}")

if __name__ == "__main__":
    # Configure these paths according to your setup
    INPUT_FOLDER = "./documents"  # Folder containing PDFs and DOCXs
    OUTPUT_FILE = "./data_lm.txt"  # Output file path (matches your training script)
    
    # Create input folder if it doesn't exist
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    # Process the documents
    process_documents(INPUT_FOLDER, OUTPUT_FILE)
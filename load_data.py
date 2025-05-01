import pdfplumber

def load_sample_data(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from all pages.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf_reader:
            data = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            return data
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""
    
from datasets import load_dataset
from collections import defaultdict

def get_data():
    dataset = load_dataset("go_emotions")
    train_data = dataset['train']

    # Get the first 50 rows
    top_50_rows = train_data.select(range(min(10, len(train_data))))

    # Format each row into a string
    formatted_rows = []
    for row in top_50_rows:
        text = row['text']
        labels = [train_data.features['labels'].feature.names[label] for label in row['labels']]
        formatted_row = f"Text: {text}\nLabels: {', '.join(labels)}\n"
        formatted_rows.append(formatted_row)

    # Join all formatted rows into a single string
    formatted_output = "\n".join(formatted_rows)

    return formatted_output
    
if __name__ == "__main__":
    get_data()

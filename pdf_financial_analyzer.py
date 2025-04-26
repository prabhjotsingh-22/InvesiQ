import os
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\iamva\Downloads\gen-lang-client-0195737084-e66a7948cb91.json"
# Download required NLTK data
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def segment_text(text, max_chunk_size=4000):
    """Segment text into smaller chunks using sentence tokenization."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_financial_chunk(chunk):
    """Analyze a chunk of financial text using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Analyze the following chunk of financial report text and provide insights for investors.
    Focus on:
    1. Key financial metrics and their trends
    2. Business performance indicators
    3. Risk factors and opportunities
    4. Notable changes in company strategy or operations
    5. Market position and competitive advantages
    
    Text to analyze:
    {chunk}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing chunk: {str(e)}"

def generate_summary_report(analyses):
    """Generate a final summary report from all analyses."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    combined_analyses = "\n\n".join(analyses)
    
    prompt = """
    Based on the following analyses of different sections of the financial report,
    provide a comprehensive investment analysis summary. Include:
    
    1. Executive Summary
    2. Financial Health Assessment
    3. Growth Prospects
    4. Risk Analysis
    5. Investment Recommendation
    
    Previous analyses:
    """ + combined_analyses
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def analyze_financial_report(pdf_path):
    """Main function to analyze a financial report PDF."""
    print("Starting PDF analysis...")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print("PDF text extracted successfully.")
    
    # Segment text into chunks
    chunks = segment_text(text)
    print(f"Text segmented into {len(chunks)} chunks.")
    
    # Analyze each chunk
    analyses = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Analyzing chunk {i}/{len(chunks)}...")
        analysis = analyze_financial_chunk(chunk)
        analyses.append(analysis)
    
    # Generate final summary
    print("Generating final summary...")
    final_summary = generate_summary_report(analyses)
    
    return final_summary

if __name__ == "__main__":
    # Example usage
    pdf_path = input("Enter the path to the financial report PDF: ")
    
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
    else:
        summary = analyze_financial_report(pdf_path)
        
        # Save the analysis to a file
        output_file = "financial_analysis_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        print("\nSummary of analysis:")
        print("=" * 80)
        print(summary) 
import os
import PyPDF2
import numpy as np
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from groq import Groq

# Download required NLTK data
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define financial keywords and topics for relevance scoring
FINANCIAL_KEYWORDS = [
    "revenue", "profit", "margin", "growth", "earnings", "EBITDA", "cash flow",
    "balance sheet", "assets", "liabilities", "debt", "equity", "ROI", "ROE",
    "market share", "competitive advantage", "risk", "strategy", "forecast",
    "dividend", "capital", "investment", "operational efficiency", "cost reduction"
]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def segment_text(text, max_chunk_size=1000):
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

def create_query_vectors():
    """Create query vectors for different aspects of financial analysis."""
    queries = [
        "financial performance metrics including revenue, profit, and growth",
        "balance sheet analysis including assets, liabilities, and equity",
        "cash flow analysis and liquidity position",
        "risk factors and market challenges",
        "competitive advantages and market position",
        "business strategy and future outlook",
        "operational efficiency and cost management"
    ]
    return model.encode(queries)

def vectorize_chunks(chunks):
    """Vectorize text chunks using sentence transformer."""
    print("Vectorizing text chunks...")
    return model.encode(chunks)

def rank_chunks(chunk_vectors, query_vectors, chunks):
    """Rank chunks based on their relevance to financial analysis queries."""
    # Calculate similarity scores for each chunk against all queries
    similarities = cosine_similarity(chunk_vectors, query_vectors)
    
    # Get the maximum similarity score for each chunk
    max_similarities = np.max(similarities, axis=1)
    
    # Create a list of (chunk, score) tuples
    ranked_chunks = list(zip(chunks, max_similarities))
    
    # Sort by similarity score in descending order
    ranked_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_chunks

def filter_top_chunks(ranked_chunks, top_n=10, threshold=0.3):
    """Filter top N most relevant chunks that meet the threshold."""
    filtered_chunks = [chunk for chunk, score in ranked_chunks if score > threshold]
    return filtered_chunks[:top_n]

def analyze_chunk_with_groq(chunk):
    """Analyze a chunk of financial text using Groq."""
    prompt = f"""
    Analyze the following financial report excerpt and provide key insights:
    1. Key financial metrics and their implications
    2. Risk factors and opportunities
    3. Business performance indicators
    4. Strategic insights
    
    Text:
    {chunk}
    
    Provide a concise, bullet-point analysis focusing on the most important insights.
    """
    
    try:
        completion = groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="mixtral-8x7b-32768",  # Using Mixtral model for better financial analysis
            temperature=0.3,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error analyzing chunk: {str(e)}"

def generate_final_summary(analyses):
    """Generate a comprehensive summary from the analyzed chunks."""
    summary_prompt = """
    Based on the following analyses of key sections from the financial report,
    provide a comprehensive investment analysis summary:

    1. Executive Summary (2-3 bullet points)
    2. Financial Health Assessment
    3. Key Risks and Opportunities
    4. Investment Recommendation
    
    Previous analyses:
    """ + "\n\n".join(analyses)
    
    try:
        completion = groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": summary_prompt
            }],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=2000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def analyze_financial_report(pdf_path):
    """Main function to analyze a financial report PDF using smart chunking."""
    print("Starting PDF analysis...")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print("PDF text extracted successfully.")
    
    # Segment text into chunks
    chunks = segment_text(text)
    print(f"Text segmented into {len(chunks)} chunks.")
    
    # Create query vectors for financial aspects
    query_vectors = create_query_vectors()
    
    # Vectorize chunks
    chunk_vectors = vectorize_chunks(chunks)
    
    # Rank chunks by relevance
    print("Ranking chunks by relevance...")
    ranked_chunks = rank_chunks(chunk_vectors, query_vectors, chunks)
    
    # Filter top chunks
    top_chunks = filter_top_chunks(ranked_chunks)
    print(f"Selected {len(top_chunks)} most relevant chunks for analysis.")
    
    # Analyze selected chunks
    analyses = []
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Analyzing chunk {i}/{len(top_chunks)}...")
        analysis = analyze_chunk_with_groq(chunk)
        analyses.append(analysis)
    
    # Generate final summary
    print("Generating final summary...")
    final_summary = generate_final_summary(analyses)
    
    return final_summary

if __name__ == "__main__":
    pdf_path = input("Enter the path to the financial report PDF: ")
    
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
    else:
        summary = analyze_financial_report(pdf_path)
        
        # Save the analysis to a file
        output_file = "smart_financial_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        print("\nSummary of analysis:")
        print("=" * 80)
        print(summary) 
# InvesiQ - Investment intelligence, redefined

This repository contains two implementations of a financial report analyzer:
1. Basic Analyzer (using Google's Gemini)
2. Smart Analyzer (using Groq with vectorization)

Both tools are designed to analyze financial reports (annual reports, quarterly reports) and provide investment-focused insights. They help investors make informed decisions by extracting and analyzing key information from financial documents.

## Features

### Basic Analyzer (Gemini Version)
- PDF text extraction
- Simple text segmentation
- Direct analysis using Google's Gemini AI
- Investment-focused insights
- Comprehensive summary generation

### Smart Analyzer (Groq Version)
- Advanced PDF text extraction
- Smart vectorization for identifying relevant sections
- Semantic ranking of content chunks
- Efficient processing using Groq's Mixtral model
- Optimized for reducing API usage

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root directory and add your API keys:
```
# For Gemini version
GOOGLE_API_KEY=your_gemini_api_key_here

# For Groq version
GROQ_API_KEY=your_groq_api_key_here
```

3. Make sure you have Python 3.7 or higher installed.

## Usage

### Basic Analyzer (Gemini)
```bash
python pdf_financial_analyzer.py
```

### Smart Analyzer (Groq)
```bash
python smart_pdf_analyzer.py
```

For both versions:
1. When prompted, enter the path to your financial report PDF file.
2. The script will process the document and generate analysis.
3. Results will be saved to a text file.

## How They Work

### Basic Analyzer (Gemini)
- Extracts text from PDF
- Splits into simple chunks
- Analyzes each chunk with Gemini
- Generates comprehensive summary
- Saves to `financial_analysis_report.txt`

### Smart Analyzer (Groq)
- Extracts and processes PDF text
- Uses vectorization to identify relevant sections
- Ranks chunks by financial relevance
- Analyzes only the most important sections
- Saves to `smart_financial_analysis.txt`

## Output Format

Both analyzers provide:
- Executive Summary
- Financial Health Assessment
- Key Risks and Opportunities
- Investment Recommendation

## Technical Details

### Basic Analyzer
- Uses PyPDF2 for text extraction
- Simple chunking based on character count
- Direct API calls to Gemini

### Smart Analyzer
- Uses sentence-transformers for vectorization
- Implements cosine similarity for relevance ranking
- Employs Groq's Mixtral-8x7b model
- Optimized chunk selection

## Choosing Between Versions

Use the Basic Analyzer (Gemini) when:
- You need quick, straightforward analysis
- The PDF is well-structured and relatively small
- You have Gemini API credits available

Use the Smart Analyzer (Groq) when:
- You need more focused, relevant insights
- The PDF is large or poorly structured
- You want to minimize API usage
- You need more sophisticated financial analysis

## Notes

- Both versions work best with well-formatted PDF financial reports
- Analysis quality depends on the input PDF's clarity and structure
- Processing time varies based on document length and complexity
- Smart Analyzer may be more cost-effective for large documents
- Consider API quotas and costs when choosing between versions 

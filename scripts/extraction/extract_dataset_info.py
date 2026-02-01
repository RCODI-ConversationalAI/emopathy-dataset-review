#!/usr/bin/env python3
"""
Dataset Paper Information Extractor
Uses Claude API to extract structured information from dataset papers
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import anthropic
import base64
import fitz  # PyMuPDF for PDF reading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PDF_DIR = 'dataset_papers_pdfs'
INPUT_CSV = 'LIST_DATASET_PAPERS.csv'
OUTPUT_XLSX = 'DATASET_PAPERS_EXTRACTED.xlsx'
OUTPUT_CSV = 'DATASET_PAPERS_EXTRACTED.csv'
LOG_FILE = 'dataset_extraction_log.json'

# Initialize Claude client
api_key = os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
client = anthropic.Anthropic(api_key=api_key)

EXTRACTION_PROMPT = """You are extracting structured information from an academic paper about an emotion/empathy dataset.

Extract the following fields from this paper. If information is not available, write "N/A".

1. **Dataset name**: The name of the dataset introduced in this paper
2. **Language**: Language of the dataset (should be English for this review)
3. **Domain**: Application domain (e.g., Movies/TV, Conversation, Social media, Customer service, Therapy/Counseling, Interview, Gaming, General)
4. **Conversation setting**: Type of conversation (dyadic, multi-party, monologue, N/A)
5. **Corpus setting**: How data was collected (acted, naturalistic, elicited, mixed)
6. **Modality**: Data modalities (text, audio, video, multimodal - list all that apply)
7. **Source**: Where the data comes from (e.g., movies, TV shows, YouTube, Twitter, Reddit, lab recordings, call center, etc.)
8. **Labels**: Emotion labels used (e.g., Ekman 6 basic emotions, Plutchik 8, VAD dimensional, custom categories - be specific)
9. **Annotation**: How annotations were done (crowdsource, expert, self-report, automatic, mixed)
10. **Statistics**: Dataset size (number of samples, speakers, utterances, hours of audio/video, etc.)
11. **Link to dataset**: URL where dataset can be accessed (if mentioned in paper)

Return ONLY a JSON object with these exact keys:
{
    "dataset_name": "",
    "language": "",
    "domain": "",
    "conversation_setting": "",
    "corpus_setting": "",
    "modality": "",
    "source": "",
    "labels": "",
    "annotation": "",
    "statistics": "",
    "dataset_link": ""
}
"""

def read_pdf_text(pdf_path, max_pages=15):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text[:50000]  # Limit to ~50k chars
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_info_with_claude(pdf_text, title):
    """Use Claude API to extract structured information"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Paper title: {title}

Paper content:
{pdf_text}

{EXTRACTION_PROMPT}"""
                }
            ]
        )

        response_text = message.content[0].text

        # Parse JSON from response
        # Find JSON object in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            return json.loads(json_str)
        else:
            return None

    except Exception as e:
        print(f"API error: {e}")
        return None

def main():
    # Load paper list
    df = pd.read_csv(INPUT_CSV)
    print(f"Total papers in list: {len(df)}")

    # Get list of downloaded PDFs
    pdf_files = list(Path(PDF_DIR).glob('*.pdf'))
    print(f"Downloaded PDFs: {len(pdf_files)}")

    # Create mapping from index to PDF file
    pdf_map = {}
    for pdf_file in pdf_files:
        # Extract index from filename (format: 0001_title.pdf)
        idx = int(pdf_file.name.split('_')[0])
        pdf_map[idx] = pdf_file

    # Results storage
    results = []
    extraction_log = {
        'start_time': datetime.now().isoformat(),
        'total': len(pdf_files),
        'success': 0,
        'failed': 0,
        'papers': []
    }

    for idx, row in df.iterrows():
        title = row.get('title', f'Paper {idx}')
        authors = row.get('authors', '')
        year = row.get('year', '')
        url = row.get('url', '')

        # Check if PDF exists
        if idx not in pdf_map:
            # No PDF downloaded - add with empty extraction
            results.append({
                'title': title,
                'authors': authors,
                'year': year,
                'paper_link': url,
                'dataset_name': 'PDF_NOT_AVAILABLE',
                'language': '',
                'domain': '',
                'conversation_setting': '',
                'corpus_setting': '',
                'modality': '',
                'source': '',
                'labels': '',
                'annotation': '',
                'statistics': '',
                'dataset_link': ''
            })
            continue

        pdf_path = pdf_map[idx]
        print(f"[{idx+1}/{len(df)}] Extracting: {title[:50]}...")

        # Read PDF
        pdf_text = read_pdf_text(pdf_path)
        if not pdf_text:
            print(f"  ✗ Could not read PDF")
            extraction_log['failed'] += 1
            results.append({
                'title': title,
                'authors': authors,
                'year': year,
                'paper_link': url,
                'dataset_name': 'PDF_READ_ERROR',
                'language': '',
                'domain': '',
                'conversation_setting': '',
                'corpus_setting': '',
                'modality': '',
                'source': '',
                'labels': '',
                'annotation': '',
                'statistics': '',
                'dataset_link': ''
            })
            continue

        # Extract with Claude
        extracted = extract_info_with_claude(pdf_text, title)

        if extracted:
            print(f"  ✓ Extracted: {extracted.get('dataset_name', 'Unknown')}")
            extraction_log['success'] += 1
            extraction_log['papers'].append({
                'idx': idx,
                'title': title,
                'status': 'success',
                'dataset_name': extracted.get('dataset_name', '')
            })

            results.append({
                'title': title,
                'authors': authors,
                'year': year,
                'paper_link': url,
                **extracted
            })
        else:
            print(f"  ✗ Extraction failed")
            extraction_log['failed'] += 1
            extraction_log['papers'].append({
                'idx': idx,
                'title': title,
                'status': 'failed'
            })
            results.append({
                'title': title,
                'authors': authors,
                'year': year,
                'paper_link': url,
                'dataset_name': 'EXTRACTION_FAILED',
                'language': '',
                'domain': '',
                'conversation_setting': '',
                'corpus_setting': '',
                'modality': '',
                'source': '',
                'labels': '',
                'annotation': '',
                'statistics': '',
                'dataset_link': ''
            })

    # Save results
    results_df = pd.DataFrame(results)

    # Reorder columns
    column_order = [
        'dataset_name', 'authors', 'year', 'language', 'domain',
        'conversation_setting', 'corpus_setting', 'modality', 'source',
        'labels', 'annotation', 'statistics', 'dataset_link', 'paper_link', 'title'
    ]
    results_df = results_df[[c for c in column_order if c in results_df.columns]]

    # Save to Excel and CSV
    results_df.to_excel(OUTPUT_XLSX, index=False)
    results_df.to_csv(OUTPUT_CSV, index=False)

    # Save log
    extraction_log['end_time'] = datetime.now().isoformat()
    with open(LOG_FILE, 'w') as f:
        json.dump(extraction_log, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total papers: {len(df)}")
    print(f"PDFs available: {len(pdf_files)}")
    print(f"Extraction success: {extraction_log['success']}")
    print(f"Extraction failed: {extraction_log['failed']}")
    print(f"\nResults saved to: {OUTPUT_XLSX}")
    print(f"CSV backup: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Empathy Dataset Paper Information Extractor
Focuses on empathy definition and measurement
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import anthropic
import fitz
from dotenv import load_dotenv
import re
import time

load_dotenv()

# Configuration
PDF_DIRS = ['empathy_papers_pdfs', 'ScopusPapers']
INPUT_CSV = 'EMPATHY_DATASET_PAPERS.csv'
OUTPUT_XLSX = 'EMPATHY_DATASET_EXTRACTED.xlsx'
OUTPUT_CSV = 'EMPATHY_DATASET_EXTRACTED.csv'

# Initialize Claude client
api_key = os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
client = anthropic.Anthropic(api_key=api_key)

EXTRACTION_PROMPT = """You are extracting structured information from an academic paper that introduces or uses an empathy-related dataset.

IMPORTANT: Focus especially on how the paper DEFINES and MEASURES empathy. This is critical.

Extract the following fields. If information is not available, write "N/A".

## Empathy Definition (MOST IMPORTANT)
1. empathy_definition: How does the paper explicitly define empathy? Quote if possible.
2. empathy_components: What components of empathy are discussed? (cognitive, affective, compassionate, perspective-taking, emotional contagion)
3. empathy_theory: What theoretical framework is used? (Davis IRI, Batson's model, Preston & de Waal, other)
4. empathy_vs_emotion: How does the paper distinguish empathy from emotion?
5. operationalization: How is empathy operationalized/made measurable in the dataset?

## Measurement
6. measurement_level: At what level is empathy measured? (utterance, turn, conversation, person, essay)
7. measurement_type: Type of measurement (categorical, dimensional/continuous, binary)
8. empathy_labels: What empathy labels are used? (e.g., 0-5 scale, high/medium/low, specific categories)
9. emotion_labels: What emotion labels are included, if any?
10. label_definitions: How are the labels defined? Provide brief definitions.

## Data Characteristics
11. dataset_name: Name of the dataset
12. language: Language(s) of the data
13. domain: Application domain (mental health, social media, therapy, customer service, general)
14. source: Where data was collected from (Reddit, Twitter, crowdsourcing, lab recordings, movies)
15. modality: Data modalities (text, audio, video, multimodal)
16. conversation_type: Type of conversation (dyadic, multi-party, monologue)
17. corpus_setting: How data was collected (naturalistic, acted, elicited, crowdsourced)

## Statistics
18. num_samples: Total number of samples
19. num_conversations: Number of conversations/dialogues
20. num_utterances: Number of utterances
21. duration: Duration in hours (for audio/video) or other size metrics

## Annotation
22. annotation_method: Who annotated? (expert, crowdsource, self-report, automatic)
23. annotator_training: How were annotators trained?
24. annotation_guideline: Brief description of annotation guidelines
25. inter_annotator_agreement: IAA metrics (Kappa, Krippendorff's alpha, etc.)

## Availability
26. availability: Is dataset publicly available? (public, upon request, private)
27. dataset_link: URL to access the dataset
28. license: License type if mentioned

## Contribution
29. novelty: What is novel about this dataset?
30. limitations: What limitations are acknowledged?
31. intended_use: What is the dataset intended to be used for?

Return ONLY a JSON object with these exact keys:
{
    "empathy_definition": "",
    "empathy_components": "",
    "empathy_theory": "",
    "empathy_vs_emotion": "",
    "operationalization": "",
    "measurement_level": "",
    "measurement_type": "",
    "empathy_labels": "",
    "emotion_labels": "",
    "label_definitions": "",
    "dataset_name": "",
    "language": "",
    "domain": "",
    "source": "",
    "modality": "",
    "conversation_type": "",
    "corpus_setting": "",
    "num_samples": "",
    "num_conversations": "",
    "num_utterances": "",
    "duration": "",
    "annotation_method": "",
    "annotator_training": "",
    "annotation_guideline": "",
    "inter_annotator_agreement": "",
    "availability": "",
    "dataset_link": "",
    "license": "",
    "novelty": "",
    "limitations": "",
    "intended_use": ""
}
"""

def read_pdf_text(pdf_path, max_pages=25):
    """Extract text from PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text[:70000]
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_info_with_claude(pdf_text, title):
    """Use Claude API to extract structured information"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
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

def find_pdf_for_paper(row, pdf_files):
    """Find matching PDF for a paper"""
    title = str(row.get('title', '')).lower()
    title_clean = re.sub(r'[^a-zA-Z0-9]', '', title)[:30]
    year = str(row.get('year', ''))

    for pdf_path in pdf_files:
        pdf_name = pdf_path.name.lower()
        pdf_clean = re.sub(r'[^a-zA-Z0-9]', '', pdf_name)

        if title_clean[:20] in pdf_clean:
            return pdf_path

        if year in pdf_name:
            title_words = title.split()[:3]
            if any(w.lower() in pdf_name for w in title_words if len(w) > 4):
                return pdf_path

    return None

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Total Dataset papers: {len(df)}")

    pdf_files = []
    for pdf_dir in PDF_DIRS:
        if os.path.exists(pdf_dir):
            pdf_files.extend(list(Path(pdf_dir).glob('*.pdf')))
    print(f"Total PDFs available: {len(pdf_files)}")

    results = []

    for idx, row in df.iterrows():
        title = row.get('title', f'Paper {idx}')

        pdf_path = find_pdf_for_paper(row, pdf_files)

        if not pdf_path:
            print(f"[{idx+1}/{len(df)}] No PDF: {title[:50]}...")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'pdf_status': 'NOT_FOUND'
            }
            results.append(result)
            continue

        print(f"[{idx+1}/{len(df)}] Processing: {title[:50]}...")

        pdf_text = read_pdf_text(pdf_path)
        if not pdf_text:
            print(f"  ✗ Could not read PDF")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'pdf_status': 'READ_ERROR'
            }
            results.append(result)
            continue

        extracted = extract_info_with_claude(pdf_text, title)

        if extracted:
            print(f"  ✓ Extracted: {extracted.get('dataset_name', 'Unknown')}")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'pdf_status': 'SUCCESS',
                **extracted
            }
        else:
            print(f"  ✗ Extraction failed")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'pdf_status': 'EXTRACTION_FAILED'
            }

        results.append(result)
        time.sleep(0.5)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel(OUTPUT_XLSX, index=False)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    success = len([r for r in results if r.get('pdf_status') == 'SUCCESS'])
    print(f"Total papers: {len(df)}")
    print(f"Success: {success}")
    print(f"No PDF: {len([r for r in results if r.get('pdf_status') == 'NOT_FOUND'])}")
    print(f"Failed: {len([r for r in results if r.get('pdf_status') in ['READ_ERROR', 'EXTRACTION_FAILED']])}")
    print(f"\nResults saved to: {OUTPUT_XLSX}")

if __name__ == '__main__':
    main()

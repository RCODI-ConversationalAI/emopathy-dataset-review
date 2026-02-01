#!/usr/bin/env python3
"""
Empathy ML Paper Information Extractor
Uses Claude API to extract structured information from papers
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
import re
import time

load_dotenv()

# Configuration
PDF_DIRS = ['empathy_papers_pdfs', 'ScopusPapers']
INPUT_CSV = 'EMPATHY_ML_PAPERS.csv'
OUTPUT_XLSX = 'EMPATHY_ML_EXTRACTED.xlsx'
OUTPUT_CSV = 'EMPATHY_ML_EXTRACTED.csv'
LOG_FILE = 'empathy_ml_extraction_log.json'
CHECKPOINT_FILE = 'empathy_ml_checkpoint.json'

# Initialize Claude client
api_key = os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
client = anthropic.Anthropic(api_key=api_key)

EXTRACTION_PROMPT = """You are extracting structured information from an academic paper about empathy in NLP/AI.

Extract the following fields. If information is not available, write "N/A".

## Theoretical Foundations
1. empathy_definition: How does the paper define empathy?
2. empathy_type: Type of empathy studied (cognitive, affective, compassionate, perspective-taking, or combination)
3. empathy_measurement: How is empathy measured? (human annotation, self-report scales, behavioral outcomes, automatic metrics)
4. empathy_RQs: Primary research focus (detection, generation, measurement, evaluation, analysis)
5. disciplinary_perspective: Primary academic field (Psychology, Computer Science, HCI, Linguistics)

## Computational/ML Model
6. model_type: Type of model (transformer, RNN, rule-based, retrieval-based, LLM, hybrid)
7. model_name: Specific model names used (e.g., GPT-3, BERT, DialoGPT, T5, LLaMA)
8. input_modality: Input types (text, speech, multimodal, physiological)
9. output_modality: Output types (text, speech, behavioral recommendations, classification)
10. training_approach: Training methodology (supervised, reinforcement learning, few-shot, fine-tuning, prompt-based)
11. emotions_model: Does the model include emotion classification? (True/False)

## Dataset Information
12. dataset_name: Names of datasets used (e.g., EmpatheticDialogues, PersonaChat, MELD)
13. dataset_size: Number of samples/conversations if mentioned
14. dataset_domain: Application domain (therapy, customer service, general conversation, healthcare, social media)
15. dataset_sample_content: What each sample contains (dialogues, stories, utterances)
16. emotions_data: Does the dataset contain emotion labels? (True/False)
17. annotation_method: How was data annotated? (expert, crowdsourcing, self-report, automatic)
18. inter_annotator_agreement: Agreement metrics if mentioned (e.g., Kappa score)
19. participant_demographics: Demographics of data participants if mentioned

## Evaluation
20. evaluation_metrics: Metrics used (BLEU, ROUGE, F1, accuracy, perplexity, empathy scores)
21. human_evaluation: Was human evaluation conducted? (True/False)
22. evaluation_criteria: Specific criteria for empathy assessment
23. baseline_comparisons: What baselines were compared against?

## Application Context
24. application_domain: Target application (mental health, education, customer service, companionship, general)
25. user_population: Target users (general, clinical, elderly, children)
26. conversation_type: Type of interaction (open-domain, task-oriented, therapeutic, crisis intervention)

## Results and Claims
27. main_findings: Key results summary (1-2 sentences)
28. empathy_improvement: Did empathy improve over baseline? (True/False/N/A)
29. quantitative_results: Key numerical results (e.g., "F1: 0.85, Accuracy: 78%")
30. limitations_reported: Were limitations discussed? (True/False)
31. reported_limitations_summary: Brief summary of limitations
32. ethical_considerations: Were ethical implications addressed? (True/False)
33. ethical_considerations_summary: Brief summary of ethical points

## Contributions and Future Work
34. key_contribution: The single most important contribution (1 sentence)
35. suggested_future_work: What authors recommend for future research

Return ONLY a JSON object with these exact keys (use snake_case):
{
    "empathy_definition": "",
    "empathy_type": "",
    "empathy_measurement": "",
    "empathy_RQs": "",
    "disciplinary_perspective": "",
    "model_type": "",
    "model_name": "",
    "input_modality": "",
    "output_modality": "",
    "training_approach": "",
    "emotions_model": "",
    "dataset_name": "",
    "dataset_size": "",
    "dataset_domain": "",
    "dataset_sample_content": "",
    "emotions_data": "",
    "annotation_method": "",
    "inter_annotator_agreement": "",
    "participant_demographics": "",
    "evaluation_metrics": "",
    "human_evaluation": "",
    "evaluation_criteria": "",
    "baseline_comparisons": "",
    "application_domain": "",
    "user_population": "",
    "conversation_type": "",
    "main_findings": "",
    "empathy_improvement": "",
    "quantitative_results": "",
    "limitations_reported": "",
    "reported_limitations_summary": "",
    "ethical_considerations": "",
    "ethical_considerations_summary": "",
    "key_contribution": "",
    "suggested_future_work": ""
}
"""

def read_pdf_text(pdf_path, max_pages=20):
    """Extract text from PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text[:60000]  # Limit to ~60k chars
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

        # Parse JSON from response
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

        # Match by title substring
        if title_clean[:20] in pdf_clean:
            return pdf_path

        # Match by year and partial title
        if year in pdf_name:
            title_words = title.split()[:3]
            if any(w.lower() in pdf_name for w in title_words if len(w) > 4):
                return pdf_path

    return None

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'results': []}

def save_checkpoint(checkpoint):
    """Save checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def main():
    # Load paper list
    df = pd.read_csv(INPUT_CSV)
    print(f"Total ML papers: {len(df)}")

    # Get all PDFs
    pdf_files = []
    for pdf_dir in PDF_DIRS:
        if os.path.exists(pdf_dir):
            pdf_files.extend(list(Path(pdf_dir).glob('*.pdf')))
    print(f"Total PDFs available: {len(pdf_files)}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_titles = set(checkpoint['processed'])
    results = checkpoint['results']

    print(f"Already processed: {len(processed_titles)}")

    # Process papers
    for idx, row in df.iterrows():
        title = row.get('title', f'Paper {idx}')

        # Skip if already processed
        if title in processed_titles:
            continue

        # Find PDF
        pdf_path = find_pdf_for_paper(row, pdf_files)

        if not pdf_path:
            print(f"[{idx+1}/{len(df)}] No PDF: {title[:50]}...")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'task_type': row.get('task_type', ''),
                'pdf_status': 'NOT_FOUND'
            }
            results.append(result)
            checkpoint['processed'].append(title)
            checkpoint['results'] = results
            save_checkpoint(checkpoint)
            continue

        print(f"[{idx+1}/{len(df)}] Processing: {title[:50]}...")

        # Read PDF
        pdf_text = read_pdf_text(pdf_path)
        if not pdf_text:
            print(f"  ✗ Could not read PDF")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'task_type': row.get('task_type', ''),
                'pdf_status': 'READ_ERROR'
            }
            results.append(result)
            checkpoint['processed'].append(title)
            checkpoint['results'] = results
            save_checkpoint(checkpoint)
            continue

        # Extract with Claude
        extracted = extract_info_with_claude(pdf_text, title)

        if extracted:
            print(f"  ✓ Extracted")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'task_type': row.get('task_type', ''),
                'pdf_status': 'SUCCESS',
                **extracted
            }
        else:
            print(f"  ✗ Extraction failed")
            result = {
                'title': title,
                'year': row.get('year', ''),
                'database': row.get('database', ''),
                'task_type': row.get('task_type', ''),
                'pdf_status': 'EXTRACTION_FAILED'
            }

        results.append(result)
        checkpoint['processed'].append(title)
        checkpoint['results'] = results
        save_checkpoint(checkpoint)

        time.sleep(0.5)  # Rate limiting

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_excel(OUTPUT_XLSX, index=False)
    results_df.to_csv(OUTPUT_CSV, index=False)

    # Summary
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

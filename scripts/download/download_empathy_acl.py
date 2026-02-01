#!/usr/bin/env python3
"""
Download missing ACL Empathy papers
"""

import pandas as pd
import os
import re
import requests
import time

PDF_DIR = 'empathy_papers_pdfs'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Academic Research)'}

def clean_filename(title, max_len=50):
    """Clean title for filename"""
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean)
    return clean[:max_len]

def download_acl_pdf(url, output_path):
    """Download PDF from ACL Anthology"""
    try:
        pdf_url = url.rstrip('/') + '.pdf'
        response = requests.get(pdf_url, headers=HEADERS, timeout=30)

        if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', ''):
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False

# Load data
df = pd.read_excel('FINAL_EMPATHY_FOR_REVIEW.xlsx')
review_df = df[df['exclusion'] == 'no'].copy()
acl_df = review_df[review_df['database'].isin(['acl', 'acl_2025'])].copy()

# Get existing PDFs
existing_pdfs = [f.lower() for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

# Find missing
missing = []
for i, row in acl_df.iterrows():
    title_clean = re.sub(r'[^a-zA-Z0-9]', '', row['title'].lower())[:25]
    found = False
    for pdf in existing_pdfs:
        pdf_clean = re.sub(r'[^a-zA-Z0-9]', '', pdf.lower())
        if title_clean[:20] in pdf_clean:
            found = True
            break
    if not found and 'aclanthology.org' in str(row['url']):
        missing.append(row)

print(f"Missing ACL papers: {len(missing)}")
print()

# Download
success = 0
failed = 0

for i, row in enumerate(missing):
    url = row['url']
    title = row['title']

    # Create filename from URL
    url_id = url.split('/')[-1].replace('/', '_')
    filename = f"{url_id}_{clean_filename(title)}.pdf"
    output_path = os.path.join(PDF_DIR, filename)

    if os.path.exists(output_path):
        print(f"[{i+1}/{len(missing)}] Already exists: {filename[:50]}")
        success += 1
        continue

    print(f"[{i+1}/{len(missing)}] Downloading: {title[:50]}...")

    if download_acl_pdf(url, output_path):
        print(f"  ✓ Saved")
        success += 1
    else:
        print(f"  ✗ Failed")
        failed += 1

    time.sleep(0.5)  # Rate limiting

print()
print("=" * 50)
print(f"Success: {success}")
print(f"Failed: {failed}")

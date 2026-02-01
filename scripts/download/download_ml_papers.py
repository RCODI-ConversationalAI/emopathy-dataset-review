#!/usr/bin/env python3
"""
ML Benchmark Papers PDF Downloader
Downloads PDFs from ACL Anthology and via DOI resolution
"""

import pandas as pd
import requests
import os
import time
import json
from datetime import datetime
from urllib.parse import urlparse
import re

# Configuration
INPUT_FILE = 'LIST_ML_BENCHMARK_PAPERS.csv'
OUTPUT_DIR = 'ml_benchmark_pdfs'
LOG_FILE = 'download_log.json'
FAILED_FILE = 'download_failed.csv'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Academic Research Bot'
}

def sanitize_filename(title):
    """Create safe filename from title"""
    # Remove special characters, keep alphanumeric and spaces
    clean = re.sub(r'[^\w\s-]', '', str(title))
    clean = re.sub(r'\s+', '_', clean)
    return clean[:100]  # Limit length

def download_acl_pdf(url, output_path):
    """Download PDF from ACL Anthology"""
    # ACL Anthology: append .pdf to URL
    pdf_url = url.rstrip('/') + '.pdf'

    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=30, allow_redirects=True)
        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True, pdf_url
        else:
            return False, f"Status {response.status_code}, Content-Type: {response.headers.get('Content-Type', 'unknown')}"
    except Exception as e:
        return False, str(e)

def download_via_doi(doi, output_path):
    """Try to download PDF via DOI using multiple sources"""
    if not doi or pd.isna(doi):
        return False, "No DOI"

    doi = str(doi).strip()

    # 1. Try Unpaywall API (free, no key needed for limited use)
    try:
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
        response = requests.get(unpaywall_url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # Look for open access PDF
            if data.get('best_oa_location'):
                pdf_url = data['best_oa_location'].get('url_for_pdf')
                if pdf_url:
                    pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=30)
                    if pdf_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(pdf_response.content)
                        return True, f"Unpaywall: {pdf_url}"
    except Exception as e:
        pass

    # 2. Try Semantic Scholar API
    try:
        s2_url = f"https://api.semanticscholar.org/v1/paper/{doi}"
        response = requests.get(s2_url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('openAccessPdf'):
                pdf_url = data['openAccessPdf'].get('url')
                if pdf_url:
                    pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=30)
                    if pdf_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(pdf_response.content)
                        return True, f"SemanticScholar: {pdf_url}"
    except Exception as e:
        pass

    # 3. Try arXiv if DOI contains arxiv
    if 'arxiv' in doi.lower():
        try:
            arxiv_id = doi.split('/')[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(pdf_url, headers=HEADERS, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True, f"arXiv: {pdf_url}"
        except Exception as e:
            pass

    return False, "No open access PDF found"

def download_scopus_via_doi(url, doi, output_path):
    """Handle Scopus URLs by using DOI"""
    return download_via_doi(doi, output_path)

def main():
    # Load papers
    df = pd.read_csv(INPUT_FILE)
    print(f"Total papers to download: {len(df)}")

    # Track results
    results = {
        'total': len(df),
        'success': 0,
        'failed': 0,
        'start_time': datetime.now().isoformat(),
        'papers': []
    }

    failed_papers = []

    for idx, row in df.iterrows():
        title = row.get('title', f'paper_{idx}')
        url = row.get('url', '')
        doi = row.get('doi', '')
        year = row.get('year', '')

        filename = f"{idx:04d}_{sanitize_filename(title)}.pdf"
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"[{idx+1}/{len(df)}] Already exists: {filename[:50]}...")
            results['success'] += 1
            results['papers'].append({
                'idx': idx,
                'title': title,
                'status': 'exists',
                'file': filename
            })
            continue

        print(f"[{idx+1}/{len(df)}] Downloading: {title[:50]}...")

        success = False
        message = ""

        # Determine source and download
        if pd.notna(url) and 'aclanthology.org' in str(url):
            success, message = download_acl_pdf(url, output_path)
        elif pd.notna(url) and 'scopus.com' in str(url):
            success, message = download_scopus_via_doi(url, doi, output_path)
        elif pd.notna(doi):
            success, message = download_via_doi(doi, output_path)
        else:
            message = "No URL or DOI available"

        if success:
            results['success'] += 1
            results['papers'].append({
                'idx': idx,
                'title': title,
                'status': 'success',
                'file': filename,
                'source': message
            })
            print(f"  ✓ Success: {message[:50]}")
        else:
            results['failed'] += 1
            results['papers'].append({
                'idx': idx,
                'title': title,
                'status': 'failed',
                'reason': message
            })
            failed_papers.append({
                'idx': idx,
                'title': title,
                'authors': row.get('authors', ''),
                'year': year,
                'doi': doi,
                'url': url,
                'benchmarks': row.get('benchmarks_used', ''),
                'reason': message
            })
            print(f"  ✗ Failed: {message}")

        # Rate limiting
        time.sleep(0.5)

        # Save progress every 50 papers
        if (idx + 1) % 50 == 0:
            results['end_time'] = datetime.now().isoformat()
            with open(LOG_FILE, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n--- Progress: {results['success']}/{idx+1} success ---\n")

    # Final save
    results['end_time'] = datetime.now().isoformat()
    with open(LOG_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save failed papers
    if failed_papers:
        failed_df = pd.DataFrame(failed_papers)
        failed_df.to_csv(FAILED_FILE, index=False)

    # Summary
    print("\n" + "="*50)
    print("DOWNLOAD COMPLETE")
    print("="*50)
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']} ({results['success']/results['total']*100:.1f}%)")
    print(f"Failed: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    print(f"\nLog saved to: {LOG_FILE}")
    print(f"Failed papers saved to: {FAILED_FILE}")
    print(f"PDFs saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()

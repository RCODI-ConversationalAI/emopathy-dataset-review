import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import re

# Settings
CSV_PATH = 'ACL_EMPATHY_DATASET_PAPERS.csv'
PDF_DIR = 'empathy_papers_pdfs'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

def fetch_title_from_acl(url):
    """Fetch correct title from ACL Anthology page"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Title is in <h2 id="title"> or <meta property="og:title">
        title_tag = soup.find('h2', {'id': 'title'})
        if title_tag:
            # Clean up the title
            title = title_tag.get_text(strip=True)
            return title

        # Fallback to og:title meta
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '').strip()

        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def download_pdf(url, pdf_dir, filename):
    """Download PDF from ACL Anthology"""
    # Convert page URL to PDF URL
    pdf_url = url.rstrip('/') + '.pdf'
    filepath = os.path.join(pdf_dir, filename)

    if os.path.exists(filepath):
        print(f"  Already exists: {filename}")
        return True

    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"  Error downloading {pdf_url}: {e}")
        return False

def sanitize_filename(title, url):
    """Create safe filename from title"""
    # Extract paper ID from URL
    paper_id = url.rstrip('/').split('/')[-1]

    # Clean title for filename
    clean_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
    clean_title = re.sub(r'\s+', '_', clean_title)

    return f"{paper_id}_{clean_title}.pdf"

def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Filter included papers only
    included = df[df['exclusion'] != 'yes'].copy()
    print(f"Processing {len(included)} included papers...\n")

    # Create PDF directory if not exists
    os.makedirs(PDF_DIR, exist_ok=True)

    updated_titles = {}
    download_results = []

    for idx, row in included.iterrows():
        url = row['url']
        old_title = row['title']
        print(f"[{included.index.get_loc(idx)+1}/{len(included)}] {url}")

        # Fetch correct title
        new_title = fetch_title_from_acl(url)
        if new_title:
            updated_titles[idx] = new_title
            print(f"  Title: {new_title[:70]}...")
        else:
            new_title = old_title
            print(f"  Title: (kept original) {old_title[:50]}...")

        # Download PDF
        filename = sanitize_filename(new_title, url)
        success = download_pdf(url, PDF_DIR, filename)
        download_results.append({'url': url, 'title': new_title, 'filename': filename, 'success': success})

        # Rate limiting
        time.sleep(0.5)

    # Update CSV with new titles
    for idx, title in updated_titles.items():
        df.at[idx, 'title'] = title

    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✓ Updated {len(updated_titles)} titles in {CSV_PATH}")

    # Summary
    successful = sum(1 for r in download_results if r['success'])
    print(f"✓ Downloaded {successful}/{len(download_results)} PDFs to {PDF_DIR}/")

    # Save download log
    log_df = pd.DataFrame(download_results)
    log_df.to_csv('empathy_download_log.csv', index=False)
    print(f"✓ Download log saved to empathy_download_log.csv")

if __name__ == '__main__':
    main()

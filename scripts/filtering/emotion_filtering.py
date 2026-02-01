import bibtexparser
import requests
import json
import time
import csv
import os
from typing import Dict, List, Optional
import re
from urllib.parse import urlparse
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv

class EmotionPaperFilter:
    def __init__(self, claude_api_key: str):
        self.claude_api_key = claude_api_key
        self.claude_url = "https://api.anthropic.com/v1/messages"
        
    def load_bibtex_file(self, file_path: str):
        """Load and parse a BibTeX file."""
        with open(file_path, 'r', encoding='utf-8') as bibtex_file:
            parser = bibtexparser.bparser.BibTexParser(common_strings=True)
            return bibtexparser.load(bibtex_file, parser)
    
    def extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """Download and extract text from PDF entirely in-memory."""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text[:8000]
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return None
    
    def analyze_emotion_paper(self, title: str, abstract: str, pdf_text: str = None) -> Dict:
        """Analyze emotion-only paper with simple prompt."""
        prompt = f"""
        Extract key info from ACL paper:

        Title: {title}
        Abstract: {abstract}
        {f"PDF Content: {pdf_text}" if pdf_text else ""}

        Return JSON with:
        {{
            "model_type": "transformer|RNN|CNN|rule-based|other",
            "dataset_name": "name if mentioned",
            "task_type": "classification|generation|detection|recognition",
            "performance": "F1, Accuracy, BLEU scores if mentioned"
        }}
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(self.claude_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result['content'][0]['text']
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"model_type": "unknown", "dataset_name": "unknown", "task_type": "unknown", "performance": "unknown"}
                
        except Exception as e:
            print(f"Claude API error: {e}")
            return {"model_type": "unknown", "dataset_name": "unknown", "task_type": "unknown", "performance": "unknown"}
    
    def get_pdf_url(self, entry: Dict) -> Optional[str]:
        """Extract PDF URL from paper entry."""
        url_fields = ['url', 'pdf', 'link', 'doi']
        
        for field in url_fields:
            if field in entry and entry[field]:
                url = entry[field]
                if url.endswith('.pdf') or 'arxiv.org' in url or 'aclweb.org' in url:
                    return url
        
        if 'doi' in entry and entry['doi']:
            return f"https://doi.org/{entry['doi']}"
        
        return None
    
    def analyze_papers(self, bib_database, max_papers: int = None) -> List[Dict]:
        """Analyze emotion papers with simple processing."""
        papers_info = []
        
        for i, entry in enumerate(bib_database.entries):
            if max_papers and i >= max_papers:
                print(f"\nReached limit of {max_papers} papers. Stopping analysis.")
                break
                
            title = entry.get('title', '')
            year = entry.get('year', '')
            
            if year and year.isdigit() and int(year) < 2014:
                continue
                
            print(f"Processing paper {i+1}/{len(bib_database.entries)}: {title[:50]}... (year: {year})")
            
            abstract = entry.get('abstract', '')
            
            # Simple emotion classification
            if not self._is_emotion_related(title, abstract):
                print(f"  ✗ EXCLUDED: not emotion-related")
                continue
            
            # Get PDF content
            pdf_text = None
            pdf_url = self.get_pdf_url(entry)
            if pdf_url:
                print(f"  Extracting PDF from: {pdf_url}")
                pdf_text = self.extract_pdf_text(pdf_url)
                if pdf_text:
                    print("  ✓ PDF text extracted")
                else:
                    print("  ⚠ PDF extraction failed")
            
            # Analyze paper
            analysis = self.analyze_emotion_paper(title, abstract, pdf_text)
            
            paper_info = {
                'title': title,
                'authors': entry.get('author', ''),
                'year': entry.get('year', ''),
                'booktitle': entry.get('booktitle', ''),
                'doi': entry.get('doi', ''),
                'abstract': abstract,
                'url': entry.get('url', ''),
                'category': 'emotion',
                'model_type': analysis.get('model_type', 'unknown'),
                'dataset_name': analysis.get('dataset_name', 'unknown'),
                'task_type': analysis.get('task_type', 'unknown'),
                'performance': analysis.get('performance', 'unknown'),
                'pdf_used': bool(pdf_text)
            }
            papers_info.append(paper_info)
            print(f"  ✓ INCLUDED: emotion paper")
            
            time.sleep(1)
        
        return papers_info
    
    def _is_emotion_related(self, title: str, abstract: str) -> bool:
        """Simple check if paper is emotion-related."""
        text = f"{title} {abstract}".lower()
        emotion_keywords = ['emotion', 'affect', 'sentiment', 'feeling', 'mood', 'valence', 'arousal']
        return any(keyword in text for keyword in emotion_keywords)
    
    def save_results(self, papers_info: List[Dict], output_file: str):
        """Save results to CSV."""
        fieldnames = [
            'title', 'authors', 'year', 'booktitle', 'doi', 'abstract', 
            'url', 'category', 'model_type', 'dataset_name', 'task_type', 
            'performance', 'pdf_used'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for paper in papers_info:
                paper_copy = paper.copy()
                paper_copy['pdf_used'] = 'Yes' if paper_copy['pdf_used'] else 'No'
                writer.writerow(paper_copy)
    
    def compute_statistics(self, papers_info: List[Dict]) -> Dict:
        """Compute emotion paper statistics."""
        stats = {
            "total_papers": len(papers_info),
            "model_types": {},
            "task_types": {},
            "datasets": {},
            "years": {}
        }
        
        for paper in papers_info:
            # Model type counts
            model_type = paper.get('model_type', 'unknown')
            stats["model_types"][model_type] = stats["model_types"].get(model_type, 0) + 1
            
            # Task type counts
            task_type = paper.get('task_type', 'unknown')
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
            
            # Dataset counts
            dataset = paper.get('dataset_name', 'unknown')
            if dataset != 'unknown':
                stats["datasets"][dataset] = stats["datasets"].get(dataset, 0) + 1
            
            # Year counts
            year = paper.get('year', 'unknown')
            if year != 'unknown':
                stats["years"][year] = stats["years"].get(year, 0) + 1
        
        return stats
    
    def write_statistics_report(self, stats: Dict, output_file: str):
        """Write statistics report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            def w(line: str):
                print(line)
                f.write(line + "\n")
            
            w("Emotion Paper Statistics (Simple Processing):")
            w(f"Total papers: {stats['total_papers']}")
            w("")
            
            w("Model Types:")
            for model_type, count in stats["model_types"].items():
                w(f"  {model_type}: {count}")
            w("")
            
            w("Task Types:")
            for task_type, count in stats["task_types"].items():
                w(f"  {task_type}: {count}")
            w("")
            
            w("Datasets:")
            for dataset, count in stats["datasets"].items():
                w(f"  {dataset}: {count}")
            w("")
            
            w("Years:")
            for year, count in sorted(stats["years"].items()):
                w(f"  {year}: {count}")

def main():
    load_dotenv()
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    
    if not CLAUDE_API_KEY:
        raise RuntimeError("Missing CLAUDE_API_KEY environment variable")
    
    input_file = '/Volumes/ssd/01-ckj-postdoc/emopathy-dataset-review-local/boolean-search/all-zot-items/anthology+abstracts-aug2025.bib'
    output_csv = 'emotion_papers_simple.csv'
    stats_file = 'emotion_statistics.txt'
    
    filter = EmotionPaperFilter(CLAUDE_API_KEY)
    
    print("Loading BibTeX file...")
    bib_database = filter.load_bibtex_file(input_file)
    
    print("Starting emotion paper analysis (simple processing)...")
    papers_info = filter.analyze_papers(bib_database, max_papers=100)
    
    filter.save_results(papers_info, output_csv)
    stats = filter.compute_statistics(papers_info)
    filter.write_statistics_report(stats, stats_file)
    
    print(f"\nAnalysis complete!")
    print(f"Found {len(papers_info)} emotion papers")
    print(f"Results saved to {output_csv}")
    print(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()
import bibtexparser
import requests
import json
import time
import csv
import os
from typing import Dict, List, Optional, Tuple
import re
from urllib.parse import urlparse
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('empathy_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedEmpathyExtractor:
    def __init__(self, claude_api_key: str, checkpoint_file: str = 'extraction_checkpoint.json'):
        self.claude_api_key = claude_api_key
        self.claude_url = "https://api.anthropic.com/v1/messages"
        self.checkpoint_file = checkpoint_file
        
        self.stats = {
            'total_processed': 0,
            'abstract_based': 0,
            'title_only': 0,
            'pdf_extracted': 0,
            'empathy_found': 0,
            'api_calls': 0,
            'failures': 0,
            'by_year': defaultdict(int)
        }
        
        # Checkpoint data
        self.processed_ids = set()
        self.results = []
        self.load_checkpoint()
        
        # Extended keyword set (diagnostic results)
        self.empathy_keywords = {
            'core': ['empathy', 'empathic', 'empathetic', 'empathize', 'empathizing'],
            'mental_health': ['mental health', 'psychological support', 'emotional wellbeing', 
                            'mental wellness', 'psychological wellbeing'],
            'related': ['compassion', 'compassionate', 'sympathy', 'sympathetic'],
            'cognitive': ['perspective-taking', 'perspective taking', 'theory of mind', 
                         'mentalizing', 'mind-reading', 'mindreading'],
            'affective': ['emotional contagion', 'emotional resonance', 'vicarious', 
                         'emotional understanding'],
            'technical': ['empathetic dialogue', 'empathetic response', 'empathy detection',
                         'empathy recognition', 'empathy classification', 'empathy classification',
                         'empathy generation', 'empathetic conversation'],
            'application': ['therapeutic', 'counseling', 'supportive', 'emotional support']
        }
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.processed_ids = set(checkpoint.get('processed_ids', []))
                    self.results = checkpoint.get('results', [])
                    self.stats = checkpoint.get('stats', self.stats)
                    logger.info(f"Checkpoint loaded: {len(self.processed_ids)} already processed")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
    
    def save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint = {
            'processed_ids': list(self.processed_ids),
            'results': self.results,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {len(self.results)} results")
    
    def load_bibtex_file(self, file_path: str):
        """BibTeX 파일 로드"""
        logger.info(f"Loading BibTeX file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as bibtex_file:
            parser = bibtexparser.bparser.BibTexParser(common_strings=True)
            bib_database = bibtexparser.load(bibtex_file, parser)
        logger.info(f"Loaded {len(bib_database.entries)} entries")
        return bib_database
    
    def is_empathy_related(self, title: str, abstract: str = "", strict: bool = False) -> Tuple[bool, str]:
        """
        Empathy 관련 여부 판단
        Returns: (is_related, matched_category:keyword)
        """
        text = f"{title} {abstract}".lower()
        
        # 키워드 매칭
        for category, keywords in self.empathy_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return True, f"{category}:{keyword}"
        
        # Strict 모드가 아니면 emotion + context 조합도 체크
        if not strict:
            emotion_keywords = ['emotion', 'emotional', 'feeling', 'affect', 'sentiment']
            context_keywords = ['understanding', 'recognition', 'detection', 'support', 
                              'dialogue', 'conversation', 'response']
            
            has_emotion = any(ek in text for ek in emotion_keywords)
            has_context = any(ck in text for ck in context_keywords)
            
            if has_emotion and has_context:
                # 여기서 Claude로 정밀 체크할 수도 있음
                return True, "emotion+context"
        
        return False, None
    
    def extract_pdf_with_retry(self, pdf_url: str, max_retries: int = 3) -> Optional[str]:
        """PDF 텍스트 추출 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"PDF extraction attempt {attempt+1}: {pdf_url}")
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                
                doc = fitz.open(stream=response.content, filetype="pdf")
                text = ""
                
                # 전체 텍스트 추출 (최대 50000자)
                for page_num, page in enumerate(doc):
                    if len(text) > 50000:  # 충분한 텍스트 확보
                        break
                    text += page.get_text()
                
                doc.close()
                
                if len(text) > 100:  # 최소한의 텍스트가 있는지 확인
                    return text
                    
            except Exception as e:
                logger.warning(f"PDF extraction attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def extract_abstract_from_pdf(self, pdf_text: str) -> Optional[str]:
        """PDF 텍스트에서 Abstract 추출"""
        if not pdf_text:
            return None
        
        # Abstract 섹션 찾기
        abstract_patterns = [
            r'abstract\s*\n(.*?)(?=\n\s*(?:introduction|keywords|1\.|1\s+introduction))',
            r'abstract(.*?)(?:introduction|keywords)',
            r'ABSTRACT\s*\n(.*?)(?=\n\s*(?:INTRODUCTION|Keywords|1\.))'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, pdf_text[:5000], re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # 클린업
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:1000]  # 최대 1000자
        
        return None
    
    def validate_and_clean_extraction(self, result: Dict) -> Dict:
        """36개 필드 검증 및 데이터 타입 정리"""
        # 필수 필드 목록
        required_fields = [
            # Theoretical Foundations
            'empathy_definition', 'empathy_type', 'empathy_measurement', 
            'empathy_RQs', 'disciplinary_perspective',
            # Computational/ML Model
            'model_type', 'model_name', 'input_modality', 
            'output_modality', 'training_approach',
            # Dataset Information
            'dataset_name', 'dataset_size', 'dataset_domain',
            'annotation_method', 'inter_annotator_agreement', 'participant_demographics',
            # Evaluation
            'evaluation_metrics', 'human_evaluation', 'evaluation_criteria',
            'baseline_comparisons',
            # Application Context
            'application_domain', 'user_population', 'conversation_type',
            # Results and Claims
            'main_findings', 'empathy_improvement', 'quantitative_results',
            'limitations_reported', 'reported_limitations_summary',
            'ethical_considerations', 'ethical_considerations_summary',
            # Methodology Quality
            'sample_size_adequate', 'methodology_rigor', 
            'reproducibility', 'statistical_analysis',
            # Contributions and Future Work
            'key_contribution', 'suggested_future_work'
        ]
        
        # 누락된 필드 채우기
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field: {field}")
                result[field] = "not_found"
        
        # Boolean 필드 타입 변환
        boolean_fields = ['human_evaluation', 'empathy_improvement', 
                         'limitations_reported', 'ethical_considerations']
        for field in boolean_fields:
            if field in result:
                if isinstance(result[field], str):
                    result[field] = result[field].lower() in ['true', 'yes', '1']
                elif result[field] is None:
                    result[field] = False
        
        # Integer 필드 처리
        if 'dataset_size' in result:
            try:
                if result['dataset_size'] not in ['not_found', None, 'null']:
                    result['dataset_size'] = int(result['dataset_size'])
                else:
                    result['dataset_size'] = None
            except:
                result['dataset_size'] = None
        
        if 'methodology_rigor' in result:
            try:
                result['methodology_rigor'] = int(result['methodology_rigor'])
                result['methodology_rigor'] = max(1, min(5, result['methodology_rigor']))
            except:
                result['methodology_rigor'] = 3  # default
        
        # Float 필드 처리
        if 'inter_annotator_agreement' in result:
            try:
                if result['inter_annotator_agreement'] not in ['not_found', None, 'null']:
                    result['inter_annotator_agreement'] = float(result['inter_annotator_agreement'])
                else:
                    result['inter_annotator_agreement'] = None
            except:
                result['inter_annotator_agreement'] = None
        
        return result
    
    def analyze_with_claude(self, title: str, abstract: str, pdf_text: str) -> Dict:
        """Claude API를 사용한 36개 필드 완전 추출"""
        prompt = f"""
        Analyze this ACL paper for comprehensive empathy research review. 
        Extract ALL 36 fields as specified below. If information is not found, use "not_found" rather than guessing.
        
        Title: {title}
        Abstract: {abstract}
        PDF Content: {pdf_text[:30000]}
        
        Return a JSON with EXACTLY these fields:
        
        === THEORETICAL FOUNDATIONS ===
        - empathy_definition: How the paper defines empathy (text)
        - empathy_type: Choose from: "cognitive", "affective", "compassionate", "perspective-taking", "other"
        - empathy_measurement: Choose from: "human annotation", "self-report scales", "behavioral outcomes", "other"
        - empathy_RQs: Choose from: "detection", "generation", "measurement", "evaluation", "other"
        - disciplinary_perspective: Choose from: "Psychology", "Computer Science", "HCI", "Linguistics", "other"
        
        === COMPUTATIONAL/ML MODEL ===
        - model_type: Choose from: "transformer", "RNN", "CNN", "rule-based", "retrieval-based", "hybrid", "other"
        - model_name: Specific model (e.g., "GPT-3", "BERT", "DialoGPT", "T5") or "not_found"
        - input_modality: Choose from: "text", "speech", "multimodal", "physiological", "other"
        - output_modality: Choose from: "text", "speech", "behavioral recommendations", "other"
        - training_approach: Choose from: "supervised", "reinforcement learning", "few-shot", "fine-tuning", "other"
        
        === DATASET INFORMATION ===
        - dataset_name: Name of dataset (e.g., "EmpatheticDialogues") or "not_found"
        - dataset_size: Number as integer (e.g., 25000) or null if not found
        - dataset_domain: Choose from: "therapy", "customer service", "general conversation", "healthcare", "other"
        - annotation_method: Choose from: "expert annotation", "crowdsourcing", "self-report", "other"
        - inter_annotator_agreement: Float value (e.g., 0.75) or null if not reported
        - participant_demographics: Text description of participants or "not_found"
        
        === EVALUATION ===
        - evaluation_metrics: List metrics separated by semicolons (e.g., "BLEU; empathy scores; user satisfaction")
        - human_evaluation: true or false
        - evaluation_criteria: Text describing specific criteria or "not_found"
        - baseline_comparisons: List baselines (e.g., "GPT-2; rule-based system") or "not_found"
        
        === APPLICATION CONTEXT ===
        - application_domain: Choose from: "mental health", "education", "customer service", "companionship", "other"
        - user_population: Choose from: "general", "clinical", "elderly", "children", "other"
        - conversation_type: Choose from: "open-domain", "task-oriented", "therapeutic", "crisis intervention", "other"
        
        === RESULTS AND CLAIMS ===
        - main_findings: Text summary of key results
        - empathy_improvement: true or false
        - quantitative_results: Key numbers (e.g., "Empathy score: 4.2/5; F1: 0.78")
        - limitations_reported: true or false
        - reported_limitations_summary: Text summary of limitations or "not_found"
        - ethical_considerations: true or false
        - ethical_considerations_summary: Text summary of ethical points or "not_found"
        
        === METHODOLOGY QUALITY (Your Assessment) ===
        Rate based on what you can observe in the paper:
        - sample_size_adequate: Choose from: "low", "medium", "high"
        - methodology_rigor: Integer 1-5 (1=poor, 5=excellent)
        - reproducibility: Choose from: "none", "partial", "full" (based on code/data availability)
        - statistical_analysis: Choose from: "poor", "fair", "good", "excellent"
        
        === CONTRIBUTIONS AND FUTURE WORK ===
        - key_contribution: Single most important novel contribution (text)
        - suggested_future_work: What authors recommend for future research (text)
        
        IMPORTANT INSTRUCTIONS:
        1. Extract information ONLY from the provided text, do not use external knowledge
        2. For categorical fields, ONLY use the provided options
        3. Use "not_found" for missing text fields, null for missing numbers
        4. For methodology_quality, make your own assessment based on the paper
        5. Be precise with numerical values - extract exact numbers when available
        
        Return a valid JSON with all 36 fields.
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(self.claude_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result['content'][0]['text']
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                self.stats['api_calls'] += 1
                extracted_data = json.loads(json_match.group())
                
                # 36개 필드 검증
                validated_data = self.validate_and_clean_extraction(extracted_data)
                return validated_data
            else:
                return {"error": "Failed to parse Claude response"}
                
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"error": f"API error: {e}"}
    
    def get_pdf_url(self, entry: Dict) -> Optional[str]:
        """PDF URL 추출"""
        # URL 필드 체크
        url_fields = ['url', 'pdf', 'link']
        for field in url_fields:
            if field in entry and entry[field]:
                url = entry[field]
                if '.pdf' in url or 'arxiv.org' in url or 'aclweb.org' in url:
                    return url
        
        # DOI를 통한 URL 생성
        if 'doi' in entry and entry['doi']:
            return f"https://doi.org/{entry['doi']}"
        
        return None
    
    def process_batch(self, entries: List[Dict], batch_name: str = "batch") -> List[Dict]:
        """배치 처리"""
        logger.info(f"Processing {batch_name}: {len(entries)} entries")
        batch_results = []
        
        for i, entry in enumerate(entries):
            # 이미 처리된 경우 스킵
            entry_id = entry.get('ID', str(hash(entry.get('title', ''))))
            if entry_id in self.processed_ids:
                continue
            
            # 연도 체크 (2014년 이후만)
            year = entry.get('year', '')
            if year and year.isdigit() and int(year) < 2014:
                continue
            
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')
            
            # 진행 상황 출력
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(entries)} in {batch_name}")
                self.save_checkpoint()  # 주기적 저장
            
            # Empathy 관련성 체크
            is_related, match_info = self.is_empathy_related(title, abstract)
            
            if not is_related:
                continue
            
            logger.info(f"  Found empathy paper: {title[:60]}... ({match_info})")
            
            # PDF URL 가져오기
            pdf_url = self.get_pdf_url(entry)
            pdf_text = None
            
            if pdf_url:
                pdf_text = self.extract_pdf_with_retry(pdf_url)
                if pdf_text:
                    self.stats['pdf_extracted'] += 1
                    
                    # Abstract 없으면 PDF에서 추출 시도
                    if not abstract:
                        abstract = self.extract_abstract_from_pdf(pdf_text) or ""
            
            # Claude 분석 (PDF 텍스트가 있는 경우만)
            analysis = {}
            if pdf_text and len(pdf_text) > 500:
                analysis = self.analyze_with_claude(title, abstract, pdf_text)
                time.sleep(1)  # Rate limiting
            
            # 결과 저장
            result = {
                'id': entry_id,
                'title': title,
                'authors': entry.get('author', ''),
                'year': year,
                'booktitle': entry.get('booktitle', ''),
                'doi': entry.get('doi', ''),
                'url': entry.get('url', ''),
                'abstract': abstract,
                'match_info': match_info,
                'pdf_available': pdf_url is not None,
                'pdf_extracted': pdf_text is not None,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            batch_results.append(result)
            self.results.append(result)
            self.processed_ids.add(entry_id)
            
            # 통계 업데이트
            self.stats['empathy_found'] += 1
            if year:
                self.stats['by_year'][year] += 1
            
            if abstract:
                self.stats['abstract_based'] += 1
            else:
                self.stats['title_only'] += 1
        
        return batch_results
    
    def run_extraction(self, bib_database, target_count: int = 875):
        """전체 추출 프로세스 실행"""
        logger.info("="*70)
        logger.info("STARTING EMPATHY PAPER EXTRACTION")
        logger.info(f"Target: ~{target_count} papers")
        logger.info("="*70)
        
        entries = bib_database.entries
        
        # Phase 1: Abstract 있는 논문 우선 처리
        logger.info("\n[Phase 1] Processing papers WITH abstracts")
        papers_with_abstract = [e for e in entries if e.get('abstract')]
        logger.info(f"Found {len(papers_with_abstract)} papers with abstracts")
        
        # Empathy 관련 논문만 필터링
        empathy_candidates_with_abstract = []
        for entry in papers_with_abstract:
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')
            is_related, _ = self.is_empathy_related(title, abstract)
            if is_related:
                empathy_candidates_with_abstract.append(entry)
        
        logger.info(f"Found {len(empathy_candidates_with_abstract)} empathy candidates with abstracts")
        
        # 배치 처리
        phase1_results = self.process_batch(
            empathy_candidates_with_abstract, 
            "Phase 1 (with abstracts)"
        )
        
        # Phase 2: Abstract 없지만 Title에 empathy 키워드 있는 논문
        logger.info("\n[Phase 2] Processing papers WITHOUT abstracts (title-based)")
        papers_without_abstract = [e for e in entries if not e.get('abstract')]
        logger.info(f"Found {len(papers_without_abstract)} papers without abstracts")
        
        # Title로 필터링
        empathy_candidates_title_only = []
        for entry in papers_without_abstract:
            title = entry.get('title', '')
            is_related, _ = self.is_empathy_related(title, "", strict=True)
            if is_related:
                empathy_candidates_title_only.append(entry)
        
        logger.info(f"Found {len(empathy_candidates_title_only)} empathy candidates (title only)")
        
        # 배치 처리
        phase2_results = self.process_batch(
            empathy_candidates_title_only,
            "Phase 2 (title only)"
        )
        
        # Phase 3: 통계 검증 및 누락 체크
        logger.info("\n[Phase 3] Validation and coverage check")
        self.validate_coverage(target_count)
        
        # 최종 저장
        self.save_checkpoint()
        self.save_final_results()
        
        return self.results
    
    def validate_coverage(self, target_count: int):
        """커버리지 검증"""
        found_count = len(self.results)
        coverage = (found_count / target_count) * 100 if target_count > 0 else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"COVERAGE VALIDATION")
        logger.info(f"{'='*50}")
        logger.info(f"Target papers: {target_count}")
        logger.info(f"Found papers: {found_count}")
        logger.info(f"Coverage: {coverage:.1f}%")
        
        # 연도별 분포
        logger.info("\nYear distribution:")
        for year in sorted(self.stats['by_year'].keys()):
            logger.info(f"  {year}: {self.stats['by_year'][year]}")
        
        # 처리 통계
        logger.info(f"\nProcessing statistics:")
        logger.info(f"  Abstract-based: {self.stats['abstract_based']}")
        logger.info(f"  Title-only: {self.stats['title_only']}")
        logger.info(f"  PDFs extracted: {self.stats['pdf_extracted']}")
        logger.info(f"  API calls made: {self.stats['api_calls']}")
        
        if coverage < 70:
            logger.warning(f"\n⚠️ Low coverage ({coverage:.1f}%). Consider:")
            logger.warning("  - Expanding keyword list")
            logger.warning("  - Checking for data issues")
            logger.warning("  - Manual inspection of missed papers")
    
    def save_final_results(self):
        """최종 결과 저장 (36개 필드 포함)"""
        # CSV 저장 - 36개 필드 모두 포함
        csv_file = f'empathy_papers_extracted_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # 메타데이터 필드
        metadata_fields = ['id', 'title', 'authors', 'year', 'booktitle', 'doi', 'url', 
                          'abstract', 'match_info', 'pdf_available', 'pdf_extracted', 'timestamp']
        
        # 36개 템플릿 필드
        template_fields = [
            # Theoretical Foundations
            'empathy_definition', 'empathy_type', 'empathy_measurement', 
            'empathy_RQs', 'disciplinary_perspective',
            # Computational/ML Model
            'model_type', 'model_name', 'input_modality', 
            'output_modality', 'training_approach',
            # Dataset Information
            'dataset_name', 'dataset_size', 'dataset_domain',
            'annotation_method', 'inter_annotator_agreement', 'participant_demographics',
            # Evaluation
            'evaluation_metrics', 'human_evaluation', 'evaluation_criteria',
            'baseline_comparisons',
            # Application Context
            'application_domain', 'user_population', 'conversation_type',
            # Results and Claims
            'main_findings', 'empathy_improvement', 'quantitative_results',
            'limitations_reported', 'reported_limitations_summary',
            'ethical_considerations', 'ethical_considerations_summary',
            # Methodology Quality
            'sample_size_adequate', 'methodology_rigor', 
            'reproducibility', 'statistical_analysis',
            # Contributions and Future Work
            'key_contribution', 'suggested_future_work'
        ]
        
        fieldnames = metadata_fields + template_fields
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {}
                
                # 메타데이터 복사
                for field in metadata_fields:
                    row[field] = result.get(field, '')
                
                # Analysis 데이터 평탄화
                analysis = result.get('analysis', {})
                if isinstance(analysis, dict) and 'error' not in analysis:
                    for field in template_fields:
                        row[field] = analysis.get(field, 'not_found')
                else:
                    # Analysis 실패한 경우
                    for field in template_fields:
                        row[field] = 'not_analyzed'
                
                writer.writerow(row)
        
        logger.info(f"\n✅ Results saved to {csv_file}")
        
        # 통계 리포트 저장
        stats_file = f'empathy_extraction_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"✅ Statistics saved to {stats_file}")
        
        # 상세 요약 리포트
        summary_file = f'empathy_extraction_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("EMPATHY PAPER EXTRACTION SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Total papers found: {len(self.results)}\n")
            f.write(f"Papers with abstracts: {self.stats['abstract_based']}\n")
            f.write(f"Papers from title only: {self.stats['title_only']}\n")
            f.write(f"PDFs successfully extracted: {self.stats['pdf_extracted']}\n")
            f.write(f"Claude API calls: {self.stats['api_calls']}\n")
            f.write("\nYear distribution:\n")
            for year in sorted(self.stats['by_year'].keys()):
                f.write(f"  {year}: {self.stats['by_year'][year]}\n")
            
            # 분석 완료도 체크
            analyzed_count = sum(1 for r in self.results 
                               if r.get('analysis') and 'error' not in r.get('analysis', {}))
            f.write(f"\nAnalysis completion:\n")
            f.write(f"  Successfully analyzed: {analyzed_count}/{len(self.results)}\n")
            f.write(f"  Analysis success rate: {analyzed_count/len(self.results)*100:.1f}%\n")
        
        logger.info(f"✅ Summary saved to {summary_file}")


def main():
    # 환경 변수 로드
    load_dotenv()
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    
    if not CLAUDE_API_KEY:
        raise RuntimeError("Missing CLAUDE_API_KEY environment variable")
    
    # 파일 경로
    input_file = '/Volumes/ssd/01-ckj-postdoc/emopathy-dataset-review-local/boolean-search/all-zot-items/anthology+abstracts-aug2025.bib'
    
    # Extractor 초기화
    extractor = ImprovedEmpathyExtractor(CLAUDE_API_KEY)
    
    # BibTeX 로드
    bib_database = extractor.load_bibtex_file(input_file)
    
    # 추출 실행
    results = extractor.run_extraction(bib_database, target_count=875)
    
    logger.info("\n" + "="*70)
    logger.info("EXTRACTION COMPLETE!")
    logger.info(f"Total papers extracted: {len(results)}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
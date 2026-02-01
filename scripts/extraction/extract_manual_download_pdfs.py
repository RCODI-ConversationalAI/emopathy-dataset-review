"""
수동 다운로드 PDF에서 ML 정보 추출 (Claude API)
대상: DOWNLOAD_JESSICA, DOWNLOAD_AUDREY, DOWNLOAD_CHRISTOPHER
"""

import os
import json
import csv
import time
import re
import fitz
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

# 설정
BASE_DIR = Path("/Volumes/ssd/01-ckj-postdoc/emopathy-dataset-review-local/EMOTION")
MATCHED_FILE = BASE_DIR / "PDF_MATCHED_FINAL.csv"
OUTPUT_FILE = BASE_DIR / "MANUAL_DOWNLOAD_EXTRACTED.csv"
CHECKPOINT_FILE = BASE_DIR / "manual_extraction_checkpoint.json"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

# 벤치마크 데이터셋 목록
BENCHMARK_DATASETS = [
    "IEMOCAP", "MELD", "CMU-MOSEI", "CMU-MOSI", "RAVDESS", "SAVEE", "EmoDB",
    "CREMA-D", "MSP-IMPROV", "MSP-PODCAST", "RECOLA", "SEMAINE", "AFEW",
    "FER2013", "AffectNet", "RAF-DB", "ExpW", "SFEW", "CK+", "Oulu-CASIA",
    "DEAP", "SEED", "DREAMER", "AMIGOS", "MAHNOB-HCI", "ASCERTAIN",
    "GoEmotions", "EmoContext", "ISEAR", "AffectiveText", "DENS",
    "EmotionLines", "EmoryNLP", "DailyDialog", "EmoWOZ", "EmoInt",
    "SemEval", "WASSA", "SST", "Twitter", "Reddit", "Yelp",
    "EmpatheticDialogues", "PersonaChat", "RECCON", "ECF",
    "Multimodal EmotionLines", "MEISD", "CH-SIMS", "M3ED",
    "MuSe", "OMG-Emotion", "AVEC", "MediaEval",
    "EMOTIC", "Aff-Wild", "Aff-Wild2", "SEWA"
]


def read_pdf_text(pdf_path: Path, max_chars: int = 50000) -> Optional[str]:
    """PDF에서 텍스트 추출"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        print(f"  PDF 읽기 오류: {e}")
        return None


def extract_with_claude(pdf_text: str, title: str) -> Dict:
    """Claude API로 정보 추출"""

    prompt = f"""You are analyzing a machine learning paper about emotion recognition.

Paper Title: {title}

Paper Content:
{pdf_text[:30000]}

Extract the following information in JSON format:

{{
    "dataset": ["list of benchmark datasets used (e.g., IEMOCAP, MELD, CMU-MOSEI)"],
    "model": {{
        "proposed_name": "name of proposed model/method",
        "backbone": ["pretrained models used (e.g., BERT, RoBERTa, wav2vec, ResNet)"],
        "architecture": ["architecture components (e.g., Transformer, LSTM, GRU, CNN, GNN)"],
        "fusion_method": "for multimodal: how modalities are fused",
        "training": "training approach (e.g., fine-tuning, from-scratch)"
    }},
    "modality": ["text", "audio", "video", "face", "physio"],
    "task": "main task (e.g., emotion recognition, sentiment analysis, ERC)",
    "emotion_labels": "emotion label scheme",
    "num_classes": "number of emotion classes as integer or null",
    "results": [
        {{
            "dataset": "dataset name",
            "acc": "accuracy as float or null",
            "wa": "weighted accuracy as float or null",
            "ua": "unweighted accuracy as float or null",
            "f1_weighted": "weighted F1 as float or null",
            "f1_macro": "macro F1 as float or null"
        }}
    ],
    "best_result": "one-line summary of best result"
}}

Important:
- Convert all percentages to decimals (68.5% -> 0.685)
- Return ONLY valid JSON, no explanations
- If information is not found, use null or empty array []
"""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(CLAUDE_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['content'][0]['text']

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return None

    except Exception as e:
        print(f"  Claude API 오류: {e}")
        return None


def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "results": []}


def save_checkpoint(checkpoint: Dict):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def save_results_csv(results: List[Dict]):
    if not results:
        return

    fieldnames = [
        "idx", "pdf_file", "folder", "title", "year", "datasets", "proposed_model",
        "backbone", "architecture", "fusion_method", "modality", "task",
        "emotion_labels", "num_classes", "best_result"
    ]

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            model = r.get("model", {})
            if isinstance(model, list):
                model = {"backbone": model}

            row = {
                "idx": r.get("idx", ""),
                "pdf_file": r.get("pdf_file", ""),
                "folder": r.get("folder", ""),
                "title": r.get("title", ""),
                "year": r.get("year", ""),
                "datasets": json.dumps(r.get("dataset", []), ensure_ascii=False),
                "proposed_model": model.get("proposed_name", ""),
                "backbone": json.dumps(model.get("backbone", []), ensure_ascii=False),
                "architecture": json.dumps(model.get("architecture", []), ensure_ascii=False),
                "fusion_method": model.get("fusion_method", ""),
                "modality": json.dumps(r.get("modality", []), ensure_ascii=False),
                "task": r.get("task", ""),
                "emotion_labels": r.get("emotion_labels", ""),
                "num_classes": r.get("num_classes", ""),
                "best_result": r.get("best_result", "")
            }
            writer.writerow(row)

    print(f"결과 저장: {OUTPUT_FILE} ({len(results)}건)")


def main():
    if not CLAUDE_API_KEY:
        print("CLAUDE_API_KEY가 설정되지 않았습니다.")
        return

    # 매칭된 PDF 목록 로드
    matched_df = pd.read_csv(MATCHED_FILE)
    print(f"처리할 PDF: {len(matched_df)}개")

    # 체크포인트 로드
    checkpoint = load_checkpoint()
    processed = set(checkpoint["processed"])
    results = checkpoint["results"]

    print(f"이미 처리됨: {len(processed)}개")
    print(f"남은 파일: {len(matched_df) - len(processed)}개")
    print("-" * 60)

    # 각 PDF 처리
    for i, row in matched_df.iterrows():
        pdf_name = row['new_name']

        if pdf_name in processed:
            continue

        folder_path = BASE_DIR / row['folder']
        pdf_path = folder_path / pdf_name

        title = row['title'] if pd.notna(row['title']) else pdf_name

        print(f"\n[{len(processed)+1}/{len(matched_df)}] {title[:50]}...")

        # PDF 텍스트 추출
        pdf_text = read_pdf_text(pdf_path)
        if not pdf_text:
            print("  PDF 읽기 실패")
            processed.add(pdf_name)
            continue

        print(f"  텍스트: {len(pdf_text):,}자")

        # Claude로 정보 추출
        extracted = extract_with_claude(pdf_text, title)

        if extracted:
            extracted["idx"] = row['idx']
            extracted["pdf_file"] = pdf_name
            extracted["folder"] = row['folder']
            extracted["title"] = title
            extracted["year"] = row['year']
            results.append(extracted)

            datasets = extracted.get("dataset", [])
            model = extracted.get("model", {})
            model_name = model.get("proposed_name", "N/A") if isinstance(model, dict) else "N/A"
            print(f"  추출 완료: {model_name}, datasets={datasets[:2]}")
        else:
            print("  추출 실패")

        processed.add(pdf_name)
        checkpoint["processed"] = list(processed)
        checkpoint["results"] = results

        # 20개마다 저장
        if len(processed) % 20 == 0:
            save_checkpoint(checkpoint)
            save_results_csv(results)
            print(f"\n  체크포인트 저장 ({len(processed)}개 완료)")

        time.sleep(1)

    # 최종 저장
    save_checkpoint(checkpoint)
    save_results_csv(results)

    print("\n" + "=" * 60)
    print("완료!")
    print(f"총 처리: {len(processed)}개")
    print(f"성공: {len(results)}개")


if __name__ == "__main__":
    main()

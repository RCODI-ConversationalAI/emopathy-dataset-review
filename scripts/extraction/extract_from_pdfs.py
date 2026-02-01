"""
PDF에서 ML Benchmark 정보 추출 (Claude API)

추출 필드 (8개):
1. dataset        - 사용한 벤치마크 (복수 가능)
2. model          - 사용한 모델들
3. modality       - text/audio/video/multimodal
4. task           - emotion recognition, sentiment, etc.
5. emotion_labels - 감정 레이블 체계
6. num_classes    - 분류 클래스 수
7. metrics        - 모든 성능 수치 (JSON)
8. best_result    - 최고 성능 요약
"""

import os
import json
import csv
import time
import re
import fitz  # PyMuPDF
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

# 설정
PDF_DIR = Path("ml_benchmark_pdfs")
OUTPUT_FILE = "ML_BENCHMARK_EXTRACTED.csv"
CHECKPOINT_FILE = "extraction_checkpoint.json"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

# 벤치마크 데이터셋 목록 (75개)
BENCHMARK_DATASETS = [
    # Emotion (64개)
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
    "EMOTIC", "Aff-Wild", "Aff-Wild2", "SEWA",
    "USC-IEMOCAP", "JL-Corpus", "VENEC", "EMO-DB",
    "YouTube", "Flickr",
    # Empathy (11개)
    "OMG-Empathy", "EmpatheticDialogues", "EmpatheticPersonas",
    "PEC", "EDOS", "EmotionPush", "EPITOME", "TalkLife",
    "ED", "Empathetic Conversations", "Reddit-Empathy"
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
        "proposed_name": "name of proposed model/method (e.g., EmoTrans, QAP, TelME)",
        "backbone": ["pretrained models used (e.g., BERT, RoBERTa, ALBERT, wav2vec, ResNet)"],
        "architecture": ["architecture components (e.g., Transformer, LSTM, GRU, CNN, GNN, Attention)"],
        "fusion_method": "for multimodal: how modalities are fused (e.g., early, late, cross-attention, tensor)",
        "training": "training approach (e.g., fine-tuning, from-scratch, frozen, adapter)"
    }},
    "modality": ["text", "audio", "video", "face", "physio"],
    "task": "main task (e.g., emotion recognition in conversation, multimodal emotion recognition, sentiment analysis)",
    "emotion_labels": "emotion label scheme (e.g., Ekman 6 basic, Plutchik 8, 4-class, 6-class, VAD continuous)",
    "num_classes": "number of emotion classes as integer (e.g., 4, 6, 7) or null",
    "results": [
        {{
            "dataset": "dataset name",
            "acc": "accuracy as float (e.g., 0.869) or null",
            "wa": "weighted accuracy as float or null",
            "ua": "unweighted accuracy as float or null",
            "f1_weighted": "weighted F1 as float or null",
            "f1_macro": "macro F1 as float or null",
            "f1_micro": "micro F1 as float or null",
            "precision": "precision as float or null",
            "recall": "recall as float or null",
            "bleu": "BLEU score as float or null (for generation tasks)",
            "other_metrics": {{"metric_name": value}}
        }}
    ],
    "best_result": "one-line summary: dataset, metric, value (e.g., 'IEMOCAP: 86.8% WA, 60.8% F1')"
}}

Important:
- Convert all percentages to decimals (68.5% -> 0.685)
- For model, be specific about what backbone/pretrained models are used
- For results, create one entry per dataset with all available metrics
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

        # JSON 추출
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            print(f"  JSON 파싱 실패")
            return None

    except json.JSONDecodeError as e:
        print(f"  JSON 디코드 오류: {e}")
        return None
    except Exception as e:
        print(f"  Claude API 오류: {e}")
        return None


def load_checkpoint() -> Dict:
    """체크포인트 로드"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "results": []}


def save_checkpoint(checkpoint: Dict):
    """체크포인트 저장"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def save_results_csv(results: List[Dict]):
    """결과 CSV 저장 - 데이터셋별로 행 분리"""
    if not results:
        return

    # 메인 CSV: 논문 정보 + 모델 상세
    main_fieldnames = [
        "pdf_file", "title", "datasets", "proposed_model", "backbone",
        "architecture", "fusion_method", "training", "modality", "task",
        "emotion_labels", "num_classes", "best_result"
    ]

    # 결과 CSV: 데이터셋별 메트릭 (정렬 가능)
    results_fieldnames = [
        "pdf_file", "title", "task", "dataset", "acc", "wa", "ua",
        "f1_weighted", "f1_macro", "f1_micro", "precision", "recall", "bleu", "other_metrics"
    ]

    # 메인 CSV 저장
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=main_fieldnames)
        writer.writeheader()

        for r in results:
            model = r.get("model", {})
            if isinstance(model, list):  # 이전 형식 호환
                model = {"backbone": model}

            row = {
                "pdf_file": r.get("pdf_file", ""),
                "title": r.get("title", ""),
                "datasets": json.dumps(r.get("dataset", []), ensure_ascii=False),
                "proposed_model": model.get("proposed_name", ""),
                "backbone": json.dumps(model.get("backbone", []), ensure_ascii=False),
                "architecture": json.dumps(model.get("architecture", []), ensure_ascii=False),
                "fusion_method": model.get("fusion_method", ""),
                "training": model.get("training", ""),
                "modality": json.dumps(r.get("modality", []), ensure_ascii=False),
                "task": r.get("task", ""),
                "emotion_labels": r.get("emotion_labels", ""),
                "num_classes": r.get("num_classes", ""),
                "best_result": r.get("best_result", "")
            }
            writer.writerow(row)

    print(f"\n메인 결과 저장: {OUTPUT_FILE} ({len(results)}건)")

    # 결과 CSV 저장 (데이터셋별 행)
    results_file = OUTPUT_FILE.replace(".csv", "_METRICS.csv")
    row_count = 0

    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results_fieldnames)
        writer.writeheader()

        for r in results:
            paper_results = r.get("results", [])
            if not paper_results:
                continue

            for res in paper_results:
                row = {
                    "pdf_file": r.get("pdf_file", ""),
                    "title": r.get("title", ""),
                    "task": r.get("task", ""),
                    "dataset": res.get("dataset", ""),
                    "acc": res.get("acc"),
                    "wa": res.get("wa"),
                    "ua": res.get("ua"),
                    "f1_weighted": res.get("f1_weighted"),
                    "f1_macro": res.get("f1_macro"),
                    "f1_micro": res.get("f1_micro"),
                    "precision": res.get("precision"),
                    "recall": res.get("recall"),
                    "bleu": res.get("bleu"),
                    "other_metrics": json.dumps(res.get("other_metrics", {}), ensure_ascii=False)
                }
                writer.writerow(row)
                row_count += 1

    print(f"메트릭 결과 저장: {results_file} ({row_count}건)")


def main():
    """메인 실행"""
    if not CLAUDE_API_KEY:
        print("CLAUDE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

    # PDF 파일 목록
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDF 파일: {len(pdf_files)}개")

    # 체크포인트 로드
    checkpoint = load_checkpoint()
    processed = set(checkpoint["processed"])
    results = checkpoint["results"]

    print(f"이미 처리됨: {len(processed)}개")
    print(f"남은 파일: {len(pdf_files) - len(processed)}개")
    print("-" * 60)

    # 각 PDF 처리
    for i, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.name

        # 이미 처리된 파일 스킵
        if pdf_name in processed:
            continue

        # 제목 추출 (파일명에서)
        title = pdf_name[5:-4].replace("_", " ")  # "0000_Title.pdf" -> "Title"

        print(f"\n[{len(processed)+1}/{len(pdf_files)}] {title[:60]}...")

        # PDF 텍스트 추출
        pdf_text = read_pdf_text(pdf_path)
        if not pdf_text:
            print("  ❌ PDF 읽기 실패")
            continue

        print(f"  📄 텍스트 추출: {len(pdf_text):,}자")

        # Claude로 정보 추출
        extracted = extract_with_claude(pdf_text, title)

        if extracted:
            extracted["pdf_file"] = pdf_name
            extracted["title"] = title
            results.append(extracted)

            # 결과 미리보기
            datasets = extracted.get("dataset", [])
            model = extracted.get("model", {})
            model_name = model.get("proposed_name", "N/A") if isinstance(model, dict) else "N/A"
            backbone = model.get("backbone", []) if isinstance(model, dict) else []
            best = extracted.get("best_result", "N/A")
            print(f"  ✅ 추출 완료")
            print(f"     Dataset: {', '.join(datasets[:3]) if datasets else 'N/A'}")
            print(f"     Model: {model_name} ({', '.join(backbone[:2]) if backbone else 'N/A'})")
            print(f"     Best: {best[:60] if best else 'N/A'}")
        else:
            print("  ⚠️ 추출 실패")

        # 체크포인트 업데이트
        processed.add(pdf_name)
        checkpoint["processed"] = list(processed)
        checkpoint["results"] = results

        # 10개마다 저장
        if len(processed) % 10 == 0:
            save_checkpoint(checkpoint)
            save_results_csv(results)
            print(f"\n  💾 체크포인트 저장 ({len(processed)}개 완료)")

        # Rate limiting (Claude API)
        time.sleep(1)

    # 최종 저장
    save_checkpoint(checkpoint)
    save_results_csv(results)

    print("\n" + "=" * 60)
    print("완료!")
    print(f"총 처리: {len(processed)}개")
    print(f"성공: {len(results)}개")
    print(f"결과 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

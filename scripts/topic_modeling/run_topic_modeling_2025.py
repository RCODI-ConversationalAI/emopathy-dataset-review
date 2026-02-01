"""
BERTopic Topic Modeling for ACL 2025 Emotion Papers
- 기존 모델 사용하여 transform
- 같은 제외 기준 적용

실행: python run_topic_modeling_2025.py
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("BERTopic - ACL 2025 Emotion Papers")
print("=" * 60)

# 1. 2025 데이터 로드
print("\n[1] ACL 2025 데이터 로드...")
df_2025 = pd.read_csv('ACL_2025_EMOTION_PAPERS.csv')
print(f"총 논문 수: {len(df_2025)}")

df_2025 = df_2025.dropna(subset=['abstract'])
df_2025 = df_2025[df_2025['abstract'].str.len() > 50]
df_2025 = df_2025.reset_index(drop=True)
print(f"유효한 abstract: {len(df_2025)}")

docs_2025 = df_2025['abstract'].tolist()

# 2. 기존 데이터와 합쳐서 BERTopic 학습
print("\n[2] 기존 데이터 로드...")
df_main = pd.read_csv('FINAL_EMOTION_PAPERS.csv')
df_main = df_main.dropna(subset=['abstract'])
df_main = df_main[df_main['abstract'].str.len() > 50]
df_main = df_main.reset_index(drop=True)
print(f"기존 데이터: {len(df_main)}편")

# 합치기
all_docs = df_main['abstract'].tolist() + docs_2025
print(f"전체: {len(all_docs)}편")

# 3. BERTopic 학습
print("\n[3] BERTopic 학습 중... (5-10분 소요)")
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))

topic_model = BERTopic(
    language="english",
    vectorizer_model=vectorizer,
    nr_topics=15,
    verbose=True
)

topics, probs = topic_model.fit_transform(all_docs)

# 4. 2025 데이터 토픽 추출 (마지막 부분)
n_main = len(df_main)
topics_2025 = topics[n_main:]
probs_2025 = probs[n_main:]

df_2025['topic'] = topics_2025
df_2025['topic_prob'] = probs_2025

# 5. 토픽 정보 출력
print("\n" + "=" * 60)
print("토픽 정보")
print("=" * 60)
topic_info = topic_model.get_topic_info()
print(topic_info[['Topic', 'Count', 'Name']].to_string())

# 6. 2025 토픽 분포
print("\n" + "=" * 60)
print("ACL 2025 토픽 분포")
print("=" * 60)
topic_counts_2025 = df_2025['topic'].value_counts().sort_index()
for t, c in topic_counts_2025.items():
    print(f"Topic {t:3d}: {c:3d}편")

# 7. 저장
df_2025.to_csv('ACL_2025_BERTOPIC_RESULTS.csv', index=False)
print("\n저장: ACL_2025_BERTOPIC_RESULTS.csv")

topic_info.to_csv('ACL_2025_BERTOPIC_TOPIC_INFO.csv', index=False)
print("저장: ACL_2025_BERTOPIC_TOPIC_INFO.csv")

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print("\n다음: ACL_2025_BERTOPIC_RESULTS.csv 다운로드 후")
print("같은 제외 토픽 적용하여 최종 데이터 생성")

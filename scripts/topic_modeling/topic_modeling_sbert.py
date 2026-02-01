"""
Sentence-BERT + KMeans Topic Modeling
Linux 서버에서 실행

사용법:
    python topic_modeling_sbert.py

필요 패키지:
    pip install sentence-transformers scikit-learn pandas numpy
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Sentence-BERT + KMeans Topic Modeling")
print("=" * 60)

# 1. 데이터 로드
df = pd.read_csv('FINAL_EMOTION_PAPERS.csv')
print(f"\n총 논문 수: {len(df)}")

df = df.dropna(subset=['abstract'])
df = df[df['abstract'].str.len() > 50]
df = df.reset_index(drop=True)
print(f"유효한 abstract: {len(df)}")

docs = df['abstract'].tolist()

# 2. Sentence-BERT 임베딩
print("\n[Step 1] Sentence-BERT 임베딩 중...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
print(f"임베딩 shape: {embeddings.shape}")

# 3. KMeans 클러스터링
print("\n[Step 2] KMeans 클러스터링...")
from sklearn.cluster import KMeans

n_clusters = 12
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)
print(f"클러스터 수: {n_clusters}")

# 4. 클러스터별 키워드 추출 (TF-IDF)
print("\n[Step 3] 클러스터별 키워드 추출...")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(docs)
feature_names = vectorizer.get_feature_names_out()

cluster_keywords = {}
for cluster_id in range(n_clusters):
    cluster_docs_idx = np.where(clusters == cluster_id)[0]
    if len(cluster_docs_idx) == 0:
        continue
    cluster_tfidf = tfidf_matrix[cluster_docs_idx].mean(axis=0).A1
    top_indices = cluster_tfidf.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    cluster_keywords[cluster_id] = top_words

# 5. 결과 출력
print("\n" + "=" * 60)
print("클러스터별 결과")
print("=" * 60)

cluster_info = []
for cluster_id in range(n_clusters):
    count = int(np.sum(clusters == cluster_id))
    pct = count / len(df) * 100
    keywords = cluster_keywords.get(cluster_id, [])

    cluster_info.append({
        'cluster': cluster_id,
        'count': count,
        'pct': round(pct, 1),
        'keywords': ', '.join(keywords[:8])
    })

    print(f"\nCluster {cluster_id} (n={count}, {pct:.1f}%):")
    print(f"  {', '.join(keywords[:6])}")

# 6. 저장
df['cluster'] = clusters
df.to_csv('EMOTION_TOPIC_RESULTS.csv', index=False)
print("\n저장: EMOTION_TOPIC_RESULTS.csv")

cluster_df = pd.DataFrame(cluster_info)
cluster_df.to_csv('EMOTION_TOPIC_INFO.csv', index=False)
print("저장: EMOTION_TOPIC_INFO.csv")

# 임베딩도 저장 (재사용 가능)
np.save('emotion_embeddings.npy', embeddings)
print("저장: emotion_embeddings.npy")

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)

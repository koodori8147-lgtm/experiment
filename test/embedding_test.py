from sentence_transformers import SentenceTransformer
MODEL_NAME = "nlpai-lab/KURE-v1"
model = SentenceTransformer(MODEL_NAME)

documents = [
    "진공 챔버 O-ring 손상으로 Pump Down 시간이 증가함",
    "전극 정렬 오차로 인한 Stacking 불량 발생",
    "PLC 통신 오류로 자동 물류 시스템 정지"
]

query = "진공 도달 시간이 길어지는 문제"

doc_vectors = model.encode(
    documents,
    normalize_embeddings=True
)

query_vector = model.encode(
    query,
    normalize_embeddings=True
)

scores = model.similarity(
    query_vector,
    doc_vectors
)[0]

print("벡터 차원:", len(query_vector))

for idx in scores.argsort(descending=True):
    print(
        documents[idx],
        "유사도:", round(scores[idx].item(), 4)
    )

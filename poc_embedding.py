"""
poc_embedding.py
Versi upgrade dari poc_tfidf_baseline.py — pakai semantic embedding multilingual.
Jalankan ini di laptopmu sendiri (butuh internet untuk download model, sekali saja,
lalu otomatis ke-cache di ~/.cache/huggingface).

Install dulu:
    pip install sentence-transformers scikit-learn numpy

Model 'paraphrase-multilingual-MiniLM-L12-v2' paham ~50 bahasa termasuk Indonesia,
ukurannya ringan (~470MB), cocok untuk deteksi parafrase lintas-bahasa.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model (sekali ini agak lama, selanjutnya cepat karena ke-cache)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def check_similarity(doc_a: str, doc_b: str, threshold: float = 0.6) -> dict:
    sentences_a = split_sentences(doc_a)
    sentences_b = split_sentences(doc_b)

    embeddings_a = model.encode(sentences_a)
    embeddings_b = model.encode(sentences_b)

    sim_matrix = cosine_similarity(embeddings_a, embeddings_b)

    matches = []
    for i, sent_a in enumerate(sentences_a):
        best_j = int(np.argmax(sim_matrix[i]))
        score = float(sim_matrix[i][best_j])
        if score >= threshold:
            matches.append({
                "source_sentence": sent_a,
                "matched_sentence": sentences_b[best_j],
                "similarity": round(score, 3),
            })

    overall = len(matches) / len(sentences_a) if sentences_a else 0
    return {
        "overall_similarity_percentage": round(overall * 100, 1),
        "matches": matches,
    }


if __name__ == "__main__":
    original = """
    Gastroesophageal reflux disease (GERD) adalah kondisi kronis yang disebabkan oleh
    naiknya asam lambung ke kerongkongan. Gejala utama meliputi nyeri ulu hati dan rasa
    panas di dada. Penanganan dini sangat penting untuk mencegah komplikasi jangka panjang.
    """

    copy_paste = """
    Gastroesophageal reflux disease (GERD) adalah kondisi kronis yang disebabkan oleh
    naiknya asam lambung ke kerongkongan. Gejala utama meliputi nyeri ulu hati dan rasa
    panas di dada.
    """

    paraphrased = """
    Penyakit refluks gastroesofageal merupakan gangguan kronis akibat naiknya cairan asam
    lambung menuju kerongkongan. Tanda-tanda khasnya antara lain rasa sakit di ulu hati
    serta sensasi terbakar pada dada. Penting untuk menanganinya sejak awal agar tidak
    menimbulkan komplikasi di masa mendatang.
    """

    # Parafrase "ekstrem" - sengaja tanpa kata kunci yang sama sama sekali,
    # ini ujian sebenarnya buat embedding vs TF-IDF
    paraphrased_no_overlap = """
    Naiknya cairan lambung yang bersifat asam menuju saluran pencernaan atas merupakan
    gangguan jangka panjang yang umum dialami. Penderitanya kerap merasakan sensasi
    terbakar di area dada bagian bawah serta nyeri di sekitar perut atas.
    """

    different_topic = """
    Algoritma pengurutan digunakan untuk menyusun data dalam struktur tertentu.
    Bubble sort adalah salah satu algoritma pengurutan paling sederhana namun kurang efisien.
    """

    print("\n=== Tes 1: Asli vs Copy-paste persis ===")
    print(check_similarity(original, copy_paste))

    print("\n=== Tes 2: Asli vs Parafrase (ada sedikit kata sama) ===")
    print(check_similarity(original, paraphrased))

    print("\n=== Tes 3: Asli vs Parafrase EKSTREM (tanpa kata sama sekali) ===")
    print(check_similarity(original, paraphrased_no_overlap))

    print("\n=== Tes 4: Asli vs Topik beda total ===")
    print(check_similarity(original, different_topic))

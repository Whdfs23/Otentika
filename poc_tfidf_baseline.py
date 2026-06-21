"""
poc_tfidf_baseline.py
Baseline pipeline test pakai TF-IDF (tidak butuh download model dari internet).
Tujuan: validasi ALUR-nya benar (split kalimat -> similarity matrix -> threshold -> skor),
bukan validasi akurasi semantik (itu tugas sentence-transformers di poc_embedding.py).
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def check_similarity(doc_a: str, doc_b: str, threshold: float = 0.25) -> dict:
    sentences_a = split_sentences(doc_a)
    sentences_b = split_sentences(doc_b)

    vectorizer = TfidfVectorizer().fit(sentences_a + sentences_b)
    vec_a = vectorizer.transform(sentences_a)
    vec_b = vectorizer.transform(sentences_b)

    sim_matrix = cosine_similarity(vec_a, vec_b)

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

    different_topic = """
    Algoritma pengurutan digunakan untuk menyusun data dalam struktur tertentu.
    Bubble sort adalah salah satu algoritma pengurutan paling sederhana namun kurang efisien.
    """

    print("=== Tes 1: Asli vs Copy-paste persis (HARUS skor tinggi) ===")
    print(check_similarity(original, copy_paste))
    print()

    print("=== Tes 2: Asli vs Parafrase (skor TF-IDF biasanya rendah-sedang, ini titik lemahnya) ===")
    print(check_similarity(original, paraphrased))
    print()

    print("=== Tes 3: Asli vs Topik beda total (HARUS skor mendekati 0) ===")
    print(check_similarity(original, different_topic))

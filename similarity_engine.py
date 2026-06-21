"""
similarity_engine.py
Versi production-ready dari poc_embedding.py. Ini calon modul inti yang nanti
dipanggil dari dalam FastAPI service.

Perubahan dari versi PoC sebelumnya:
1. Selalu laporkan skor SEMUA pasangan kalimat terbaik (bukan cuma yang lolos
   threshold) -> supaya UI nanti bisa highlight tiap kalimat dengan gradasi
   warna, bukan binary ada/tidak ada.
2. Pakai band severity (tinggi/sedang/rendah/abaikan), bukan satu threshold keras.
3. Ukur waktu encoding -> ini benchmark performa buat estimasi kapasitas worker
   nanti (penting untuk desain job queue di fase berikutnya).
4. Bentuk output sudah meniru struktur tabel `internal_matches` di Supabase.
"""

import re
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model...")
_start = time.time()
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(f"Model siap dalam {time.time() - _start:.2f} detik\n")


def classify_band(score: float) -> str:
    """
    Sudah dikalibrasi ulang pakai data nyata dari uji Tes 1-4:
    noise/tidak terkait selalu di bawah ~0.2, match asli (termasuk parafrase
    ekstrem) selalu di atas ~0.5. Threshold ini taruh di tengah jurang itu.
    """
    if score >= 0.75:
        return "tinggi"      # kemungkinan besar copy/parafrase dekat
    elif score >= 0.45:      # diturunkan dari 0.55 -> nangkep parafrase berat
        return "sedang"      # parafrase, layak ditinjau dosen
    elif score >= 0.25:      # diturunkan dari 0.35
        return "rendah"      # mirip tema, bisa kebetulan
    return "abaikan"         # terlalu lemah untuk jadi temuan


_ABBREVIATIONS = r"\b(dll|dst|dsb|yg|no|hal|cm|bln|thn|jl|prof|dr|tbk)\."

def split_sentences(text: str) -> list[str]:
    """
    Lindungi singkatan umum Indonesia dulu (supaya nggak ikut kepotong jadi
    kalimat baru), baru pisah kalimat berdasarkan tanda baca akhir.
    """
    protected = re.sub(_ABBREVIATIONS, lambda m: m.group(0).replace(".", "<DOT>"), text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", protected.strip())
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def check_similarity(doc_a: str, doc_b: str) -> dict:
    sentences_a = split_sentences(doc_a)
    sentences_b = split_sentences(doc_b)

    t0 = time.time()
    embeddings_a = model.encode(sentences_a)
    embeddings_b = model.encode(sentences_b)
    encode_time = time.time() - t0

    sim_matrix = cosine_similarity(embeddings_a, embeddings_b)

    results = []
    for i, sent_a in enumerate(sentences_a):
        best_j = int(np.argmax(sim_matrix[i]))
        score = float(sim_matrix[i][best_j])
        results.append({
            "source_sentence": sent_a,
            "matched_sentence": sentences_b[best_j],
            "similarity": round(score, 3),
            "band": classify_band(score),
        })

    overall = sum(1 for r in results if r["band"] in ("tinggi", "sedang")) / len(results) if results else 0
    band_counts = {"tinggi": 0, "sedang": 0, "rendah": 0, "abaikan": 0}
    for r in results:
        band_counts[r["band"]] += 1

    return {
        "overall_similarity_percentage": round(overall * 100, 1),
        "encode_time_seconds": round(encode_time, 4),
        "band_distribution": band_counts,
        "results": results,
    }


def check_similarity_multi(source_text: str, candidate_docs: list[dict]) -> dict:
    """
    Bandingkan satu dokumen sumber terhadap BANYAK dokumen pembanding sekaligus.
    candidate_docs: list berisi {"document_id": ..., "text": ...}

    Untuk tiap kalimat sumber, dicari match TERBAIK di seluruh kumpulan kalimat
    pembanding lintas dokumen -- bukan dihitung per-dokumen terpisah, supaya
    satu kalimat sumber tidak dihitung berkali-kali kalau kebetulan mirip ke
    beberapa dokumen pembanding sekaligus.
    """
    sentences_a = split_sentences(source_text)

    pool_sentences, pool_doc_ids = [], []
    for doc in candidate_docs:
        for s in split_sentences(doc["text"]):
            pool_sentences.append(s)
            pool_doc_ids.append(doc["document_id"])

    if not sentences_a or not pool_sentences:
        return {
            "overall_similarity_percentage": 0,
            "encode_time_seconds": 0,
            "band_distribution": {"tinggi": 0, "sedang": 0, "rendah": 0, "abaikan": 0},
            "results": [],
        }

    t0 = time.time()
    emb_a = model.encode(sentences_a, show_progress_bar=False)
    emb_pool = model.encode(pool_sentences, show_progress_bar=False)
    encode_time = time.time() - t0

    sim_matrix = cosine_similarity(emb_a, emb_pool)

    results = []
    for i, sent_a in enumerate(sentences_a):
        best_j = int(np.argmax(sim_matrix[i]))
        score = float(sim_matrix[i][best_j])
        results.append({
            "source_sentence": sent_a,
            "matched_sentence": pool_sentences[best_j],
            "matched_document_id": pool_doc_ids[best_j],
            "similarity": round(score, 3),
            "band": classify_band(score),
        })

    overall = sum(1 for r in results if r["band"] in ("tinggi", "sedang")) / len(results)
    band_counts = {"tinggi": 0, "sedang": 0, "rendah": 0, "abaikan": 0}
    for r in results:
        band_counts[r["band"]] += 1

    return {
        "overall_similarity_percentage": round(overall * 100, 1),
        "encode_time_seconds": round(encode_time, 4),
        "band_distribution": band_counts,
        "results": results,
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

    paraphrased_no_overlap = """
    Naiknya cairan lambung yang bersifat asam menuju saluran pencernaan atas merupakan
    gangguan jangka panjang yang umum dialami. Penderitanya kerap merasakan sensasi
    terbakar di area dada bagian bawah serta nyeri di sekitar perut atas.
    """

    different_topic = """
    Algoritma pengurutan digunakan untuk menyusun data dalam struktur tertentu.
    Bubble sort adalah salah satu algoritma pengurutan paling sederhana namun kurang efisien.
    """

    for label, text_b in [
        ("Tes 1: Copy-paste persis", copy_paste),
        ("Tes 2: Parafrase ringan", paraphrased),
        ("Tes 3: Parafrase ekstrem (0 kata sama)", paraphrased_no_overlap),
        ("Tes 4: Topik beda total", different_topic),
    ]:
        print(f"=== {label} ===")
        print(json.dumps(check_similarity(original, text_b), indent=2, ensure_ascii=False))
        print()
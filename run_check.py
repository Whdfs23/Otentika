"""
run_check.py
Pipeline lengkap: ekstrak teks dari 2 dokumen (PDF/DOCX, boleh campur format),
lalu jalankan pengecekan kemiripan semantik di antara keduanya.

Ini "vertical slice" terakhir sebelum pindah ke FastAPI + Supabase -- tujuannya
membuktikan SELURUH alur jalan di atas dokumen asli yang panjang, bukan cuma
kalimat contoh pendek seperti uji-uji sebelumnya.

Pakai:
    python run_check.py "dokumen_a.pdf" "dokumen_b.docx"

Butuh document_parser.py dan similarity_engine.py ada di folder yang sama.
"""

import sys
import time

from document_parser import extract_clean_text
from similarity_engine import check_similarity


def main():
    if len(sys.argv) < 3:
        print('Pakai: python run_check.py "dokumen_a.pdf-atau-docx" "dokumen_b.pdf-atau-docx"')
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    print(f"Mengekstrak teks dari A: {path_a}")
    t0 = time.time()
    text_a = extract_clean_text(path_a)
    extract_time_a = time.time() - t0

    print(f"Mengekstrak teks dari B: {path_b}")
    t0 = time.time()
    text_b = extract_clean_text(path_b)
    extract_time_b = time.time() - t0

    print(f"\nWaktu ekstraksi -> A: {extract_time_a:.2f}s, B: {extract_time_b:.2f}s")
    print(f"Panjang teks bersih -> A: {len(text_a)} karakter, B: {len(text_b)} karakter")

    result = check_similarity(text_a, text_b)

    print(f"\n=== HASIL PENGECEKAN ===")
    print(f"Skor kemiripan keseluruhan: {result['overall_similarity_percentage']}%")
    print(f"Waktu encoding (similarity engine): {result['encode_time_seconds']}s")
    print(f"Total waktu end-to-end: {extract_time_a + extract_time_b + result['encode_time_seconds']:.2f}s")
    print(f"Distribusi band: {result['band_distribution']}")

    top_matches = sorted(
        [r for r in result["results"] if r["band"] != "abaikan"],
        key=lambda r: r["similarity"],
        reverse=True,
    )[:8]

    print(f"\n--- Top {len(top_matches)} pasangan kalimat paling mirip ---")
    for m in top_matches:
        print(f"\n[{m['band'].upper()}] skor={m['similarity']}")
        print(f"  A: {m['source_sentence'][:150]}")
        print(f"  B: {m['matched_sentence'][:150]}")


if __name__ == "__main__":
    main()

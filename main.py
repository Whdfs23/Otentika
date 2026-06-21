"""
main.py
FastAPI service Turnitin Mandiri -- versi pertama (Tahap A).

SENGAJA belum pakai autentikasi JWT (itu baru masuk di Tahap C, setelah ada
frontend Next.js yang bisa login). Tujuan versi ini cuma satu: buktikan
FastAPI bisa nyambung dengan BENAR ke Supabase Storage & Database.

Install:
    pip install fastapi "uvicorn[standard]" python-multipart supabase python-dotenv

Siapkan file .env di folder yang sama (lihat .env.example):
    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_SERVICE_ROLE_KEY=xxxxx
    TEST_OWNER_ID=xxxxx-uuid-user-percobaan

Jalankan:
    uvicorn main:app --reload

Lalu buka http://localhost:8000/docs -- FastAPI otomatis kasih UI interaktif
buat coba upload file, nggak perlu Postman/curl.
"""

import os
import hashlib
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

from document_parser import extract_clean_text
from similarity_engine import check_similarity_multi

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
TEST_OWNER_ID = os.environ["TEST_OWNER_ID"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="Turnitin Mandiri API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nanti dipersempit ke domain Vercel pas production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Terima file -> ekstrak teks -> simpan ke Storage + Database.
    BELUM ada perbandingan similarity -- itu endpoint /check terpisah,
    nyusul di Tahap B setelah ini terbukti jalan.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(400, "Format harus .pdf atau .docx")

    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_text = extract_clean_text(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not raw_text.strip():
        raise HTTPException(422, "Gagal mengekstrak teks dari file ini")

    storage_path = f"{TEST_OWNER_ID}/{file_hash}{ext}"

    try:
        supabase.storage.from_("documents").upload(
            storage_path, content, {"content-type": file.content_type}
        )
    except Exception as e:
        raise HTTPException(
            409, f"Gagal upload ke storage (kemungkinan file ini sudah pernah diupload): {e}"
        )

    result = supabase.table("documents").insert({
        "owner_id": TEST_OWNER_ID,
        "title": file.filename,
        "file_path": storage_path,
        "file_format": ext.replace(".", ""),
        "file_hash": file_hash,
        "raw_text": raw_text,
        "status": "uploaded",
    }).execute()

    document = result.data[0]

    return {
        "document_id": document["id"],
        "title": document["title"],
        "extracted_characters": len(raw_text),
        "preview": raw_text[:300],
    }


@app.post("/check/{document_id}")
async def check_document(document_id: str):
    """
    Jalankan pengecekan similarity untuk dokumen yang sudah diupload,
    dibandingkan dengan semua dokumen LAIN milik user yang sama (korpus
    internal). Pengecekan ke Crossref/arXiv/web -- nyusul Tahap G.
    """
    doc_result = supabase.table("documents").select("*").eq("id", document_id).execute()
    if not doc_result.data:
        raise HTTPException(404, "Dokumen tidak ditemukan")
    document = doc_result.data[0]

    if not document["raw_text"]:
        raise HTTPException(422, "Dokumen ini belum punya teks terekstrak")

    others_result = (
        supabase.table("documents")
        .select("id, title, raw_text")
        .eq("owner_id", document["owner_id"])
        .neq("id", document_id)
        .not_.is_("raw_text", "null")
        .execute()
    )
    candidate_docs = [
        {"document_id": d["id"], "text": d["raw_text"]} for d in others_result.data
    ]

    check_insert = supabase.table("checks").insert({
        "document_id": document_id,
        "status": "processing",
    }).execute()
    check_id = check_insert.data[0]["id"]

    if not candidate_docs:
        # Bukan error -- korpus pembandingnya memang masih kosong
        empty_bands = {"tinggi": 0, "sedang": 0, "rendah": 0, "abaikan": 0}
        supabase.table("checks").update({
            "status": "done",
            "overall_similarity_percentage": 0,
            "band_distribution": empty_bands,
        }).eq("id", check_id).execute()

        return {
            "check_id": check_id,
            "overall_similarity_percentage": 0,
            "message": "Belum ada dokumen lain di korpus untuk dibandingkan",
        }

    result = check_similarity_multi(document["raw_text"], candidate_docs)

    matches_to_insert = [
        {
            "check_id": check_id,
            "matched_document_id": r["matched_document_id"],
            "source_sentence": r["source_sentence"],
            "matched_sentence": r["matched_sentence"],
            "similarity": r["similarity"],
            "band": r["band"],
        }
        for r in result["results"]
        if r["band"] in ("tinggi", "sedang")
    ]

    if matches_to_insert:
        supabase.table("internal_matches").insert(matches_to_insert).execute()

    supabase.table("checks").update({
        "status": "done",
        "overall_similarity_percentage": result["overall_similarity_percentage"],
        "band_distribution": result["band_distribution"],
    }).eq("id", check_id).execute()

    return {
        "check_id": check_id,
        "overall_similarity_percentage": result["overall_similarity_percentage"],
        "band_distribution": result["band_distribution"],
        "total_matches_found": len(matches_to_insert),
        "encode_time_seconds": result["encode_time_seconds"],
    }
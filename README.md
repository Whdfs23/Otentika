---
title: Turnitin Mandiri API
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Turnitin Mandiri API

Backend FastAPI untuk sistem deteksi plagiarisme mandiri (semantic embedding multilingual + Supabase).

Endpoint:
- `GET /health` - cek status server
- `POST /upload` - upload dokumen (.pdf / .docx)
- `POST /check/{document_id}` - jalankan pengecekan similarity

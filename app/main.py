import json
import logging
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel

from app.config import OPENAI_API_KEY, UPLOAD_DIR
from app.ingest.loader import chunk_text, load_text_from_file
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.chat import chat_engine
from app.rag.memory import memory_store
from app.rag.retriever import hybrid_retriever
from app.storage.faiss_store import vector_store

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".csv"}

app = FastAPI(title="Hybrid RAG Chatbot", version="1.0.0")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SuggestRequest(BaseModel):
    session_id: str | None = None
    mode: str = "initial"
    last_answer: str = ""


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status():
    return {
        "ready": vector_store.is_ready,
        "documents": vector_store.get_documents(),
        "chunk_count": len(vector_store.records),
    }


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Add it to your .env file and restart the server.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    processed = []
    total_chunks = 0

    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Use txt, pdf, or csv.",
                )

            safe_name = Path(upload.filename or "upload").name
            dest = UPLOAD_DIR / safe_name
            async with aiofiles.open(dest, "wb") as out:
                content = await upload.read()
                await out.write(content)

            try:
                text = load_text_from_file(dest)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                processed.append({"filename": safe_name, "chunks": 0})
                continue

            chunk_pairs = [(chunk, safe_name) for chunk in chunks]
            added = vector_store.add_documents(chunk_pairs)
            total_chunks += added
            processed.append({"filename": safe_name, "chunks": added})

        hybrid_retriever._ensure_bm25()
        session = memory_store.get_or_create(None)
        suggestions = chat_engine.generate_suggestions("initial", session)

        return {
            "processed": processed,
            "total_chunks": total_chunks,
            "documents": vector_store.get_documents(),
            "suggestions": suggestions,
        }
    except HTTPException:
        raise
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error while indexing documents: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {exc}",
        ) from exc


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session = memory_store.get_or_create(payload.session_id)
    result = chat_engine.chat(message, session)
    followups = chat_engine.generate_suggestions(
        "followup",
        session,
        result["answer"],
    )
    result["suggestions"] = followups
    return result


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session = memory_store.get_or_create(payload.session_id)

    def event_stream():
        try:
            for event in chat_engine.stream_chat(message, session):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/suggestions")
async def suggestions(payload: SuggestRequest):
    session = memory_store.get_or_create(payload.session_id)
    items = chat_engine.generate_suggestions(
        payload.mode,
        session,
        payload.last_answer,
    )
    return {"suggestions": items}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    from openai import OpenAI
    from app.config import OPENAI_API_KEY, WHISPER_MODEL

    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    temp_path = UPLOAD_DIR / f"voice_{audio.filename or 'recording.webm'}"
    async with aiofiles.open(temp_path, "wb") as out:
        await out.write(audio_bytes)

    try:
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
            )
    finally:
        temp_path.unlink(missing_ok=True)

    return {"text": transcript.text}


@app.delete("/api/documents")
async def remove_document(filename: str):
    safe_name = Path(filename).name
    if safe_name not in vector_store.get_sources():
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_removed = vector_store.remove_source(safe_name)
    file_path = UPLOAD_DIR / safe_name
    if file_path.exists():
        file_path.unlink()

    hybrid_retriever._ensure_bm25()
    session = memory_store.get_or_create(None)
    suggestions = chat_engine.generate_suggestions("initial", session)

    return {
        "removed": safe_name,
        "chunks_removed": chunks_removed,
        "ready": vector_store.is_ready,
        "documents": vector_store.get_documents(),
        "chunk_count": len(vector_store.records),
        "suggestions": suggestions,
    }


@app.delete("/api/clear")
async def clear_all():
    vector_store.clear()
    hybrid_retriever._ensure_bm25()
    for path in UPLOAD_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    memory_store._sessions.clear()
    return {"status": "cleared"}


@app.on_event("startup")
async def startup():
    if not OPENAI_API_KEY:
        logger.warning(
            "OPENAI_API_KEY is not set. Uploads, chat, and transcription will fail until it is configured."
        )
    hybrid_retriever._ensure_bm25()

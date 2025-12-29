import time
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from rag_pipeline import get_rag_pipeline, RAGPipeline
from initialize_pinecone import initialize_pinecone
import os

app = FastAPI(title="Portfolio RAG Agent API")

# Configure CORS - Allow both local development and Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_origin_regex=r".*",  # Allow ALL origins (Permissive for Portfolio)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stats tracking
stats = {
    "total_queries": 0,
    "avg_response_time": 0.0,
    "last_reindex_time": None,
    "index_status": "Ready (Pinecone)"
}

class ChatRequest(BaseModel):
    message: str
    stream: bool = True

@app.get("/")
async def root():
    return {
        "message": "Portfolio RAG Backend is running",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }

@app.on_event("startup")
async def startup_event():
    # Pinecone is cloud-based, so we just ensure we can connect
    try:
        get_rag_pipeline()
        stats["index_status"] = "Ready (Pinecone)"
        stats["last_reindex_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        stats["index_status"] = f"Error: {str(e)}"

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "index_status": stats["index_status"],
        "timestamp": time.time()
    }

@app.post("/api/reindex")
async def reindex():
    success = initialize_pinecone()
    if success:
        # Reload pipeline
        import rag_pipeline as rp
        rp.rag_pipeline = rp.RAGPipeline()
        stats["index_status"] = "Ready (Pinecone)"
        stats["last_reindex_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return {"message": "Pinecone index rebuilt successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to rebuild Pinecone index")

@app.get("/api/stats")
async def get_stats():
    return stats

@app.post("/api/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    pipeline = get_rag_pipeline()
    
    stats["total_queries"] += 1
    
    if request.stream:
        async def event_generator():
            for chunk in pipeline.get_response_stream(request.message):
                yield chunk
            
            # Update average response time (simplified)
            duration = time.time() - start_time
            stats["avg_response_time"] = (stats["avg_response_time"] + duration) / 2

        return StreamingResponse(event_generator(), media_type="text/plain")
    else:
        response = pipeline.get_response(request.message)
        duration = time.time() - start_time
        stats["avg_response_time"] = (stats["avg_response_time"] + duration) / 2
        return {"response": response}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline = get_rag_pipeline()
    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()
            stats["total_queries"] += 1
            
            for chunk in pipeline.get_response_stream(data):
                await websocket.send_text(chunk)
            
            await websocket.send_text("[DONE]")
            
            duration = time.time() - start_time
            stats["avg_response_time"] = (stats["avg_response_time"] + duration) / 2
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

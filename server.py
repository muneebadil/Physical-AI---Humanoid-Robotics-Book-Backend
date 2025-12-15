from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import agent  # <-- Your existing agent.py (NO CHANGES NEEDED)

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running!"}


# ---------------------------------------------------
# 🔥 MAIN API: Chat Endpoint (Frontend Will Use This)
# ---------------------------------------------------
@app.post("/ask")
async def ask_question(data: dict):
    question = data.get("question", "")

    if not question.strip():
        return {"error": "Question cannot be empty"}

    answer = await agent.run_agent(question)
    return {"answer": answer}



# ---------------------------------------------------
# INGESTION API — CALLS YOUR UNTOUCHED main.py
# ---------------------------------------------------
import main  # <-- your ingestion file

@app.post("/ingest")
def ingest_book():
    main.ingest_book()
    return {"status": "Ingestion completed"}


# ---------------------------------------------------
# LOCAL RUN
# ---------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

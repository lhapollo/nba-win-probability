from fastapi import FastAPI

app = FastAPI(title="NBA Win Probability")

@app.get("/health")
async def health():
    return {"status": "ok"}
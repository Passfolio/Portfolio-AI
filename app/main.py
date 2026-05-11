from fastapi import FastAPI

app = FastAPI(title="Passfolio AI Server")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

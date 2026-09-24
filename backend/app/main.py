from fastapi import FastAPI

app = FastAPI(title="CSHSM Backend", description="Counter-Speech Generation for Hate Content on Social Media")

@app.get("/health")
def health_check():
    return {"status": "ok"}

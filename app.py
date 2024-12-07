from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel  # type: ignore
from transformers import pipeline  # type: ignore

app = FastAPI()

# Load the summarization pipeline with the fine-tuned model
summarizer = pipeline(
    "summarization", model="devesh1011/bart-large-cnn-finetuned-news-summarizer"
)


class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 100  # Default maximum length


@app.post("/summarize/")
async def summarize(request: SummarizationRequest):
    # Validate input text
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Generate summary using the pipeline
        summary = summarizer(request.text, max_length=request.max_length)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        # Handle any exceptions during summarization
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {"status": "API is running"}


import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

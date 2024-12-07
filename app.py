from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model = PeftModel.from_pretrained(
    base_model, "devesh1011/bart-large-cnn-finetuned-news-summarizer"
)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 100  # Default maximum length


@app.post("/summarize/")
async def summarize(request: SummarizationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        summary = summarizer(request.text, max_length=request.max_length)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {"status": "API is running"}

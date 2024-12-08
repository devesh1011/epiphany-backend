from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline(
    "summarization", model="devesh1011/bart-large-cnn-finetuned-news-summarizer"
)

# Define the FastAPI application
app = FastAPI()


# Define the request body model
class SummarizationRequest(BaseModel):
    text: str


# Define the response model
class SummarizationResponse(BaseModel):
    summary: str


# Define the POST request endpoint
@app.post("/summarize/", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    # Check if the request body is valid
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Summarize the text using the pipeline
    summary = summarizer(request.text, max_length=130, min_length=30, do_sample=False)

    # Return the summary as a response
    return {"summary": summary[0]["summary_text"]}


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

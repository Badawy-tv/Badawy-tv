from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# CPU-friendly models (replace with bigger models if server allows)
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.get("/")
def root():
    return {"message": "Badawy-tv AI backend running"}

@app.post("/generate")
def generate_text(prompt: str):
    result = generator(prompt, max_length=150)
    return {"response": result[0]['generated_text']}

@app.post("/summarize")
def summarize_text(text: str):
    result = summarizer(text, max_length=100, min_length=30)
    return {"summary": result[0]['summary_text']}

@app.post("/qa")
def answer_question(question: str, context: str):
    result = qa(question=question, context=context)
    return {"answer": result['answer']}

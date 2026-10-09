# Medical Assistant

A university project by **Bhukya Naresh**. This Flask web application answers health-related questions using retrieval-augmented generation (RAG): relevant text is retrieved from a medical reference document stored in Pinecone, then an OpenAI model prepares a concise response.

> Educational use only. The application is not a substitute for professional medical advice, diagnosis, or treatment.

## Features

- Web chat interface built with Flask
- PDF ingestion and text chunking
- Sentence-transformer embeddings and Pinecone vector search
- Context-aware answers from an OpenAI chat model

## Technology

Python, Flask, LangChain, Pinecone, OpenAI, and Sentence Transformers.

## Run locally

1. Clone your GitHub repository and enter the project directory.

   ```bash
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```

2. Create and activate a Python 3.10 environment.

   ```bash
   conda create -n medical-chatbot python=3.10 -y
   conda activate medical-chatbot
   ```

3. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root. Do not commit this file.

   ```ini
   PINECONE_API_KEY=your-pinecone-api-key
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_INDEX_NAME=medical-chatbot
   OPENAI_MODEL=gpt-4o-mini
   ```

5. Build the vector index from the PDF files in `data/`.

   ```bash
   python store_index.py
   ```

6. Start the application.

   ```bash
   python app.py
   ```

Open `http://127.0.0.1:8080` in a browser.

## GitHub submission checklist

- Use a repository name such as `medical-chatbot-university-project`.
- Add a concise project description and this README on GitHub.
- Keep `.env`, API keys, and generated secrets out of version control.
- Commit the source code, `requirements.txt`, and relevant documentation.

## Project structure

```text
app.py             Flask application
store_index.py     Creates and populates the Pinecone index
data/              Medical reference documents
src/               Retrieval, embedding, and prompt helpers
templates/         Chat interface
static/            Stylesheet
```

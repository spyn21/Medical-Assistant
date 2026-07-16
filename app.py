import os
from functools import lru_cache

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")


@lru_cache(maxsize=1)
def get_rag_chain():
    """Create the retrieval chain after the application has been configured."""
    missing = [key for key in ("PINECONE_API_KEY", "OPENAI_API_KEY") if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    embeddings = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(
        ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")), prompt
    )
    return create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    message = request.form.get("msg", "").strip()
    if not message:
        return jsonify(error="Please enter a question."), 400
    try:
        response = get_rag_chain().invoke({"input": message})
    except Exception as exc:
        app.logger.exception("Unable to process chat request")
        return jsonify(error=f"The chatbot is not ready: {exc}"), 503
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=os.getenv("FLASK_DEBUG") == "1")
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings, format_docs
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# ------------------ Flask ------------------
app = Flask(__name__)
load_dotenv()

# ------------------ API Keys ------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in .env")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ------------------ Embeddings & VectorDB ------------------
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="rag",
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
)

# ------------------ Routes ------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")

    if not msg.strip():
        return "Please type something."

    response = rag_chain.invoke(msg)

    # Extract content safely (Gemini returns object)
    try:
        output = response.content
    except:
        output = str(response)

    return output


# ------------------ Main ------------------
# if __name__ == "__main__":
#     app.run(debug=True)              # for local


if __name__ == "__main__":                          #for deployment
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)


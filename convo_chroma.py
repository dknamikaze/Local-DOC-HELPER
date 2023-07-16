import os
from typing import Any

from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
model_path = os.environ.get('MODEL_PATH')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

from constants import CHROMA_SETTINGS

def run_llm(query: str) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    chat = LlamaCpp(model_path=model_path, n_ctx=2048, n_threads=8)

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa({"query": query})

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=chat,
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True
    # )

    # return qa({"question":query, "chat_history":chat_history})


if __name__ == "__main__":
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = run_llm(query=query)
        end = time.time()
        print(res['result'])
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
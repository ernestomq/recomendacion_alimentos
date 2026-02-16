import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from usdanutrient import USDANutrientLoader

def main():
    index_dir = "faiss_index_BAAI"
    if os.path.exists(index_dir) and os.path.exists(os.path.join(index_dir, "index.faiss")):
            print("El índice ya existe. No hay que recrearlo.")
    else:
            loader = USDANutrientLoader(
                    directory_path=r"json",
                    encoding="utf-8",
                    include_calculated=True
            )
            docs = loader.load()
            embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(index_dir)
            print("Índice creado.")

if __name__ == "__main__":
    main()


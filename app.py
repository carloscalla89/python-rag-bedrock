import os

import boto3
import tempfile
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document


from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize the Bedrock client with explicit credentials
boto3_bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    model_id="amazon.titan-embed-text-v2:0"
)

# prompt = ChatPromptTemplate.from_messages(
#    [
#        ("system", "Your task is to answer the question using as few words as possible. "),
#        ("user", "Question: {query}")
#    ]
# )

# parser = StrOutputParser()

# chain = prompt | llm | parser

urls = [
    "https://adictosaltrabajo.com/2023/08/21/devops-eso-es-cosa-del-pasado-conoce-mlops/",
    "https://adictosaltrabajo.com/2023/07/27/nltk-python/",
    "https://adictosaltrabajo.com/2023/05/06/diagramas-de-arquitectura-con-c4-model/",
    "https://adictosaltrabajo.com/2023/05/10/como-ia-puede-mejorar-eficiencia-programador/",
    "https://adictosaltrabajo.com/2023/05/12/structurizr-para-generar-diagramas-de-arquitectura-con-c4-model/"
]

documents = []


for url in urls:
    try:
        # Hacer la solicitud GET
        response = requests.get(url, timeout=10)

        # Verificar si la respuesta es exitosa (código 200)
        if response.status_code == 200:
            # Preprocesar HTML con BeautifulSoup (para evitar errores de lxml)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extraer solo el texto limpio de la página
            cleaned_text = soup.get_text()

            # Guardar el contenido en un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(cleaned_text.encode("utf-8"))
                temp_file_path = temp_file.name  # Obtener la ruta del archivo temporal

            # Cargar el texto en UnstructuredStringLoader
            loader = UnstructuredFileLoader(temp_file_path)
            docs = loader.load()

            # Agregar el documento procesado a la lista
            documents.extend(docs)

            # print(f"✅ Procesado: {url}")

        else:
            print(f"❌ Error al acceder a {url}: Código {response.status_code}")

    except Exception as e:
        print(f"⚠️ Error al procesar {url}: {e}")


# loader = UnstructuredURLLoader(urls=urls)
# docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=5
)

doc_splits = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="python_docs",
    embedding=bedrock_embeddings,
)
retriever = vectorstore.as_retriever()


def retrieve(question: str) -> List[str]:
    documents = retriever.invoke(question)
    return [doc.page_content for doc in documents]


def generate_answer(question: str, context: List[str]) -> str:

    llm = ChatBedrock(
        model_id="amazon.nova-lite-v1:0",
        region="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    context_text = "\n".join(context)
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    response = llm.invoke(prompt)
    return response


def run_pipeline(question: str) -> str:
    print("Retrieving documents...")
    documents = retrieve(question)

    print("Generating answer...")
    answer = generate_answer(question, documents)
    return answer


if __name__ == "__main__":
    question = "Me interesa saber de mlops"
    answer = run_pipeline(question)
    print(f"Question: {question}\nAnswer: {answer}")

# response = chain.invoke("What is the capital of Peru ?")
# print(response)

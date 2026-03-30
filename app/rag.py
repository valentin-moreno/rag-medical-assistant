# Motor RAG (Retrieval-Augmented Generation) medico
# Carga documentos de medicamentos y enfermedades, los indexa en FAISS
# y responde preguntas clinicas usando LM Studio como LLM (100% local)
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

# ── Configuracion ─────────────────────────────────────────────────────────────

DIRECTORIO_DOCS   = "data/documentos"
DIRECTORIO_FAISS  = "data/vectorstore"
MODELO_EMBEDDINGS = "BAAI/bge-small-en-v1.5"

# LM Studio corre localmente en el puerto 1234
# Usa la misma interfaz de OpenAI — solo cambiamos la URL base
LM_STUDIO_URL     = "http://localhost:1234/v1"
MODELO_LLM        = "meta-llama-3-8b-instruct"

CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 80
TOP_K             = 6


# ── Prompt del sistema ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """Eres MedRAG, un asistente médico clínico especializado en farmacología y medicina clínica.

Tu objetivo es responder preguntas médicas utilizando únicamente el contexto proporcionado, de forma clara, estructurada y clínicamente útil.

ENFOQUE GENERAL:
- Prioriza respuestas útiles, organizadas y directamente relacionadas con la pregunta.
- Puedes integrar y razonar con el contexto (no tiene que estar textual).
- Ignora cualquier información del contexto que no sea relevante para la pregunta.

USO DEL CONTEXTO:
1. Usa exclusivamente la información del contexto.
2. Si el contexto es suficiente → responde con claridad.
3. Si es parcial → responde con lo disponible e indica brevemente qué falta.
4. Si es irrelevante → responde:
   "Esta información no está en la base de conocimiento. Consulta con un médico."

INFERENCIA CONTROLADA:
- Puedes relacionar información clínica del contexto.
- NO inventes datos específicos (dosis, fármacos, contraindicaciones) si no aparecen.

SEGURIDAD CLÍNICA:
- No des diagnósticos definitivos.
- No sugieras cambiar tratamientos actuales.
- Ante síntomas graves indica: "Acude a urgencias inmediatamente."

FORMATO DE RESPUESTA (OBLIGATORIO):

## Respuesta clínica
(explicación clara y directa)

## Tratamiento / Manejo (si aplica)
(lista de opciones SOLO si están en el contexto)

## Consideraciones clínicas
(puntos relevantes, limitaciones o contexto adicional)

- No incluyas información irrelevante.
- No repitas ideas.
- Sé preciso y profesional.

CIERRE (OBLIGATORIO):
Debes SIEMPRE terminar la respuesta exactamente con esta línea final (sin modificarla):

⚠️ Información educativa. Consulta siempre con un profesional de la salud.
---

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

prompt = PromptTemplate(
    template        = PROMPT_TEMPLATE,
    input_variables = ["context", "question"],
)


# ── Funciones principales ──

def cargar_documentos(directorio: str) -> list:
    """
    Carga todos los archivos TXT y PDF del directorio de documentos.
    Usa TextLoader para TXT y PyPDFLoader para PDF.
    """
    print(f"Cargando documentos desde: {directorio}")
    documentos = []

    # Cargamos los TXT
    loader_txt = DirectoryLoader(
        directorio,
        glob          = "**/*.txt",
        loader_cls    = TextLoader,
        loader_kwargs = {"encoding": "utf-8"},
    )
    docs_txt = loader_txt.load()
    documentos.extend(docs_txt)
    print(f"  -> {len(docs_txt)} archivos TXT cargados")

    # Cargamos los PDF
    loader_pdf = DirectoryLoader(
        directorio,
        glob       = "**/*.pdf",
        loader_cls = PyPDFLoader,
    )
    docs_pdf = loader_pdf.load()
    documentos.extend(docs_pdf)
    print(f"  -> {len(docs_pdf)} archivos PDF cargados")

    print(f"  -> {len(documentos)} documentos totales")
    return documentos


def dividir_en_chunks(documentos: list) -> list:
    """Divide los documentos en fragmentos para la busqueda semantica."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documentos)
    print(f"  -> {len(chunks)} fragmentos generados")
    return chunks


def construir_o_cargar_vectorstore(chunks: list = None) -> FAISS:
    """
    Si ya existe el indice FAISS en disco lo carga.
    Si no existe lo construye y lo guarda para proximas veces.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name    = MODELO_EMBEDDINGS,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    if os.path.exists(DIRECTORIO_FAISS) and os.listdir(DIRECTORIO_FAISS):
        print("Cargando indice FAISS existente...")
        vectorstore = FAISS.load_local(
            DIRECTORIO_FAISS,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("  -> Indice cargado correctamente")
    else:
        print("Construyendo indice FAISS (primera vez, puede tardar un momento)...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(DIRECTORIO_FAISS, exist_ok=True)
        vectorstore.save_local(DIRECTORIO_FAISS)
        print("  -> Indice construido y guardado en disco")

    return vectorstore


def construir_cadena_rag(vectorstore):

    llm = ChatOpenAI(
        base_url    = LM_STUDIO_URL,
        api_key     = "lm-studio",
        model       = MODELO_LLM,
        temperature = 0.3,
        max_tokens  = 1200,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 8,
            "lambda_mult": 0.8,
        },
    )


    cadena = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}, 
    )

    return cadena

def inicializar_rag() -> RetrievalQA:
    """Inicializa el sistema RAG completo."""
    print("\nInicializando Medical RAG con LM Studio...")
    print("=" * 50)
    documentos  = cargar_documentos(DIRECTORIO_DOCS)
    chunks      = dividir_en_chunks(documentos)
    vectorstore = construir_o_cargar_vectorstore(chunks)
    cadena      = construir_cadena_rag(vectorstore)
    print("=" * 50)
    print("Sistema RAG listo para consultas\n")
    return cadena


def consultar(cadena: RetrievalQA, pregunta: str) -> dict:
    """Realiza una consulta al sistema RAG y retorna respuesta con fuentes."""
    resultado = cadena.invoke({"query": pregunta})
    fuentes = list(set(
        doc.metadata.get("source", "desconocida")
        for doc in resultado["source_documents"]
    ))
    return {
        "pregunta": pregunta,
        "respuesta": resultado["result"],
        "fuentes": fuentes,
    }

# Tests del Medical RAG
# Verifican que el sistema carga, indexa y responde correctamente
# Ejecutar con: pytest tests/ -v
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag import cargar_documentos, dividir_en_chunks, construir_o_cargar_vectorstore


def test_carga_documentos():
    """Verifica que se cargan los documentos medicos correctamente."""
    documentos = cargar_documentos("data/documentos")
    assert len(documentos) >= 2   # al menos medicamentos.txt y enfermedades.txt
    assert all(len(doc.page_content) > 0 for doc in documentos)


def test_division_en_chunks():
    """Verifica que los chunks tienen el tamano correcto."""
    documentos = cargar_documentos("data/documentos")
    chunks = dividir_en_chunks(documentos)
    assert len(chunks) > len(documentos)   # debe haber mas chunks que documentos
    assert all(len(chunk.page_content) > 0 for chunk in chunks)


def test_vectorstore_se_construye():
    """Verifica que el vectorstore FAISS se construye correctamente."""
    documentos = cargar_documentos("data/documentos")
    chunks     = dividir_en_chunks(documentos)
    vectorstore = construir_o_cargar_vectorstore(chunks)
    assert vectorstore is not None


def test_busqueda_semantica():
    """Verifica que la busqueda semantica devuelve resultados relevantes."""
    documentos  = cargar_documentos("data/documentos")
    chunks      = dividir_en_chunks(documentos)
    vectorstore = construir_o_cargar_vectorstore(chunks)

    # Busqueda sobre hipertension — debe traer fragmentos relevantes
    resultados = vectorstore.similarity_search("hipertension arterial tratamiento", k=3)
    assert len(resultados) == 3
    textos = " ".join([r.page_content for r in resultados]).lower()
    assert any(palabra in textos for palabra in ["hipertension", "presion", "antihipertensivo"])


def test_busqueda_medicamentos():
    """Verifica que encuentra informacion sobre medicamentos especificos."""
    documentos  = cargar_documentos("data/documentos")
    chunks      = dividir_en_chunks(documentos)
    vectorstore = construir_o_cargar_vectorstore(chunks)

    resultados = vectorstore.similarity_search("metformina diabetes contraindicaciones", k=3)
    textos = " ".join([r.page_content for r in resultados]).lower()
    assert any(palabra in textos for palabra in ["metformina", "diabetes", "glucosa"])

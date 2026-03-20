# 🏥 Clinical RAG Assistant

Sistema de **Retrieval-Augmented Generation (RAG)** aplicado al dominio médico-clínico. Responde preguntas clínicas en lenguaje natural sobre medicamentos, enfermedades, diagnósticos y tratamientos, usando documentos médicos reales como fuente de conocimiento y un LLM local como motor de generación.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-yellow)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-blue)
![LLaMA](https://img.shields.io/badge/LLM-LLaMA3%208B%20local-orange)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow?logo=huggingface)

---

## 🎯 ¿Qué hace?

Funciona como un **vademécum médico inteligente** que puedes consultar en lenguaje natural:

```
"¿Qué medicamentos se usan para la hipertensión arterial?"
"¿Cuáles son los síntomas de la sepsis y cómo se trata?"
"¿Qué contraindicaciones tiene la metformina?"
"¿Qué antibiótico usar en neumonía grave con ingreso a UCI?"
"¿Cómo se calcula el score CHA2DS2-VASc?"
"¿Cuál es el manejo inicial del shock séptico?"
```

---

## 🧠 Arquitectura RAG

```
  Pregunta del usuario
         │
         ▼
┌─────────────────────────┐
│  Embeddings             │  sentence-transformers/all-MiniLM-L6-v2
│  (HuggingFace local)    │  Convierte la pregunta en vector
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  FAISS Vector Store     │  Busqueda semantica de similitud
│  14,000+ fragmentos     │  Recupera los 4 mas relevantes
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Contexto recuperado    │  Fragmentos de documentos medicos reales
│  de documentos reales   │  PDFs de farmacologia, vademecum, guias
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  LLaMA 3 8B             │  Corre localmente via LM Studio
│  (LM Studio local)      │  Sin API externa, sin costos, sin limites
└──────────┬──────────────┘
           │
           ▼
  Respuesta medica estructurada
  + Fuentes citadas
  + Advertencia de seguridad
```

---

## 📁 Estructura del proyecto

```
clinical-rag-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py          ← API REST con FastAPI
│   └── rag.py           ← Motor RAG completo
├── data/
│   ├── documentos/      ← PDFs y TXTs medicos (no incluidos en repo)
│   │   ├── *.pdf        ← Vademecum, farmacologia, guias clinicas
│   │   └── *.txt        ← Documentos base de medicamentos y enfermedades
│   └── vectorstore/     ← Indice FAISS (se genera automaticamente)
├── tests/
│   └── test_rag.py      ← Tests unitarios
├── consultar.py         ← Modo interactivo desde consola
├── ejemplos.py          ← Consultas de demo organizadas por categoria
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Instalacion y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/valentin-moreno/clinical-rag-assistant.git
cd clinical-rag-assistant
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configurar LM Studio

1. Descarga **LM Studio** desde [lmstudio.ai](https://lmstudio.ai)
2. Descarga el modelo **Meta Llama 3 8B Instruct Q4**
3. Ve a la pestaña **Developer**
4. Carga el modelo y haz click en **Start Server**
5. El servidor queda en `http://localhost:1234`

### 4. Agregar documentos médicos

Coloca tus archivos `.txt` o `.pdf` en `data/documentos/`. El sistema acepta cualquier documento médico — vademécum, guías clínicas, protocolos, farmacología.

### 5. Correr la API

```bash
python -m uvicorn app.main:app --reload
```

> ⚠️ La primera vez tarda varios minutos indexando los documentos. Las siguientes veces carga el índice instantáneamente desde disco.

Abre: **http://localhost:8000/docs**

### 6. Modo interactivo sin API

```bash
python consultar.py
```

### 7. Ver consultas de ejemplo

```bash
python ejemplos.py
```

### 8. Correr los tests

```bash
pytest tests/ -v
```

---

## 📡 Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Estado de la API |
| `GET` | `/health` | Estado del RAG (listo / inicializando) |
| `POST` | `/consultar` | Consulta médica en lenguaje natural |

### Ejemplo de consulta

```bash
curl -X POST "http://localhost:8000/consultar" \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"¿Qué medicamentos se usan para la hipertensión arterial?\"}"
```

### Respuesta

```json
{
  "pregunta": "¿Qué medicamentos se usan para la hipertensión arterial?",
  "respuesta": "## Medicamentos para la Hipertensión Arterial\n\n### IECAs\nEnalapril, lisinopril — inhiben la conversión de angiotensina I a II...\n\n### ARA II\nLosartan, valsartan — bloquean el receptor AT1...\n\n⚠️ Información educativa. Consulta siempre con un profesional de la salud.",
  "fuentes": [
    "data\\documentos\\vademecum1.pdf",
    "data\\documentos\\enfermedades.txt"
  ]
}
```

---

## 🔒 Seguridad clínica

El sistema tiene restricciones estrictas para evitar respuestas peligrosas:

- **No inventa datos** — si la información no está en los documentos, lo dice explícitamente
- **No emite diagnósticos definitivos** — informa sobre enfermedades y criterios clínicos
- **No sugiere cambios de medicación** — nunca recomienda suspender o ajustar dosis
- **Ante emergencias** — ante síntomas graves siempre indica acudir a urgencias
- **Advierte interacciones** — menciona interacciones relevantes entre medicamentos
- **Siempre incluye advertencia** — cada respuesta termina recordando consultar al médico

---

## 💡 Decisiones técnicas

- **LM Studio sobre APIs en la nube**: Sin costos por consulta, sin límites de rate, sin dependencia de internet. El modelo corre completamente local.
- **FAISS sobre ChromaDB**: Sin servidor, instalación simple, suficiente para decenas de miles de fragmentos.
- **all-MiniLM-L6-v2**: Modelo de embeddings liviano (80MB), rápido en CPU y excelente calidad para búsqueda semántica en textos médicos.
- **LLaMA 3 8B Q4**: Cuantización Q4 reduce el modelo a 4.37GB manteniendo calidad suficiente para respuestas clínicas detalladas.
- **Temperatura 0.1**: Respuestas deterministas y precisas — en medicina no queremos creatividad, queremos exactitud.
- **Persistencia del índice FAISS**: El índice se construye una vez y se reutiliza — no se recalculan embeddings en cada arranque.
- **Soporte TXT y PDF**: El loader carga ambos formatos automáticamente desde la carpeta de documentos.

---

## 🔄 Agregar nuevos documentos

1. Coloca el archivo `.txt` o `.pdf` en `data/documentos/`
2. Elimina la carpeta `data/vectorstore/` para forzar reconstrucción del índice
3. Reinicia la API — el índice se reconstruye automáticamente

```powershell
rmdir /s /q data\vectorstore
python -m uvicorn app.main:app --reload
```

---

## 🛠️ Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| **LangChain** | Orquestación del pipeline RAG |
| **FAISS** | Base de datos vectorial para búsqueda semántica |
| **HuggingFace Transformers** | Modelo de embeddings (all-MiniLM-L6-v2) |
| **LM Studio** | Servidor local del LLM (LLaMA 3 8B Instruct) |
| **FastAPI** | API REST |
| **PyPDF** | Lectura de documentos PDF |
| **python-dotenv** | Gestión de variables de entorno |
| **pytest** | Tests unitarios |

---

## ⚠️ Aviso importante

Este sistema es exclusivamente para fines **educativos y de demostración**. No reemplaza el criterio clínico de un profesional de la salud titulado. Siempre consulta con un médico antes de tomar decisiones clínicas.

---

## 👤 Autor

**Valentín Moreno Vásquez** — Ingeniero Biomédico, Especialista en IA

---

## 📄 Licencia

MIT — libre para uso personal y comercial.

---

# 🏥 Clinical RAG Assistant (English)

**Retrieval-Augmented Generation (RAG)** system for the clinical domain. Answers medical questions in natural language using real medical documents (PDFs, clinical guides, pharmacology texts) as knowledge base and a local LLM via LM Studio — no external APIs, no costs, no limits.

## Quick Start

```bash
pip install -r requirements.txt
# Start LM Studio with Meta Llama 3 8B on port 1234
python -m uvicorn app.main:app --reload
# Open http://localhost:8000/docs
```

## Key Features

- Indexes 14,000+ fragments from real medical PDFs
- Runs 100% locally — no API keys, no internet dependency
- Strict clinical safety rules — no hallucinated doses or diagnoses
- Supports TXT and PDF documents
- Cites sources for every response

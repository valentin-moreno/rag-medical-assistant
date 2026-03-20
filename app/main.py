# API REST del Medical RAG
# Expone el sistema RAG como servicio web con FastAPI
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.rag import inicializar_rag, consultar


# ── Estado global del RAG ─────────────────────────────────────────────────────
# El RAG se inicializa una sola vez al arrancar la API
# y se reutiliza en todas las consultas (no se recarga en cada peticion)

estado = {"cadena": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa el RAG al arrancar y lo libera al cerrar."""
    print("Arrancando Medical RAG API...")
    estado["cadena"] = inicializar_rag()
    yield
    print("Cerrando Medical RAG API...")


# ── Configuracion de la app ───────────────────────────────────────────────────

app = FastAPI(
    title      = "Medical RAG API",
    description= "Vademecum medico inteligente — consulta sobre medicamentos y enfermedades usando RAG con LM Studio.",
    version    = "1.0.0",
    lifespan   = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)


# ── Modelos de datos ──────────────────────────────────────────────────────────

class Consulta(BaseModel):
    pregunta: str

    class Config:
        json_schema_extra = {
            "example": {
                "pregunta": "¿Qué medicamentos se usan para la hipertensión arterial?"
            }
        }


class Respuesta(BaseModel):
    pregunta: str
    respuesta: str
    fuentes: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def raiz():
    return {
        "mensaje": "Medical RAG API activa",
        "descripcion": "Vademecum medico inteligente con RAG + LM Studio. Consulta sobre medicamentos, enfermedades, contraindicaciones, sintomas y tratamientos.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    rag_listo = estado["cadena"] is not None
    return {
        "status": "ok" if rag_listo else "inicializando",
        "rag_listo": rag_listo,
    }


@app.post("/consultar", response_model=Respuesta)
def consultar_endpoint(consulta: Consulta):
    """
    Recibe una pregunta medica y retorna una respuesta basada en el vademecum.

    Ejemplos de preguntas:
    - ¿Qué medicamentos sirven para la hipertensión?
    - ¿Cuáles son los síntomas de la sepsis?
    - ¿Qué contraindicaciones tiene la metformina?
    - ¿Qué antibiótico usar en neumonía grave?
    - ¿Cómo se trata la fibrilación auricular?
    """
    if not estado["cadena"]:
        raise HTTPException(status_code=503, detail="RAG aun inicializandose, intenta en unos segundos.")

    if not consulta.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    try:
        resultado = consultar(estado["cadena"], consulta.pregunta)
        return Respuesta(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

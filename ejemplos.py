# Consultas de ejemplo — Medical RAG
# Corre con: python ejemplos.py
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

from app.rag import inicializar_rag, consultar

# ── Consultas organizadas por categoria ───────────────────────────────────────

CONSULTAS = {

    "Medicamentos — Indicaciones": [
        "¿Para qué sirve la metformina y en qué pacientes se usa?",
        "¿Qué medicamentos se usan para tratar la hipertensión arterial?",
        "¿Cuándo se indica la insulina glargina?",
        "¿Qué anticoagulante es mejor para un paciente con fibrilación auricular?",
        "¿Para qué se usa la furosemida y cuál es su mecanismo?",
    ],

    "Medicamentos — Contraindicaciones y efectos adversos": [
        "¿Qué contraindicaciones tiene la metformina?",
        "¿Qué efectos adversos tiene la atorvastatina?",
        "¿En qué pacientes está contraindicada la warfarina?",
        "¿Qué pasa si combino un IECA con un ARA II?",
        "¿Por qué el enalapril produce tos?",
    ],

    "Enfermedades — Diagnostico y sintomas": [
        "¿Cuáles son los síntomas de la diabetes mellitus tipo 2?",
        "¿Cómo se diagnostica la hipertensión arterial?",
        "¿Qué es el score CURB-65 y para qué se usa?",
        "¿Cuáles son los criterios diagnósticos de sepsis según Sepsis-3?",
        "¿Qué significa FAST en el contexto del ACV?",
    ],

    "Enfermedades — Tratamiento": [
        "¿Cómo se trata la insuficiencia cardíaca?",
        "¿Cuál es el tratamiento de emergencia del infarto con elevación del ST?",
        "¿Qué antibióticos se usan en la neumonía grave que requiere UCI?",
        "¿Qué medicamentos se usan en el shock séptico?",
        "¿Cómo se trata la fibrilación auricular con respuesta ventricular rápida?",
    ],

    "Consultas clinicas complejas": [
        "Paciente diabético con hipertensión y proteinuria, ¿qué antihipertensivo es de elección?",
        "¿Qué diferencia hay entre STEMI y NSTEMI en el infarto de miocardio?",
        "¿Cuándo se indica diálisis en la insuficiencia renal aguda?",
        "¿Qué es el score CHA2DS2-VASc y cómo se calcula?",
        "Paciente con EPOC que también tiene insuficiencia cardíaca, ¿se puede usar metoprolol?",
    ],
}


if __name__ == "__main__":
    # Inicializamos el RAG una sola vez
    cadena = inicializar_rag()

    print("\n" + "=" * 60)
    print("DEMOSTRACION — MEDICAL RAG")
    print("=" * 60)

    for categoria, preguntas in CONSULTAS.items():
        print(f"\n\n{'='*60}")
        print(f"CATEGORIA: {categoria}")
        print(f"{'='*60}")

        # Mostramos solo la primera pregunta de cada categoria para la demo
        pregunta = preguntas[0]
        print(f"\nPREGUNTA: {pregunta}")
        print("-" * 40)

        resultado = consultar(cadena, pregunta)
        print(f"RESPUESTA:\n{resultado['respuesta']}")
        print(f"\nFUENTES: {', '.join(resultado['fuentes'])}")

    print("\n\n" + "=" * 60)
    print("Para hacer tus propias consultas corre: python consultar.py")
    print("=" * 60)

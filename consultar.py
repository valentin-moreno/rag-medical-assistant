# Prueba el RAG directamente desde consola
# Util para probar sin levantar la API
# Corre con: python consultar.py
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

from app.rag import inicializar_rag, consultar

# Preguntas de prueba — variadas para demostrar capacidades
PREGUNTAS_DEMO = [
    "¿Qué medicamentos se usan para tratar la hipertensión arterial?",
    "¿Cuáles son los síntomas de la diabetes mellitus tipo 2?",
    "¿Qué contraindicaciones tiene la metformina?",
    "¿Cómo se trata la insuficiencia cardíaca?",
    "¿Qué antibióticos se usan en la neumonía grave?",
    "¿Cuál es el tratamiento del shock séptico?",
    "¿Qué es el score CHA2DS2-VASc y para qué sirve?",
    "¿Qué efectos adversos tiene la furosemida?",
]

if __name__ == "__main__":
    # Inicializamos el sistema RAG
    cadena = inicializar_rag()

    print("\nMODO INTERACTIVO — escribe 'demo' para ver ejemplos o 'salir' para terminar\n")

    while True:
        pregunta = input("Tu pregunta: ").strip()

        if not pregunta:
            continue

        if pregunta.lower() == "salir":
            print("Hasta luego!")
            break

        if pregunta.lower() == "demo":
            # Corremos todas las preguntas de demo
            for p in PREGUNTAS_DEMO:
                print(f"\n{'='*60}")
                print(f"PREGUNTA: {p}")
                print(f"{'='*60}")
                resultado = consultar(cadena, p)
                print(f"\nRESPUESTA:\n{resultado['respuesta']}")
                print(f"\nFUENTES: {', '.join(resultado['fuentes'])}")
            continue

        # Consulta normal
        print("\nConsultando...")
        resultado = consultar(cadena, pregunta)
        print(f"\nRESPUESTA:\n{resultado['respuesta']}")
        print(f"\nFUENTES: {', '.join(resultado['fuentes'])}\n")

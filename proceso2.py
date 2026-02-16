import PyPDF2
import pytesseract
import os
from docx import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.schema import HumanMessage, SystemMessage
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from pathlib import Path
from PIL import Image
from langchain_classic.retrievers import MultiQueryRetriever

##################
### VARIABLES ####
##################
# Configuración del modelo LLM
repord_llm = "llama-3.3-70b-versatile"
os.environ["GROQ_API_KEY"] ="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"
# Configuración de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Rutas y archivos
directorio_archivos = 'archivos'  # Directorio donde están los archivos
archivo_entrada = 'revision_medica.pdf'  # Nombre del archivo a procesar
recuperacion = "faiss_index_BAAI"

# Configuración del modelo de embeddings
name_modelo = "BAAI/bge-m3"
kwargs_model = {"device": "cpu"}
kwargs_encode = {"normalize_embeddings": True}
kwargs_search = {"k": 10}

# Directorio de salida
output_dir = Path('planes_generados')

##################
#### PROMPTS #####
##################
few_shot = """
Ejemplo de entrada:
"Paciente varón de 45 años, con hipertensión y diabetes. Alergia a marisco. Medicación: enalapril."
Ejemplo de salida:
{"edad": 45, "sexo": "hombre", "condiciones": ["hipertensión", "diabetes"], "alergias": ["marisco"], "medicacion": ["enalapril"], "imc": null, "presion_arterial": null, "glucemia": null}
"""

system_prompt_extraccion = """
Eres un asistente que extrae información médica estructurada en JSON.
SOLO extrae datos que aparezcan EXPLÍCITAMENTE en el texto.
NO inventes nada.
{format_instructions}

## EJEMPLO:
{few_shot}
"""

template_plan_dietetico = """
Eres un asistente de planificación dietética especializado en adaptar menús a condiciones de salud específicas. 
Tus respuestas deben estar redactadas en español.

**Reglas estrictas:**  
- No inventes datos clínicos (usa solo la información del paciente).  
- Si falta algún dato, haz una suposición razonable y menciónala.  
- Usa los alimentos proporcionados; si faltan, complementa con opciones comunes apropiadas.  
- RESPUESTA SIEMPRE EN ESPAÑOL.  

A continuación, recibirás:
- **INFORMACIÓN DEL PACIENTE**: datos extraídos de su historia clínica (edad, condiciones médicas, alergias, etc.).
- **INFORMACIÓN SOBRE ALIMENTOS**: archivos con descripciones de alimentos recuperados mediante un sistema de búsqueda.

---
**INFORMACIÓN DEL PACIENTE:**  
{paciente_info}

**ALIMENTOS DISPONIBLES:**  
{contenido}

---

Tu tarea es informar de los ALIMENTOS RECOMENDADOS y de los ALIMENTOS A MODERAR O EVITAR para este paciente, basándote en su perfil clínico y en la información de los alimentos.:


### ALIMENTOS RECOMENDADOS  
...  
### ALIMENTOS A MODERAR O EVITAR 
...
"""

################################################################################
#### PROMPTS #####
################################################################################

response_schemas = [
    ResponseSchema(name="edad", description="Edad del paciente en años (solo número). Si no aparece, null."),
    ResponseSchema(name="sexo", description="Sexo: 'hombre' o 'mujer'. Si no aparece, null."),
    ResponseSchema(name="condiciones", description="Lista de condiciones médicas diagnosticadas (ej. diabetes, hipertensión). Vacío [] si no hay."),
    ResponseSchema(name="alergias", description="Lista de alergias alimentarias o intolerancias. Vacío [] si no hay."),
    ResponseSchema(name="medicacion", description="Lista de medicamentos actuales. Vacío [] si no hay."),
    ResponseSchema(name="imc", description="Valor numérico del IMC. Si no aparece, null."),
    ResponseSchema(name="presion_arterial", description="Presión arterial en formato 'sistólica/diastólica' (ej. '120/80'). Si no aparece, null."),
    ResponseSchema(name="glucemia", description="Glucosa en mg/dL (solo número). Si no aparece, null."),
]

def leer_archivo(ruta_archivo):
    path = Path(ruta_archivo)
    ext = path.suffix.lower()
    text = ''
    
    try:
        if ext == '.txt':
            text = path.read_text(encoding='utf-8', errors='ignore')
            
        elif ext == '.pdf':
            with path.open('rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ''
                    
        elif ext == '.docx':
            doc = Document(path)
            text = '\n'.join(p.text for p in doc.paragraphs)
            
        elif ext in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}:
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            
        else:
            text = f'Formato no soportado: {ext}'
            
    except Exception as e:
        text = f'ERROR al leer archivo: {e}'
    
    return text


def extraer_datos_paciente(texto, llm):
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    system_content = system_prompt_extraccion.format(
        format_instructions=format_instructions,
        few_shot=few_shot
    )
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Texto del paciente:\n\n{texto}")
    ]
    
    respuesta = llm.invoke(messages)
    
    if isinstance(respuesta, str):
        content = respuesta
    else:
        content = respuesta.content
    
    datos = output_parser.parse(content)
    
    return datos


def inicializar_embeddings(modelo_name, model_kwargs, encode_kwargs):
    embeddings = HuggingFaceEmbeddings(
        model_name=modelo_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


def cargar_vectorstore(ruta_index, embeddings):
    faiss_db = FAISS.load_local(
        ruta_index, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    return faiss_db


def crear_retriever(vectorstore, llm, k=10):
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        llm=llm,
        include_original=True
    )
    return retriever


def generar_plan_dietetico(datos_paciente, contenido_alimentos, llm, template):
    prompt = ChatPromptTemplate.from_template(template)

    paciente_info = "\n".join([f"- {k}: {v}" for k, v in datos_paciente.items()])

    mensaje = prompt.format(
        paciente_info=paciente_info,
        contenido=contenido_alimentos
    )

    respuesta = llm.invoke(mensaje)
    
    if isinstance(respuesta, str):
        plan = respuesta
    else:
        plan = respuesta.content
    
    return plan


def guardar_plan(plan, nombre_archivo, directorio_salida):
    directorio_salida.mkdir(exist_ok=True)
    archivo_salida = directorio_salida / f"plan_{nombre_archivo}.txt"
    archivo_salida.write_text(plan, encoding='utf-8')
    print(f"Plan guardado en: {archivo_salida}")

def main(llama, url_llm, tesseract_path, directorio_archivos, archivo_entrada, 
         recuperacion, name_modelo, kwargs_model, kwargs_encode, kwargs_search, output_dir):
    """
    Función principal que ejecuta el flujo completo:
    1. Lee el archivo del paciente
    2. Extrae datos estructurados
    3. Carga el vector store de alimentos
    4. Genera el plan dietético
    5. Guarda el resultado
    """

    ruta_completa = Path(directorio_archivos) / archivo_entrada
    filename = ruta_completa.stem
    
    print(f"Procesando archivo: {ruta_completa}")

    print("Leyendo archivo...")
    texto_paciente = leer_archivo(ruta_completa)
    
    if texto_paciente.startswith('ERROR'):
        print(texto_paciente)
        return

    print("Inicializando modelo de lenguaje...")
    llama_llm = ChatGroq(
        model=repord_llm,
        temperature=0.2,
        max_tokens=3000,
    )

    print("Extrayendo datos del paciente...")
    datos_paciente = extraer_datos_paciente(texto_paciente, llama_llm)
    print(f"Datos extraídos: {datos_paciente}")

    print("Inicializando embeddings...")
    embeddings = inicializar_embeddings(name_modelo, kwargs_model, kwargs_encode)

    print("Cargando base de datos de alimentos...")
    vectorstore = cargar_vectorstore(recuperacion, embeddings)

    print("Configurando recuperador de información...")
    retriever = crear_retriever(vectorstore, llama_llm, k=kwargs_search["k"])

    print("Buscando alimentos relevantes...")
    query_alimentos = f"Alimentos apropiados para: {', '.join(datos_paciente.get('condiciones', []))}"
    docs_alimentos = retriever.invoke(query_alimentos)
    contenido_alimentos = "\n\n".join([doc.page_content for doc in docs_alimentos])

    print("Generando Respuesta...")
    plan_dietetico = generar_plan_dietetico(
        datos_paciente, 
        contenido_alimentos, 
        llama_llm, 
        template_plan_dietetico
    )

    print("Guardando plan...")
    guardar_plan(plan_dietetico, filename, output_dir)
    
    print("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    ##################
    ### VARIABLES ####
    ##################
    # Configuración del modelo LLM
    llama = "llama3.2:latest"
    url_llm = "http://localhost:11434"

    # Configuración de Tesseract OCR
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Rutas y archivos
    directorio_archivos = 'archivos'
    archivo_entrada = 'revision_medica.pdf'
    recuperacion = "faiss_index_BAAI"

    # Configuración del modelo de embeddings
    name_modelo = "BAAI/bge-m3"
    kwargs_model = {"device": "cpu"}
    kwargs_encode = {"normalize_embeddings": True}
    kwargs_search = {"k": 10}

    # Directorio de salida
    output_dir = Path('planes_generados')
    
    # Ejecutar función principal
    main(
        llama=llama,
        url_llm=url_llm,
        tesseract_path=tesseract_path,
        directorio_archivos=directorio_archivos,
        archivo_entrada=archivo_entrada,
        recuperacion=recuperacion,
        name_modelo=name_modelo,
        kwargs_model=kwargs_model,
        kwargs_encode=kwargs_encode,
        kwargs_search=kwargs_search,
        output_dir=output_dir
    )
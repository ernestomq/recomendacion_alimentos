# recomendacion_alimentos
Este proyecto es un sistema inteligente basado en IA que ayuda a nutricionistas a seleccionar alimentos óptimos para crear dietas personalizadas, agilizando el proceso de planificación de nutrientes tras el informe de una revisión médica del paciente.

Para realizarlo se han utilizado modelos de embeddings y modelos de lenguaje para analizar la información nutricional y generar recomendaciones alimentarias personalizadas, permitiendo a los nutricionistas crear recomendaciones más variadas y precisas, ya que en ocasiones puede olvidarse algún alimento adecuado. De esta manera se generan dietas más eficientes y mejor adaptadas.

Para poder realizar este proyecto se ha indicado las librerías que se han utilizado junto con la versión que se ha utilizado en el archivo **requirements.py**

Este proyecto se divide en 3 archivos principales:
* El primer archivo es **usdanutrient.py**, en este archivo se procesa la información del archivo JSON con los datos nutricionales para poder extraer y normalizar la información, preparar los datos y generar los embeddings.
* El segundo archivo es **embeddings.py**, en este archivo se crean los embeddings una vez se llama al módulo **usdanutrient.py**, para poder extraer la información y crear los embeddings de la manera más óptima sobre el archivo descargado en [https://fdc.nal.usda.gov/download-datasets].
* El tercer archivo es **proceso2.py**, este archivo contiene el flujo que se realiza a partir de los LLM. Primeramente se realiza la lectura de uno de los distintos archivos de la carpeta archivos. Una vez realizada la lectura y extraído el texto, un modelo LLM obtiene la información relevante y los valores deseados para poder realizar la recuperación de los embeddings y el asesoramiento de la alimentación que podría seguir el paciente, generando una respuesta en texto plano. El objetivo es que el nutricionista pueda adecuar estos alimentos y asesorar la dieta del paciente.

Por otro lado la carpetas **faiss_index_BAAI** son el almacenamiento de los embeddings creados una vez realizada la lectura a partir del archivo **emmbedings.py**.
La carpeta **planes_generados** contiene la salida obtenida tras realizar el proceso completo.
La carpeta **archivos** contiene los archivos con los que se ha trabajado para realizar el proyecto.

En el archivo **proceso2.py** hacen falta 2 conexiones en la linea 20 y 22, hacen falta editarlas.
Primero para el uso de la API de Groq, se debe acceder a la web [https://groq.com/]. Una vez dentro, pulsar en el desplegable de la parte superior derecha (las tres líneas), entrar en Developers, donde aparece la opción Free API Key. Después se debe pulsar en Create API Key, asignarle un nombre y establecer el tiempo de uso permitido. Finalmente, copiar la clave generada y añadirla en la línea 20 del código **proceso2.py** después del igual.
Por otro lado, para la lectura con pytesseract se debe acceder al enlace [https://github.com/UB-Mannheim/tesseract/wiki], una vez dentro, se descarga el archivo tesseract-ocr-w64-setup-5.5.0.20241111.exe (64 bit), instalarlo seleccionando el idioma deseado y posteriormente utilizar la ruta donde se haya instalado para poder realizar la lectura de imágenes en la línea 22 del codigo **proceso2.py** después del igual.


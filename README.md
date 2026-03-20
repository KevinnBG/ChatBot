# Implementación de un sistema RAG con interfaz interactiva

## Descripción general

Este código implementa un sistema básico de **Retrieval-Augmented Generation (RAG)** que combina búsqueda semántica y generación de texto mediante un modelo de lenguaje. El sistema permite realizar preguntas sobre un conjunto de documentos, recuperando primero información relevante y generando posteriormente una respuesta basada únicamente en ese contexto.

Además, se incluye una interfaz gráfica interactiva utilizando `ipywidgets`, lo que facilita la interacción del usuario con el sistema.

---

## Instalación de librerías

Se instalan las dependencias necesarias para el funcionamiento del sistema, incluyendo herramientas para procesamiento de lenguaje natural, embeddings, modelos de lenguaje y visualización interactiva.

Entre las principales librerías utilizadas se encuentran:

* langchain
* langchain-community
* langchain-text-splitters
* sentence-transformers
* transformers
* accelerate
* faiss-cpu
* ipywidgets
<img width="426" height="285" alt="image" src="https://github.com/user-attachments/assets/d648063f-2086-4e5a-8752-2c2053dc3a1d" />

Estas permiten gestionar documentos, generar representaciones vectoriales, realizar búsquedas eficientes y construir la interfaz.

---

## Importación de librerías

El código importa los módulos necesarios para cada parte del sistema:

* `torch`: manejo del modelo y aceleración por GPU
* `CharacterTextSplitter`: división de textos en fragmentos
* `HuggingFaceEmbeddings`: generación de embeddings
* `FAISS`: almacenamiento y búsqueda vectorial
* `transformers`: carga del modelo de lenguaje
* `ipywidgets`: creación de la interfaz interactiva
* `threading` y `time`: control de la animación de carga
<img width="659" height="299" alt="image" src="https://github.com/user-attachments/assets/b55de7f4-8538-48d4-92a4-eb0ade7bed18" />

---

## Verificación de GPU

Se verifica si existe una GPU disponible mediante:

```python
torch.cuda.is_available()
```
<img width="555" height="126" alt="image" src="https://github.com/user-attachments/assets/2980be74-5cb1-4513-80ba-7a5e0452e14f" />

Esto permite determinar si el modelo puede ejecutarse con aceleración por hardware o si se utilizará únicamente CPU.

---

## Definición de documentos

Se define una lista de documentos de ejemplo que funcionan como base de conocimiento. Estos textos contienen información breve sobre distintos conceptos tecnológicos como programación, contenedores, sistemas operativos y modelos de inteligencia artificial.
<img width="847" height="326" alt="image" src="https://github.com/user-attachments/assets/45b2236f-161b-4613-b3f0-b5ec1c9354a6" />

---

## División de documentos

Los documentos se dividen en fragmentos más pequeños utilizando `CharacterTextSplitter`, lo cual mejora la precisión en la recuperación de información.

Parámetros utilizados:

* Tamaño de fragmento: 150 caracteres
* Superposición: 20 caracteres
<img width="731" height="154" alt="image" src="https://github.com/user-attachments/assets/819c911c-0878-4566-be8c-aa225a76dcf9" />

Esto permite mantener contexto entre fragmentos consecutivos.

---

## Creación de embeddings y almacenamiento vectorial

Se generan embeddings utilizando un modelo preentrenado de `sentence-transformers`. Estos embeddings convierten el texto en vectores numéricos que capturan su significado semántico.
<img width="897" height="139" alt="image" src="https://github.com/user-attachments/assets/9daf3687-82f8-47f0-9594-36861591b413" />

Posteriormente, estos vectores se almacenan en un índice FAISS, lo que permite realizar búsquedas rápidas y eficientes basadas en similitud.

---

## Carga del modelo de lenguaje

Se carga un modelo de lenguaje tipo causal junto con su tokenizador. Este modelo es responsable de generar las respuestas en lenguaje natural.

Configuraciones relevantes:

* Uso de precisión reducida para optimizar memoria
* Asignación automática del dispositivo (CPU o GPU)
<img width="575" height="257" alt="image" src="https://github.com/user-attachments/assets/5c76fb93-5eae-4d6d-8ca1-87562f8b3294" />

---

## Generación de respuestas

La función encargada de generar texto realiza los siguientes pasos:

1. Tokeniza el texto de entrada
2. Ejecuta la generación con el modelo
3. Decodifica la salida a texto legible

Parámetros importantes:

* Número máximo de tokens generados
* Temperatura baja para respuestas más deterministas
* Muestreo activado para variación controlada
<img width="713" height="382" alt="image" src="https://github.com/user-attachments/assets/6ed42ab7-85c9-4c14-a412-fb627e2642cc" />

---

## Interfaz gráfica

Se construye una interfaz interactiva compuesta por:

* Un área de texto para ingresar preguntas
* Un botón para ejecutar la consulta
* Un área de salida para mostrar resultados
<img width="739" height="505" alt="image" src="https://github.com/user-attachments/assets/2f298946-adb3-4304-a701-e05781f670a6" />

Esto permite una interacción directa y sencilla con el sistema RAG.

---

## Animación de carga

Se implementa una animación simple que muestra puntos suspensivos mientras el sistema procesa la consulta. Esto se realiza mediante un hilo independiente que actualiza la interfaz en tiempo real.
<img width="714" height="379" alt="image" src="https://github.com/user-attachments/assets/7149b7c3-a53a-474f-a5d9-3b9a1157be5a" />

---

## Implementación del sistema RAG

La función principal del sistema realiza:

1. Búsqueda de los documentos más relevantes mediante similitud semántica

2. Construcción de un contexto a partir de esos documentos

3. Creación de un prompt restringido que obliga al modelo a:

   * Usar únicamente el contexto proporcionado
   * No agregar información externa
   * Generar respuestas breves

4. Generación de la respuesta final

También se incluye una limpieza del texto generado para evitar contenido innecesario.
<img width="710" height="630" alt="image" src="https://github.com/user-attachments/assets/c6ccc509-7342-4893-95d6-a8b192711892" />

---

## Manejo de eventos del botón

Cuando el usuario presiona el botón:

1. Se obtiene la pregunta ingresada
2. Se activa la animación de carga
3. Se ejecuta el sistema RAG
4. Se detiene la animación
5. Se muestran los resultados en pantalla, incluyendo:

   * El contexto recuperado
   * La respuesta generada
<img width="591" height="633" alt="image" src="https://github.com/user-attachments/assets/9210284a-e276-4638-b7dc-8e74cb832c7e" />
<img width="373" height="128" alt="image" src="https://github.com/user-attachments/assets/3d8bfc95-e657-44ca-a84c-cf74a0f684af" />

---

## Visualización de la interfaz

Todos los elementos de la interfaz se organizan en un contenedor vertical y se muestran en pantalla, permitiendo su uso inmediato dentro del entorno interactivo.

<img width="550" height="150" alt="image" src="https://github.com/user-attachments/assets/a8a994b3-6a2e-4e4b-acda-e0e11c3919a9" />

---

## Notas adicionales

* Pueden aparecer advertencias relacionadas con dependencias, pero no afectan el funcionamiento general.
* Algunas librerías utilizadas pueden estar en proceso de actualización o deprecación.
* Si no se dispone de GPU, el sistema funcionará en CPU, lo que puede reducir el rendimiento.
* La autenticación en servicios externos es opcional, pero puede mejorar la velocidad de descarga de modelos.

---
<img width="1064" height="608" alt="image" src="https://github.com/user-attachments/assets/022d08c4-f702-44cb-82a6-edeaf3b89dbe" />


## Conclusión

Este código representa una implementación funcional de un sistema RAG que integra recuperación de información y generación de texto. Es una base sólida para desarrollar aplicaciones más avanzadas de հարց-respuesta, asistentes inteligentes o sistemas de consulta sobre documentos.


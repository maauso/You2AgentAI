Aquí tienes la lista de accionables para el bloque **DATASET**, enfocada en la transición de datos locales de Telegram hacia un flujo de trabajo profesional con Hugging Face y Mistral-7B-v0.3 (Base).

---

### Bloque 1: DATASET (Accionables)

#### 1. Migración a Fuente de Datos en la Nube (Hugging Face)

* **Descripción**: Sustituir la lógica de carga de archivos JSON locales de Telegram por la carga automatizada del dataset `timdettmers/openassistant-guanaco`. Esto garantiza que cualquier lector del blog pueda ejecutar el código sin necesidad de tener un historial de chat propio.
* **Código de ejemplo**:
```python
from datasets import load_dataset
# Carga la partición de entrenamiento directamente desde el Hub
raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

```


* **Acceptance Criteria**: El script de preparación de datos ya no debe requerir la carpeta `telegram_chats/` ni archivos `.json` locales para funcionar.

#### 2. Estandarización al Formato ChatML

* **Descripción**: Dado que se utilizará un modelo **Base**, es imperativo convertir el formato original del dataset (`### Human: / ### Assistant:`) al estándar **ChatML**. Esto enseña al modelo dónde empieza y termina cada interlocutor de forma explícita.
* **Ejemplo de formato**:
```text
<|im_start|>user
¿Qué es un sistema distribuido?<|im_end|>
<|im_start|>assistant
Es un conjunto de computadoras que trabajan juntas...<|im_end|>

```


* **Acceptance Criteria**: El 100% de los ejemplos procesados deben estar encapsulados entre los tokens `<|im_start|>` y `<|im_end|>`.

#### 3. Simplificación del Script de Filtrado

* **Descripción**: Eliminar el archivo `1_Dataset/generate_filtered_dataset.py` tal como existe (con filtros de `question_words` y Regex) y sustituirlo por un script de mapeo sencillo. El dataset de Guanaco ya está curado, por lo que el filtrado manual es innecesario y añade complejidad cognitiva al lector.
* **Acción**: Crear un nuevo script `prepare_dataset.py` que simplemente cargue el dataset y aplique una función de formateo de texto.
* **Acceptance Criteria**: El nuevo script debe tener menos de 40 líneas de código y centrarse únicamente en la transformación de formato.

#### 4. Adición de Utilidad de Inspección Visual

* **Descripción**: Implementar una función que imprima en consola los primeros 3 ejemplos del dataset una vez formateado. Esto es crucial para que el usuario "vea" qué está inyectando al modelo antes de gastar recursos de GPU en el entrenamiento.
* **Código de ejemplo**:
```python
def preview_dataset(dataset, num_samples=3):
    for i in range(num_samples):
        print(f"--- Ejemplo {i+1} ---\n{dataset[i]['text']}\n")

```


* **Acceptance Criteria**: Al ejecutar la fase de dataset, el output de la consola debe mostrar claramente los tags de ChatML para que el redactor técnico pueda tomar capturas de pantalla para el blog.

#### 5. Documentación de la Elección del Dataset (Contexto Técnico)

* **Descripción**: Proveer al redactor técnico los motivos de la elección: "Guanaco" es el estándar para tutoriales de QLoRA por su alta calidad y equilibrio entre respuestas técnicas y conversacionales.
* **Nota para el redactor**: Explicar que se usa este dataset para asegurar la **reproducibilidad**, permitiendo que el foco del artículo sea la técnica de fine-tuning y no la limpieza de datos.
* **Acceptance Criteria**: El borrador del blog debe incluir una sección titulada "¿Por qué OpenAssistant Guanaco?" que mencione la calidad de los datos y la facilidad de integración.

---

**¿Deseas que hagamos algún ajuste en estos accionables de Dataset o pasamos directamente al siguiente bloque (TOKENIZER)?**

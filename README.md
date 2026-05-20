# Comprensión Automática de Documentos Científicos

Sistema multimodal para comprensión automática de papers científicos mediante técnicas de visión por computador, OCR, extracción estructural, modelos visión-lenguaje y modelos de lenguaje de gran escala.

Repositorio oficial:  
[comprension_automatica_documentos](https://github.com/jalvarado12/comprension_automatica_documentos)

---

# Descripción del proyecto

Este proyecto implementa un pipeline completo de **Document Understanding** orientado a documentos científicos en formato PDF.

El sistema es capaz de:

- Convertir PDFs científicos a imágenes
- Aplicar preprocesamiento visual
- Detectar regiones relevantes del documento
- Extraer texto mediante OCR
- Detectar y reconstruir tablas
- Extraer figuras y gráficas
- Generar captions contextuales usando Florence-2
- Integrar información multimodal
- Corregir errores OCR mediante Gemini
- Generar resúmenes científicos automáticos
- Evaluar el desempeño del sistema mediante métricas

El objetivo final es transformar documentos científicos complejos en representaciones estructuradas y comprensibles automáticamente.

---

# Arquitectura general

```text
PDF
│
├── Conversión PDF → imágenes
│
├── Preprocesamiento visual
│
├── Layout Detection
│
├── OCR
│
├── Extracción de tablas
│   └── Reconstrucción markdown
│
├── Extracción de figuras
│   └── Captioning contextual Florence-2
│
├── Integración multimodal
│
├── Corrección contextual con Gemini
│
└── Generación de resumen científico
```

---

# Tecnologías utilizadas

## Visión por computador

- OpenCV
- PaddleOCR
- EasyOCR
- pdf2image
- LayoutParser

## Modelos multimodales

- Florence-2
- Table Transformer (TATR)

## Modelos de lenguaje

- Gemini Flash (Google AI Studio)

## Evaluación

- ROUGE
- BERTScore
- CER
- WER

---

# Requisitos del sistema

## Sistema operativo

Recomendado:

- Windows 10/11
- Ubuntu Linux

---

## Python

Versión recomendada:

```text
Python 3.11
```

---

## GPU (opcional pero recomendada)

El pipeline puede ejecutarse en CPU, pero Florence-2 y OCR funcionan considerablemente mejor con GPU NVIDIA compatible con CUDA.

---

# Instalación completa

# 1. Clonar el repositorio

```bash
git clone https://github.com/jalvarado12/comprension_automatica_documentos.git
```

Entrar a la carpeta:

```bash
cd comprension_automatica_documentos
```

---

# 2. Crear entorno virtual

## Windows

```bash
python -m venv venv
```

Activar entorno:

```bash
venv\Scripts\activate
```

---

## Linux / Mac

```bash
python3 -m venv venv
```

Activar entorno:

```bash
source venv/bin/activate
```

---

# 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

---

# 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Dependencias importantes

El proyecto requiere versiones específicas compatibles entre OCR, OpenCV y transformers.

Las versiones utilizadas durante el desarrollo fueron:

```text
numpy==1.26.4
opencv-python==4.6.0.66
paddlepaddle==2.6.2
paddleocr==2.7.3
transformers==4.46.3
einops==0.8.1
```

---

# 5. Instalar Poppler (MUY IMPORTANTE)

El proyecto utiliza `pdf2image`, por lo que necesitas instalar Poppler.

## Windows

Descargar:

https://github.com/oschwartz10612/poppler-windows/releases

---

## Pasos

1. Descargar el ZIP
2. Extraerlo
3. Agregar la carpeta `bin` al PATH del sistema

Ejemplo:

```text
C:\poppler\Library\bin
```

4. Reiniciar VS Code o terminal

---

## Verificar instalación

```bash
pdftoppm -h
```

Si aparece ayuda de Poppler, quedó correctamente instalado.

---

# Configuración de API

El proyecto utiliza Google AI Studio para:

- corrección contextual OCR
- integración semántica
- generación de resumen científico

---

# Crear archivo `.env`

Debes crear manualmente un archivo llamado:

```text
.env
```

en la raíz del proyecto.

---

# Contenido del `.env`

```env
GOOGLE_API_KEY=TU_API_KEY_AQUI
```

---

# Obtener API Key

Entrar a:

https://aistudio.google.com/app/apikey

Crear una API Key y copiarla dentro del archivo `.env`.

---

# Estructura del proyecto

```text
comprension_automatica_documentos/

│
├── data/
│
├── outputs/
│
├── src/
│
├── run_pipeline.py
│
├── requirements.txt
│
├── .env
│
└── README.md
```

---

# Ejecución del pipeline

El pipeline completo se ejecuta desde terminal.

---

# Ejemplo de ejecución

```bash
python run_pipeline.py --pdf "C:\Users\user\Downloads\v60_n2_217_220.pdf" --run-name prueba2
```

---

# Parámetros

## `--pdf`

Ruta completa del PDF científico.

Ejemplo:

```text
"C:\Users\user\Downloads\paper.pdf"
```

---

## `--run-name`

Nombre de la ejecución.

Esto crea una carpeta independiente dentro de `data/outputs/`.

Ejemplo:

```text
data/outputs/prueba2/
```

---

# Qué hace automáticamente el pipeline

Cuando ejecutas:

```bash
python run_pipeline.py --pdf "ruta_pdf" --run-name nombre
```

el sistema realiza:

1. Conversión PDF → imágenes
2. Preprocesamiento visual
3. OCR
4. Layout detection
5. Extracción de figuras
6. Extracción de tablas
7. Reconstrucción markdown tabular
8. Captioning contextual Florence-2
9. Integración multimodal
10. Corrección contextual con Gemini
11. Generación de resumen técnico
12. Evaluación automática

---

# Dónde quedan los resultados

Todos los resultados se guardan dentro de:

```text
data/outputs/
```

---

# Carpeta de ejecución

Si usas:

```bash
--run-name prueba2
```

los resultados quedarán en:

```text
data/outputs/prueba2/
```

---

# Archivos MÁS IMPORTANTES

# 1. Documento corregido final

```text
data/outputs/prueba2/40_llm/markdown/corrected_document.md
```

Contiene:

- OCR corregido
- integración multimodal
- tablas integradas
- captions contextualizados
- documento estructurado

---

# 2. Resumen científico final

```text
data/outputs/prueba2/40_llm/markdown/final_summary.md
```

Contiene:

- objetivos
- metodología
- resultados
- conclusiones

generados automáticamente mediante Gemini.

---

# 3. Métricas del sistema

```text
outputs/prueba2/50_metrics/csv/metrics.csv
```

Incluye:

- OCR metrics
- reducción de ruido
- preservation ratio
- compression ratio
- métricas de resumen

---

# Otros outputs importantes

## OCR raw

```text
data/outputs/prueba2/09_ocr_txt/ocr/
```

---

## Tablas reconstruidas

```text
data/outputs/prueba2/22_tables_markdown/
```

---

## Figuras extraídas

```text
data/outputs/prueba2/06_figures/
```

---

## Captions Florence-2

```text
data/outputs/prueba2/12_image2text/
```

---

# Problemas comunes

# Error con Poppler

Verificar:

```bash
pdftoppm -h
```

y confirmar que Poppler está agregado al PATH.

---

# Error con PaddleOCR

El proyecto requiere versiones compatibles específicas de Paddle y NumPy.

---

# Error con Florence-2

Instalar:

```bash
pip install transformers==4.46.3
pip install einops==0.8.1
```

---

# Error CUDA

Si no tienes GPU compatible, el pipeline puede ejecutarse en CPU, aunque será considerablemente más lento.

---

# Estado actual del proyecto

Actualmente el sistema ya implementa:

- OCR multimodal
- extracción estructural
- reconstrucción tabular
- captioning contextual
- integración multimodal
- corrección contextual con LLM
- síntesis automática

Los siguientes pasos incluyen:

- evaluación formal completa
- optimización del pipeline
- transición definitiva a producción

---

# Autor

Sebastián Alvarado  
Universidad del Rosario

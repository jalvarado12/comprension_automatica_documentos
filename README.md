# Comprensión Automática de Documentos Científicos

Sistema multimodal para comprensión automática de papers científicos mediante técnicas de visión por computador, OCR, extracción estructural, modelos visión-lenguaje y modelos de lenguaje de gran escala.

---

## Descripción del proyecto

Este proyecto implementa un pipeline completo de **Document Understanding** orientado a documentos científicos en formato PDF.

El sistema es capaz de:

- Convertir PDFs científicos a imágenes de alta resolución
- Aplicar preprocesamiento visual por página
- Detectar regiones relevantes del documento (texto, tablas, figuras)
- Extraer texto mediante OCR con PaddleOCR
- Detectar y reconstruir tablas en formato markdown usando Table Transformer (TATR)
- Extraer figuras y gráficas por región
- Generar captions contextuales usando Florence-2
- Integrar toda la información en un documento multimodal estructurado
- Corregir errores OCR y generar resúmenes científicos automáticos mediante Gemini
- Evaluar el desempeño del sistema mediante métricas (ROUGE, BERTScore, CER, WER)

---

## Arquitectura general

```
PDF
│
├── Etapa 1 — Conversión PDF → imágenes
│
├── Etapa 2-3 — Layout Detection + TATR + Merge de regiones
│
├── Etapa 3 — OCR sobre bloques de texto (PaddleOCR)
│
├── Etapa 4 — Extracción de tablas → Reconstrucción markdown (TATR)
│
├── Etapa 5 — Extracción de figuras → Captioning contextual (Florence-2)
│
├── Etapa 6 — Integración multimodal (documento JSON + markdown)
│
├── Etapa 7 — Corrección contextual OCR + resumen científico (Gemini)
│
└── Etapa 8 — Evaluación automática (ROUGE, BERTScore, CER, WER)
```

---

## Tecnologías utilizadas

### Visión por computador
- OpenCV
- PaddleOCR
- pdf2image + Poppler

### Modelos multimodales
- Florence-2 (captioning contextual de figuras)
- Table Transformer — TATR (detección y reconstrucción de tablas)

### Modelos de lenguaje
- Gemini Flash (Google AI Studio) — corrección OCR y síntesis documental

### Evaluación
- ROUGE
- BERTScore
- CER (Character Error Rate)
- WER (Word Error Rate)

---

## Requisitos del sistema

### Sistema operativo
- Windows 10/11
- Ubuntu Linux

### Python
```
Python 3.11
```

### GPU (opcional pero recomendada)
El pipeline puede ejecutarse en CPU, pero Florence-2 y PaddleOCR funcionan considerablemente mejor con GPU NVIDIA compatible con CUDA.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/jalvarado12/comprension_automatica_documentos.git
cd comprension_automatica_documentos
```

### 2. Crear y activar entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

Las versiones críticas utilizadas durante el desarrollo son:

```
numpy==1.26.4
opencv-python==4.6.0.66
paddlepaddle==2.6.2
paddleocr==2.7.3
transformers==4.46.3
einops==0.8.1
```

> **Nota:** estas versiones deben respetarse. PaddleOCR, OpenCV y transformers tienen incompatibilidades conocidas con versiones más recientes.

---

## Instalación de Poppler (obligatorio)

El proyecto usa `pdf2image`, que requiere Poppler instalado en el sistema.

### Windows

1. Descargar desde: https://github.com/oschwartz10612/poppler-windows/releases
2. Extraer el ZIP
3. Agregar la carpeta `bin` al PATH del sistema (ejemplo: `C:\poppler\Library\bin`)
4. Reiniciar la terminal o VS Code

**Verificar instalación:**
```bash
pdftoppm -h
```
Si aparece la ayuda de Poppler, la instalación fue exitosa.

### Linux

```bash
sudo apt-get install poppler-utils
```

---

## Configuración de API key

El proyecto usa Google AI Studio (Gemini) para la corrección contextual OCR y la generación de resúmenes.

### Crear archivo `.env`

Crear un archivo llamado `.env` en la raíz del proyecto con el siguiente contenido:

```
GEMINI_API_KEY=TU_API_KEY_AQUI
```

> **Importante:** la variable debe llamarse exactamente `GEMINI_API_KEY`, que es el nombre que lee el código en `run_pipeline.py`.

### Obtener API Key

Ingresar a https://aistudio.google.com/app/apikey, crear una API Key y copiarla en el `.env`.

---

## Estructura del proyecto

```
comprension_automatica_documentos/
│
├── app/                        # Servidor FastAPI (API REST del pipeline)
├── configs/                    # Archivos de configuración YAML
├── data/
│   ├── raw/                    # PDFs de entrada
│   ├── ground_truth/           # Textos de referencia para evaluación
│   └── outputs/                # Resultados organizados por run-name
├── src/
│   ├── pdf/                    # Renderizado PDF
│   ├── layout/                 # Detección de layout documental
│   ├── ocr/                    # Motor OCR (PaddleOCR)
│   ├── tables/                 # TATR: detección y reconstrucción de tablas
│   ├── captioning/             # Florence-2: captioning de figuras
│   ├── document/               # Construcción del documento multimodal
│   ├── llm/                    # Cliente Gemini: corrección y resumen
│   ├── evaluation/             # Métricas automáticas
│   ├── io/                     # Lectura/escritura de manifests y figuras
│   └── utils/                  # Utilidades (geometría, config, paths, etc.)
├── tests/                      # Tests del proyecto
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run_pipeline.py             # Entrypoint del pipeline completo
```

---

## Ejecución del pipeline

```bash
python run_pipeline.py --pdf "ruta/al/paper.pdf" --run-name nombre_ejecucion
```

### Parámetros disponibles

| Parámetro | Descripción |
|---|---|
| `--pdf` | Ruta al PDF científico de entrada (obligatorio) |
| `--run-name` | Nombre de la ejecución; crea una carpeta en `data/outputs/` (obligatorio) |
| `--config` | Ruta al archivo de configuración (por defecto: `configs/pipeline.yaml`) |
| `--skip-ocr` | Omite las etapas 1–4 (PDF, layout, OCR, tablas) |
| `--skip-florence` | Omite la etapa 5 (captioning de figuras) |
| `--skip-llm` | Omite la etapa 7 (corrección y resumen con Gemini) |
| `--skip-metrics` | Omite la etapa 8 (evaluación) |
| `--only-metrics` | Ejecuta únicamente la etapa de evaluación |

### Ejemplo completo

```bash
python run_pipeline.py --pdf "data/raw/paper.pdf" --run-name prueba1
```

### Ejemplo sin LLM (sin consumir API)

```bash
python run_pipeline.py --pdf "data/raw/paper.pdf" --run-name prueba1 --skip-llm
```

---

## Outputs generados

Todos los resultados se guardan en `data/outputs/<run-name>/`:

| Carpeta | Contenido |
|---|---|
| `01_page_images/` | Imágenes por página del PDF |
| `02_preprocessed_pages/` | Páginas preprocesadas |
| `06_figures/` | Figuras y gráficas extraídas |
| `08_ocr_json/` | OCR estructurado en JSON por bloque |
| `09_ocr_txt/` | Texto OCR plano por bloque |
| `11_manifests/` | Detecciones en CSV y JSONL |
| `12_image2text/` | Captions contextuales de Florence-2 |
| `22_tables_markdown/` | Tablas reconstruidas en formato markdown |
| `30_multimodal/` | Documento multimodal integrado (JSON + markdown) |
| `40_llm/markdown/corrected_document.md` | Documento corregido final por Gemini |
| `40_llm/markdown/final_summary.md` | Resumen científico automático |
| `50_metrics/csv/metrics.csv` | Métricas de evaluación del pipeline |

---

## Docker

El proyecto incluye un `Dockerfile` y `docker-compose.yml`. La imagen instala automáticamente Poppler, Tesseract y todas las dependencias Python.

### Construir la imagen

```bash
docker build -t comprension-docs .
```

### Ejecutar el pipeline con Docker

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  comprension-docs \
  python run_pipeline.py --pdf data/raw/paper.pdf --run-name prueba_docker
```

> Para usar GPU con Docker se requiere `nvidia-docker2` instalado en el host.

---

## Problemas comunes

### Error con Poppler
```
PDFPageCountError
```
Verificar que `pdftoppm -h` funciona en la terminal y que la carpeta `bin` de Poppler está en el PATH.

### Error con PaddleOCR / NumPy
Respetar exactamente las versiones de `requirements.txt`. Conflictos frecuentes ocurren con `numpy > 1.26` y `paddlepaddle > 2.6.2`.

### Error con Florence-2 / transformers
```bash
pip install transformers==4.46.3 einops==0.8.1
```

### GEMINI_API_KEY no definida
Verificar que el archivo `.env` existe en la raíz del proyecto y que la variable se llama exactamente `GEMINI_API_KEY`.

### Sin GPU (CPU only)
El pipeline funciona en CPU pero Florence-2 y PaddleOCR serán considerablemente más lentos. Se puede omitir Florence-2 con `--skip-florence`.

---

## Autor

Sebastián Alvarado  
Universidad del Rosario

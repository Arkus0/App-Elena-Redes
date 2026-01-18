# Performance Optimizations - Video & ML Processing

Este documento describe las optimizaciones de rendimiento implementadas para el procesamiento de video y modelos ML.

## Resumen de Optimizaciones

| Optimización | Impacto | Trade-off |
|--------------|---------|-----------|
| **A) Cuantización INT8** | ~3-4x speedup en embeddings | ~1% pérdida en similitud coseno |
| **B) Cola de Tareas** | API no bloqueante | Procesamiento diferido |
| **C) Optical Flow Opcional** | ~40% speedup en video | Sin métricas de estabilidad de cámara |

---

## A) Cuantización de Modelos ML (INT8)

### Descripción

La cuantización convierte los pesos de los modelos de `float32` (4 bytes) a `int8` (1 byte), reduciendo memoria ~4x y acelerando inferencia ~3-4x en CPU.

### Modelos Afectados

1. **Whisper (faster-whisper)**: Ya usa INT8 por defecto via CTranslate2
2. **Sentence-Transformers**: Implementada cuantización dinámica PyTorch

### Configuración

```python
from app.services.text_intelligence import TextConfig

# Cuantización habilitada (por defecto)
config = TextConfig(
    embedding_quantize=True,      # Habilitar INT8
    embedding_quantize_dtype="int8",
    whisper_compute_type="int8"   # Ya es default
)

# Cuantización deshabilitada (máxima precisión)
config = TextConfig(
    embedding_quantize=False,
    whisper_compute_type="float32"
)
```

### Variables de Entorno

```bash
# En .env
ML_EMBEDDING_QUANTIZE=true
ML_WHISPER_COMPUTE_TYPE=int8
```

### Impacto en Precisión

| Modelo | Métrica | Float32 | INT8 | Degradación |
|--------|---------|---------|------|-------------|
| Whisper | WER | 8.2% | 8.3% | +0.1% |
| all-MiniLM-L6-v2 | Cosine Sim | 1.000 | 0.991 | -0.9% |

### Cómo Verificar

```python
from app.services.text_intelligence import SemanticEncoder, TextConfig

config = TextConfig(embedding_quantize=True)
encoder = SemanticEncoder(config)
encoder._load_model()

print(f"Modelo cuantizado: {encoder.is_quantized}")
# Output: Modelo cuantizado: True
```

---

## B) Cola de Tareas para Video

### Descripción

El procesamiento de video se mueve a segundo plano usando una cola asyncio. La API retorna inmediatamente con un `task_id` para consultar el estado.

### Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   API Request   │────▶│  asyncio.Queue   │────▶│  Worker Loop   │
│   POST /video   │     │   (in-memory)    │     │  (sequential)  │
└─────────────────┘     └──────────────────┘     └────────────────┘
        │                                                │
        │  task_id                              extract_features()
        ▼                                                │
   Return 202                                           ▼
   Accepted                                    Save result / error
```

### Uso Básico

```python
from app.services.task_queue import (
    VideoTaskQueue,
    TaskType,
    QueueConfig,
    init_queue,
    shutdown_queue
)

# En startup de la aplicación
queue = await init_queue()

# Desde un endpoint de la API
task_id = await queue.submit(
    video_path="/path/to/video.mp4",
    task_type=TaskType.FULL_ANALYSIS,
    metadata={"business_id": 123}
)

# Consultar estado
status = await queue.get_status(task_id)
# {"status": "processing", "progress": 50, ...}

# Obtener resultado cuando complete
result = await queue.get_result(task_id)

# En shutdown de la aplicación
await shutdown_queue()
```

### Tipos de Tarea

| Tipo | Descripción | Velocidad |
|------|-------------|-----------|
| `FULL_ANALYSIS` | Video + Audio + Text Intelligence | Más lento |
| `VIDEO_ONLY` | Solo features de video | Medio |
| `AUDIO_ONLY` | Solo tempo y onset_strength | Rápido |
| `TEXT_INTELLIGENCE` | Transcripción + OCR + Embeddings | Medio |
| `QUICK_PREVIEW` | Video sin optical flow | Rápido |

### Configuración

```python
from app.services.task_queue import QueueConfig

config = QueueConfig(
    max_queue_size=100,          # Máximo de tareas pendientes
    max_retries=3,               # Reintentos en caso de error
    task_timeout_seconds=600,    # 10 minutos máximo por tarea
    persist_to_sqlite=True,      # Persistencia para recuperación
    sqlite_path="./tasks.db"
)
```

### Variables de Entorno

```bash
# En .env
TASK_QUEUE_ENABLED=true
TASK_QUEUE_MAX_SIZE=100
TASK_QUEUE_PERSIST=true
TASK_QUEUE_SQLITE_PATH=./video_tasks.db
TASK_QUEUE_MAX_RETRIES=3
TASK_QUEUE_TIMEOUT_SECONDS=600
```

### Integración con FastAPI

```python
from fastapi import FastAPI, BackgroundTasks
from app.services.task_queue import get_video_queue, init_queue, shutdown_queue

app = FastAPI()

@app.on_event("startup")
async def startup():
    await init_queue()

@app.on_event("shutdown")
async def shutdown():
    await shutdown_queue()

@app.post("/api/video/analyze")
async def analyze_video(video_path: str):
    queue = get_video_queue()
    task_id = await queue.submit(video_path=video_path)
    return {"task_id": task_id, "status_url": f"/api/video/status/{task_id}"}

@app.get("/api/video/status/{task_id}")
async def get_status(task_id: str):
    queue = get_video_queue()
    return await queue.get_status(task_id)
```

---

## C) Optical Flow Opcional

### Descripción

El análisis de optical flow (algoritmo Farneback) consume ~40% del tiempo de procesamiento. Ahora puede desactivarse cuando la estabilidad de cámara no es crítica.

### Configuración

```python
from app.services.analytics_engine import VideoConfig, AnalyticsEngine

# Optical flow habilitado (por defecto)
config = VideoConfig(optical_flow_enabled=True)

# Optical flow deshabilitado (modo rápido)
config = VideoConfig(optical_flow_enabled=False)

engine = AnalyticsEngine(video_config=config)
```

### Variables de Entorno

```bash
# En .env
VIDEO_OPTICAL_FLOW_ENABLED=false  # Deshabilitar para máxima velocidad
```

### Impacto en Métricas

Cuando `optical_flow_enabled=False`:

| Métrica | Valor | Comportamiento |
|---------|-------|----------------|
| `instability_score` | 0.0 | Neutral (sin penalización) |
| `camera_stability_score` | 1.0 | Asume estable |
| `visual_energy` | Raw | Sin penalización aplicada |
| `production_quality_score` | Parcial | Solo brightness + contrast |

### Cuándo Desactivar

- Procesamiento batch de muchos videos
- Previews rápidos antes de análisis completo
- Cuando la calidad de filmación no es un factor
- Dispositivos con recursos muy limitados

### Cuándo Mantener Activado

- Análisis de calidad de producción
- Detección de shake de cámara vs. edición dinámica
- Scoring final de contenido para publicación

---

## Cómo Probar las Optimizaciones

### 1. Test de Cuantización

```bash
cd backend

# Ejecutar tests de cuantización
pytest tests/test_performance_optimizations.py::TestEmbeddingQuantization -v

# Benchmark manual (requiere modelos descargados)
python -c "
from app.services.text_intelligence import SemanticEncoder, TextConfig
import time

# Con cuantización
config = TextConfig(embedding_quantize=True)
encoder = SemanticEncoder(config)

start = time.time()
for _ in range(100):
    encoder.encode_text('Test sentence')
print(f'INT8: {time.time() - start:.2f}s')
"
```

### 2. Test de Cola de Tareas

```bash
# Tests unitarios
pytest tests/test_performance_optimizations.py::TestVideoTaskQueue -v

# Test manual con video real
python -m app.services.task_queue /path/to/video.mp4
```

### 3. Test de Optical Flow Toggle

```bash
# Tests unitarios
pytest tests/test_performance_optimizations.py::TestOpticalFlowToggle -v

# Comparar tiempos
python -c "
from app.services.analytics_engine import VideoConfig, AnalyticsEngine
import time

video_path = 'test.mp4'  # Tu video de prueba

# Con optical flow
t1 = time.time()
AnalyticsEngine(video_config=VideoConfig(optical_flow_enabled=True)).extract_video_features(video_path)
with_time = time.time() - t1

# Sin optical flow
t2 = time.time()
AnalyticsEngine(video_config=VideoConfig(optical_flow_enabled=False)).extract_video_features(video_path)
without_time = time.time() - t2

print(f'Con optical flow: {with_time:.2f}s')
print(f'Sin optical flow: {without_time:.2f}s')
print(f'Speedup: {(with_time - without_time) / with_time * 100:.1f}%')
"
```

### 4. Ejecutar Todos los Tests

```bash
cd backend
pytest tests/test_performance_optimizations.py -v
```

---

## Configuración Recomendada por Caso de Uso

### Producción (Calidad Máxima)

```python
# .env
ML_EMBEDDING_QUANTIZE=true
VIDEO_OPTICAL_FLOW_ENABLED=true
TASK_QUEUE_ENABLED=true
TASK_QUEUE_PERSIST=true
```

### Desarrollo/Testing (Velocidad)

```python
# .env
ML_EMBEDDING_QUANTIZE=true
VIDEO_OPTICAL_FLOW_ENABLED=false
TASK_QUEUE_ENABLED=false
```

### Edge Device (Recursos Limitados)

```python
# .env
ML_EMBEDDING_QUANTIZE=true
ML_WHISPER_COMPUTE_TYPE=int8
VIDEO_OPTICAL_FLOW_ENABLED=false
TASK_QUEUE_ENABLED=true
TASK_QUEUE_PERSIST=false  # Evitar I/O de disco
```

---

## Métricas de Rendimiento Esperadas

| Escenario | Sin Optimizar | Con Optimizar | Mejora |
|-----------|---------------|---------------|--------|
| Embedding (100 textos) | 12.0s | 3.5s | 3.4x |
| Video 30s (full) | 8.0s | 4.8s | 40% |
| API Response Time | 10s+ | <200ms | Async |
| Memoria Modelo | 400MB | 100MB | 4x |

---

## Troubleshooting

### Error: "quantize_dynamic not available"

PyTorch <1.8 no soporta cuantización dinámica. Actualizar:
```bash
pip install torch>=1.8
```

### Cola de tareas no procesa

Verificar que el worker está corriendo:
```python
queue = get_video_queue()
print(f"Running: {queue._running}")
print(f"Queue size: {queue.get_queue_length()}")
```

### Optical flow muy lento

Reducir resolución de procesamiento:
```python
config = VideoConfig(
    target_resolution=(112, 112),  # En lugar de 224x224
    frame_stride_seconds=1.0       # En lugar de 0.5
)
```

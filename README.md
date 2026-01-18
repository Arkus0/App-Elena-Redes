# BrandPulse AI

> **El co-piloto definitivo para generar contenido de ALTO ENGAGEMENT para negocios locales**

BrandPulse AI combina la potencia generativa de **Grok (xAI)** con un motor de predicción de crecimiento basado en **XGBoost** y **SHAP**. No solo genera contenido: *predice* su éxito basándose en "Hook Theory" y patrones reales de competidores.

![BrandPulse AI](https://via.placeholder.com/800x400/1f2937/0ea5e9?text=BrandPulse+AI+v2)

## ✨ Características Principales

### 🧠 Arquitectura Híbrida ML/LLM
- **Grok (xAI) - Motor LLM principal para análisis creativo y generación de contenido**: Usando el modelo `grok-4-1-fast-reasoning` para guiones, copys y análisis de patrones.
- **Growth Prediction Engine**: Motor XGBoost que predice el **RPI Score** (Relative Performance Index) de cada idea.
- **Explicabilidad SHAP**: Entiende *por qué* un contenido funcionará con Top 5 factores (ej: "RPI alto por: pregunta en caption (+22%), hora 20:00 (+18%), formato Reel (+15%)").

### 📊 Feature Engineering Avanzado

#### 🧬 Semantic Embeddings (Configurable Precision - REQUIRED)
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB, multilingual)
- **Output**: 384-dim raw embeddings, reducción configurable via TruncatedSVD
- **Precisión configurable (REQUERIDO - no hay default)**:
  - `ultra_low` (64 dims): Ultra rápido para PC modesto/sobremesa Almería
  - `low` (128 dims): **RECOMENDADO** sobremesa normal (train <30s, RAM <0.5GB)
  - `medium` (256 dims): Balance precisión/velocidad
  - `high` (384 dims): Full dims con TruncatedSVD
  - `max` (full raw 384): Sin reducción - mejor matices creativos/slang local
- **Default cambiado**: `low` (128 dims) - seguro para SMB típico (100-2000 posts)
- **Ventajas**: Captura slang regional (Almería/andaluz) con `max`, velocidad con `low`
- **Soporte**: Español + Inglés nativamente
- **Descarga**: Automática en primera ejecución desde HuggingFace

#### 📝 Features Manuales (Backup)
- **Análisis de Caption**: longitud, emoji_count, hashtag_count, has_question (regex), has_strong_cta (Comenta, Guarda, DM, Visita, Taggea), lexical_richness.
- **Análisis de Sentimiento**: VADER (nltk) para score emocional (-1 a +1). Contenido emocional/positivo impulsa engagement.
- **Features Temporales**: post_hour, post_day_of_week, is_weekend, is_prime_time.
- **Formato**: One-hot encoding (Reel, Carousel, Static, TikTok).
- **Niche Flags**: Detección de keywords por vertical (inmobiliaria: "casa", "tour", "Triana"; floristería: "flores", "arreglo", "ramo").

> **Arquitectura de Features (Configurable)**: El sistema usa embeddings semánticos configurables:
> - Con `low` (128 dims): 128 caption + 128 transcript + 128 OCR + 10 interacciones + 58 heurísticas = **~452 features** (RECOMENDADO sobremesa normal)
> - Con `max` (384 dims): 384 caption + 384 transcript + 384 OCR + 10 interacciones + 58 heurísticas = **~1220 features** (hardware potente)
> Ambos son seguros para volúmenes SMB típicos (100-2000 posts, train <1min, RAM <1GB con `low`).

### 🎬 Multimodal Late Fusion (Nuevo)

Sistema robusto de fusión multimodal para maximizar predicción de engagement en Reels/TikTok. Combina información de múltiples fuentes (texto, audio transcrito, texto visual) para predicciones más precisas.

#### Arquitectura Late Fusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              MULTIMODAL LATE FUSION PIPELINE (Full Dims by Default)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │ Caption Text │ ──► MiniLM-L6-v2 ──► 384-dim (full) ─────────────────┐   │
│  └──────────────┘                                                       │   │
│                                                                         │   │
│  ┌──────────────┐                                                       │   │
│  │   Whisper    │ ──► MiniLM-L6-v2 ──► 384-dim (full) ─────────────────┼──►│CONCAT│──► XGBoost
│  │  Transcript  │                                                       │   │       │     │
│  └──────────────┘                                                       │   │       │     │
│                                                                         │   │       │     ▼
│  ┌──────────────┐                                                       │   │       │   Score
│  │   EasyOCR    │ ──► MiniLM-L6-v2 ──► 384-dim (full) ─────────────────┤   │       │   0-100
│  │  Visual Text │                                                       │   │       │     +
│  └──────────────┘                                                       │   │       │   SHAP
│                                                                         │   │
│  ┌──────────────┐                                                       │   │
│  │ Heuristics   │ ──► ~58 features (hooks, CTAs, timing, etc.) ────────┤   │
│  └──────────────┘                                                       │   │
│                                                                         │   │
│  ┌──────────────┐                                                       │   │
│  │ Interactions │ ──► 10 cross-modal synergy features ─────────────────┘   │
│  └──────────────┘                                                           │
│                                                                              │
│  Total (max precision): 384 + 384 + 384 + 58 + 10 = ~1220 features          │
│  Reducción opcional: TruncatedSVD a 256 o 128 dims por modalidad            │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### ¿Por qué Late Fusion?

| Aspecto | Ventaja |
|---------|---------|
| **Compatibilidad XGBoost** | Features heterogéneos funcionan perfectamente con árboles de decisión |
| **SHAP Explainability** | Preserva explicabilidad completa - puedes ver qué modalidad contribuye más |
| **Degradación Graceful** | Si falta una modalidad (ej: sin audio), se rellena con zeros sin romper el modelo |
| **Simplicidad** | Sin arquitecturas complejas de atención o transformers end-to-end |
| **Entrenamiento Eficiente** | PCA offline, XGBoost rápido de entrenar |

#### Componentes del Módulo

**Ubicación:** `backend/ml/multimodal_fusion.py`

```python
# Imports principales
from backend.ml.multimodal_fusion import (
    # Función principal de fusión
    fuse_multimodal_features,
    add_multimodal_features_conditional,
    is_video_content,

    # Extractor de embeddings multimodales
    MultimodalEmbeddingExtractor,
    get_multimodal_extractor,

    # Nombres de columnas de features
    TRANSCRIPT_FEATURE_COLUMNS,   # 20 features
    OCR_FEATURE_COLUMNS,          # 20 features
    INTERACTION_FEATURE_COLUMNS,  # 8 features
    MULTIMODAL_FEATURE_COLUMNS,   # 48 features total
)
```

#### Features Generados (Por Modalidad)

**1. Caption Embeddings (384 dims full, o reducido):**
```
embedding_1, embedding_2, ..., embedding_384
# Con precision="medium": embedding_1 a embedding_256
# Con precision="low": embedding_1 a embedding_128
```

**2. Transcript Embeddings (384 dims full, o reducido) - Audio via Whisper:**
```
transcript_emb_1, transcript_emb_2, ..., transcript_emb_384
```

**3. OCR Embeddings (384 dims full, o reducido) - Texto visual via EasyOCR:**
```
ocr_emb_1, ocr_emb_2, ..., ocr_emb_384
```

**4. Cross-Modal Interaction Features (10 dims):**

| Feature | Fórmula | Captura |
|---------|---------|---------|
| `interaction_hook_x_sentiment` | hook_score × sentiment_compound | Hook potente + tono positivo = viral |
| `interaction_hook_x_is_reel` | hook_score × is_reel | Reels benefician más de hooks fuertes |
| `interaction_hook_x_cta_count` | hook_score × cta_count | Hook + múltiples CTAs = máximo engagement |
| `interaction_sentiment_x_cta_count` | sentiment × cta_count | Emoción positiva + CTAs = conversión |
| `interaction_is_reel_x_video_optimal` | is_reel × video_optimal_length | Formato óptimo para Reels (15-60s) |
| `interaction_transcript_richness` | min(len(transcript)/500, 1.0) | Densidad de contenido hablado |
| `interaction_ocr_richness` | min(len(ocr)/100, 1.0) | Cantidad de texto visual overlay |
| `interaction_multimodal_text_density` | min(total_text/1000, 1.0) | Coherencia caption/audio/visual |
| `interaction_semantic_hook_x_vader` | semantic_hook_score × \|sentiment\| | Sinergia hook semántico + emoción |
| `interaction_semantic_hook_x_cta_strong` | semantic_hook_score × has_strong_cta | Hook + CTA fuerte = conversión |

#### Uso Básico

```python
import pandas as pd
from backend.ml.multimodal_fusion import fuse_multimodal_features

# DataFrame con datos de video/reel
df = pd.DataFrame({
    'caption': ['Nuevo Reel! 3 trucos para tu negocio 🔥'],
    'whisper_transcript': ['Hola a todos, hoy les traigo tres trucos que van a cambiar tu negocio...'],
    'easyocr_text': ['3 TRUCOS | NEGOCIO'],
    'hook_score': [0.85],
    'sentiment_compound': [0.6],
    'is_reel': [1],
    'cta_count': [2],
    'video_optimal_length': [1],
    'caption_length': [42],
    'media_type': ['reel'],
})

print(f"Shape antes: {df.shape}")  # (1, 10)

# Aplicar fusión multimodal
df_fused = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=True)

print(f"Shape después: {df_fused.shape}")  # (1, 58) - añade 48 columnas

# Ver features de interacción
print(df_fused['interaction_hook_x_sentiment'].iloc[0])  # 0.51 (0.85 × 0.6)
print(df_fused['interaction_transcript_richness'].iloc[0])  # ~0.15 (74 chars / 500)
```

#### Uso Condicional (Video vs Imagen)

```python
from backend.ml.multimodal_fusion import add_multimodal_features_conditional

# Automáticamente detecta si es video y aplica fusión completa
# Para imágenes/texto solo: añade zeros para consistencia dimensional
df_processed = add_multimodal_features_conditional(df)
```

#### Integración con Pipeline de Training

El módulo se integra automáticamente en `ml/train.py`:

```python
from ml.train import prepare_features

# El pipeline detecta columnas whisper_transcript y easyocr_text
# y aplica fusión multimodal automáticamente
X, y = prepare_features(df)

# X ahora incluye ~136 features:
# - 30 caption embeddings
# - 20 transcript embeddings
# - 20 OCR embeddings
# - 8 interaction features
# - ~58 heuristic features
```

#### Entrenamiento con Datos Multimodales

```bash
# El CSV debe incluir columnas:
# - caption (requerido)
# - whisper_transcript (opcional, para video)
# - easyocr_text (opcional, para video)
# - media_type o is_reel (para detectar video)

python ml/train.py --niche restaurante --data-file data/reels_restaurante.csv
```

**Formato CSV esperado:**
```csv
caption,whisper_transcript,easyocr_text,media_type,engagement_rate,business_type
"Mi Reel viral","Hola, hoy les muestro...","3 TIPS",reel,85.2,restaurante
"POV: cuando el café...","El secreto está en...","CAFÉ PERFECTO",reel,72.1,cafeteria
```

#### SHAP Explainability con Features Multimodales

Las explicaciones SHAP ahora incluyen features multimodales en español:

```json
{
  "score": 78.5,
  "explanation": {
    "explanation_text": "RPI alto por: sinergia hook+sentimiento (+18%), hook potenciado por Reel (+15%), riqueza de transcripción (+12%), formato Reel (+10%), CTA fuerte (+8%)",
    "top_positive_factors": [
      {"feature": "interaction_hook_x_sentiment", "impact": 1.82},
      {"feature": "interaction_hook_x_is_reel", "impact": 1.51},
      {"feature": "interaction_transcript_richness", "impact": 1.23}
    ]
  }
}
```

#### Manejo de Datos Faltantes

El sistema maneja gracefully cuando faltan modalidades:

| Escenario | Comportamiento |
|-----------|----------------|
| Sin `whisper_transcript` | Transcript embeddings = zeros (20 dims) |
| Sin `easyocr_text` | OCR embeddings = zeros (20 dims) |
| Texto vacío `""` | Embeddings = zeros para esa modalidad |
| No es video (imagen/carousel) | Todos multimodal features = zeros |
| Sin PCA entrenado | Fit PCA automático si hay suficientes samples |

#### Configuración de PCA

Los modelos PCA se guardan en `/models/`:

```
models/
├── embedding_svd_64.pkl      # TruncatedSVD para ultra_low (64 dims)
├── embedding_svd_128.pkl     # TruncatedSVD para low (128 dims)
├── embedding_svd_256.pkl     # TruncatedSVD para medium (256 dims)
├── embedding_svd_384.pkl     # TruncatedSVD para high (384 dims)
```

**Fit manual de PCA (opcional):**
```python
from backend.ml.multimodal_fusion import get_multimodal_extractor

extractor = get_multimodal_extractor()

# Fit con corpus de transcripciones
transcripts = ["texto1...", "texto2...", ...]
extractor.fit_transcript_pca(transcripts, save=True)

# Fit con corpus de OCR
ocr_texts = ["TEXTO1", "TEXTO2", ...]
extractor.fit_ocr_pca(ocr_texts, save=True)
```

#### Hiperparámetros XGBoost Ajustados

Para el aumento de ~88 a ~136 features:

```python
# ml/train.py y ml_service.py
xgb.XGBRegressor(
    max_depth=7,          # Aumentado de 6 → 7 para interacciones cross-modal
    n_estimators=100,
    learning_rate=0.1,
    colsample_bytree=0.8,  # Sampling de features para diversidad
    subsample=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
```

#### Tests

```bash
# Ejecutar tests del módulo multimodal
pytest tests/test_multimodal_fusion.py -v

# Tests incluidos:
# - test_import_multimodal_fusion
# - test_feature_column_names
# - test_fuse_multimodal_features_basic
# - test_fuse_without_multimodal_columns
# - test_interaction_features_calculation
# - test_is_video_content_detection
# - test_empty_text_handling
# - test_multimodal_extractor_singleton
```

### 👁️ Video & Audio Analytics (Edge Optimized)
- **Hook Theory Analysis**: Análisis crítico de los primeros 3 segundos (energía visual, cortes, presencia de caras).
- **Text Intelligence**:
  - **Whisper**: Transcripción automática de audio a texto.
  - **EasyOCR**: Detección de texto en pantalla (overlays).
  - **Semantic PCA**: Comprensión profunda del significado del contenido.
- **Optimización Edge**: Procesamiento eficiente con bajo consumo de memoria (stride frames, float32).

### 📈 A/B Testing & Governance
- **A/B Test Logging**: Tabla `ab_test_logs` para tracking de predicciones vs resultados reales.
- **Prediction Logs**: Registro anonimizado (hash de usernames) para auditoría y mejora continua.
- **GDPR Compliance**: Checkbox de consentimiento obligatorio en onboarding.
- **Endpoint de resultados**: `POST /api/v1/abtest/log-result` para reportar engagement real.

### 🎯 Multi-Objective Engagement Prediction (Nuevo)

Sistema de predicción multi-objetivo que permite a cada usuaria configurar los pesos de engagement según sus KPIs de negocio.

#### Arquitectura Multi-Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-OBJECTIVE PREDICTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                      ┌─────────────────────────────────┐  │
│  │   Features   │                      │   Multi-Output Predictions      │  │
│  │   (~136)     │ ──► MultiOutput ──►  │   [log_likes, log_comments,    │  │
│  │              │     XGBRegressor     │    log_shares, log_saves,       │  │
│  └──────────────┘                      │    log_views]                   │  │
│                                        └─────────────┬───────────────────┘  │
│                                                      │                      │
│  ┌──────────────┐                                   ▼                      │
│  │   User KPI   │      Weighted RPI = dot(predictions, weights) / Σweights │
│  │   Weights    │ ─────────────────────────────────────────────────────────│
│  │              │                                   │                      │
│  │ likes: 1.0   │                                   ▼                      │
│  │ comments: 2.0│                      ┌─────────────────────────────────┐  │
│  │ shares: 10.0 │                      │  RPI Score (0-100) + Per-Metric │  │
│  │ saves: 5.0   │                      │  SHAP Explanations              │  │
│  │ views: 3.0   │                      └─────────────────────────────────┘  │
│  └──────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Pesos Configurables por Usuario

Cada métrica tiene un peso configurable (0-20) que determina su importancia en el RPI:

| Métrica | Default | Descripción |
|---------|---------|-------------|
| `likes_weight` | 1.0 | Engagement básico, fácil de obtener |
| `comments_weight` | 2.0 | Indica engagement profundo y comunidad |
| `shares_weight` | 10.0 | Potencial viral, amplificación orgánica |
| `saves_weight` | 5.0 | Intención de compra, contenido valioso |
| `views_weight` | 3.0 | Alcance, especialmente importante para Reels |

#### Templates Predefinidos

| Template | Uso Recomendado | Prioriza |
|----------|-----------------|----------|
| **Brand Awareness** | Negocios nuevos, lanzamientos | Views + Likes |
| **Leads/Conversions** | Inmobiliarias, e-commerce | Saves + Shares |
| **Community** | Restaurantes, cafeterías, locales | Comments |
| **Viral** | Contenido educativo, entretenimiento | Shares |
| **Balanced** | Uso general, A/B testing | Igual peso |

#### Endpoints API

```bash
# Obtener pesos configurados
GET /api/v1/kpi/weights?business_id=1

# Guardar pesos personalizados
POST /api/v1/kpi/weights
{
  "business_id": 1,
  "weights": {
    "likes_weight": 1.0,
    "comments_weight": 3.0,
    "shares_weight": 10.0,
    "saves_weight": 12.0,
    "views_weight": 2.0
  }
}

# Crear desde template
POST /api/v1/kpi/weights/from-template?business_id=1&template_name=leads_conversions

# Preview de impacto
POST /api/v1/kpi/preview
# Muestra cómo los pesos afectan el RPI

# Predicción multi-output
POST /api/v1/multi-output/predict
{
  "caption": "3 dormitorios con vistas al mar...",
  "content_format": "reel",
  "business_id": 1
}
# Devuelve predicciones individuales + RPI ponderado + SHAP por métrica
```

#### Ejemplo de Uso

```python
# Backend: Predicción con pesos personalizados
from app.services.multi_output_predictor import get_multi_output_predictor

predictor = get_multi_output_predictor("inmobiliaria")

# Pesos para inmobiliaria (prioriza leads)
weights = {
    "likes_weight": 1.0,
    "comments_weight": 3.0,
    "shares_weight": 10.0,
    "saves_weight": 12.0,  # Alto: guardados = interés en propiedades
    "views_weight": 2.0,
}

result = predictor.predict(features, weights)

print(f"Predicted likes: {result.predicted_likes:.0f}")
print(f"Predicted saves: {result.predicted_saves:.0f}")
print(f"Weighted RPI: {result.weighted_rpi:.1f}")
# vs RPI con defaults: 58.3 → con leads_conversions: 72.5
```

```typescript
// Frontend: Configurar KPIs
import { kpiApi } from './services/api'

// Cargar template para inmobiliaria
const config = await kpiApi.createFromTemplate(
  businessId,
  "leads_conversions"
)

// O configuración manual
await kpiApi.saveWeights(businessId, {
  likes_weight: 1,
  comments_weight: 3,
  shares_weight: 10,
  saves_weight: 12,
  views_weight: 2,
})
```

#### SHAP Multi-Métrica

El sistema proporciona explicaciones SHAP separadas para cada métrica:

```json
{
  "prediction": {
    "predicted_likes": 150,
    "predicted_saves": 45,
    "weighted_rpi": 72.5
  },
  "shap_likes": {
    "hook_energy": 0.82,
    "is_reel": 0.45,
    "semantic_hook_score": 0.38
  },
  "shap_saves": {
    "has_strong_cta": 0.91,
    "caption_length": 0.52,
    "niche_inmobiliaria": 0.41
  },
  "explanation": "RPI alto (72.5/100). Mayor predicción en likes (150). Por encima del promedio del autor (1.3x)."
}
```

#### Backward Compatibility

El sistema es **100% retrocompatible**:
- Si un usuario no ha configurado pesos, se usan los defaults automáticamente
- El endpoint `/api/v1/growth/predict` sigue funcionando igual
- Los modelos single-output existentes siguen válidos

### 🎯 Configurable Embedding Precision (Actualizado)

Sistema de precisión de embeddings configurable por usuario/negocio. Permite ajustar el trade-off entre precisión semántica y rendimiento.

**IMPORTANTE**: Precision es REQUERIDO - no hay default automático. El caller debe elegir conscientemente.

#### Niveles de Precisión

| Nivel | Dims | Descripción | Uso Recomendado |
|-------|------|-------------|-----------------|
| **ultra_low** | 64 | Ultra rápido | PC modesto, datasets muy grandes (>10000 posts) |
| **low** (RECOMENDADO) | 128 | Balance seguro | **Sobremesa normal Almería** (100-2000 posts) |
| **medium** | 256 | Balance precision/velocidad | Datasets 2000-5000 posts |
| **high** | 384 | Full dims con TruncatedSVD | Precisión máxima con reducción |
| **max** | full raw 384 | Sin reducción | Hardware potente, mejor matices creativos/slang |

#### ¿Por qué "low" como Recomendado?

```
┌─────────────────────────────────────────────────────────────────────┐
│           BENCHMARK: 500 samples, sobremesa normal Almería           │
├─────────────────────────────────────────────────────────────────────┤
│  Precision    Dims    Time      RAM       Features   Recomendado     │
│  ─────────────────────────────────────────────────────────────────   │
│  ultra_low    64      0.8s      ~0.3GB    ~260       PC modesto      │
│  low          128     1.2s      ~0.5GB    ~452       SOBREMESA ✓     │
│  medium       256     1.8s      ~1.0GB    ~836       Balance         │
│  high         384     2.1s      ~1.5GB    ~1220      High-end        │
│  max          384     2.1s      ~2.0GB    ~1220      Hardware top    │
└─────────────────────────────────────────────────────────────────────┘

Conclusión: "low" (128 dims) es RECOMENDADO para sobremesa normal.
- Training < 30 segundos
- RAM < 0.5GB
- Buen balance precisión/velocidad
- Seguro para volúmenes SMB típicos (100-2000 posts)
```

#### API Endpoints

```bash
# Obtener opciones disponibles
GET /api/v1/embeddings/info

# Obtener configuración actual de un negocio
GET /api/v1/embeddings/{business_id}

# Actualizar precisión
PUT /api/v1/embeddings/{business_id}
{
  "precision": "max"  // "low", "medium", "high", "max"
}

# Obtener benchmark estimado
GET /api/v1/embeddings/{business_id}/benchmark?n_samples=500
```

#### Configuración en Business Model

```python
# backend/app/models/business.py
class EmbeddingPrecision(str, Enum):
    LOW = "low"        # 128 dims
    MEDIUM = "medium"  # 256 dims
    HIGH = "high"      # 384 dims
    MAX = "max"        # 384 dims (default)

class Business(Base):
    # ...
    embedding_precision = Column(
        Enum(EmbeddingPrecision),
        default=EmbeddingPrecision.MAX
    )
```

#### CLI Training con Precisión

```bash
# Training con full dims (default - recomendado)
python ml/train.py --niche restaurante --data-file data.csv

# Training con precisión reducida (datasets muy grandes)
python ml/train.py --niche restaurante --data-file data.csv --precision medium

# Benchmark para confirmar rendimiento
python ml/embedding_benchmark.py --samples 500 --all-precisions
```

#### Componente Frontend

```tsx
import { EmbeddingConfiguration } from './components/EmbeddingConfiguration'

<EmbeddingConfiguration
  businessId={business.id}
  onConfigChange={(precision) => console.log('Changed:', precision)}
/>
```

El componente muestra:
- Radio buttons: Baja / Media / Alta / Máxima (Recomendada)
- Tooltip: "Full dims captura mejor slang Almería/andaluz"
- Estimaciones de RAM/tiempo por opción
- Indicador de configuración actual

#### Componente Frontend

El componente `KPIConfiguration.tsx` proporciona:
- Sliders para cada peso (1-20)
- Selección rápida de templates con iconos
- Preview en tiempo real del RPI resultante
- Barra visual de distribución de pesos
- Recomendaciones específicas por tipo de negocio

```tsx
import { KPIConfiguration } from './components/KPIConfiguration'

<KPIConfiguration
  businessId={business.id}
  businessType="inmobiliaria"
  onWeightsSaved={(weights) => console.log('Saved:', weights)}
/>
```

### 🔧 Configuración Unificada de Usuario (Nuevo)

Sistema centralizado para gestionar TODAS las configuraciones de usuario que afectan los pipelines ML (ingest/train/inference). Elimina hardcodes y garantiza sincronización real-time entre frontend y backend.

#### Arquitectura de Sincronización

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              UNIFIED USER CONFIG - REAL-TIME SYNC                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                      ┌─────────────────────────────────┐  │
│  │   Frontend   │  POST /user/config   │        UserConfigService        │  │
│  │   Dashboard  │ ────────────────────►│   (Central Source of Truth)     │  │
│  │              │                      │                                 │  │
│  │ - Embedding  │◄─────────────────────│   ┌─────────────────────────┐  │  │
│  │ - KPI Weights│  GET /user/config    │   │    UserConfig Model     │  │  │
│  │ - Multimodal │                      │   │   (DB Persistence)      │  │  │
│  │ - Own Profile│                      │   └─────────────────────────┘  │  │
│  └──────────────┘                      │              │                  │  │
│                                        └──────────────┼──────────────────┘  │
│                                                       │                      │
│                                                       ▼                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    ALL BACKEND PIPELINES USE CONFIG                     │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ingest.py ──────► own_username (feedback detection)                    │ │
│  │                    multimodal_mode (light/full processing)              │ │
│  │                                                                          │ │
│  │  ml_service.py ──► embedding_precision (64-384 dims)                    │ │
│  │                    kpi_weights (RPI calculation)                        │ │
│  │                                                                          │ │
│  │  growth_engine ──► embedding_precision + kpi_weights                    │ │
│  │                                                                          │ │
│  │  train.py ───────► embedding_precision for training                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Critical Logging: "User config loaded: precision={X}, multimodal={Y},      │
│                     own=@{Z}"                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Campos de Configuración

| Campo | Tipo | Descripción | Afecta a |
|-------|------|-------------|----------|
| `embedding_precision` | Enum | ultra_low/low/medium/high/max | ML Service, Train, Growth Engine |
| `multimodal_mode` | Enum | light (~5s) / full (~20s) | Ingest Pipeline |
| `kpi_weights` | Object | likes, comments, shares, saves, views (0-20) | RPI Calculation |
| `own_instagram_username` | String | Tu username (sin @) | Human-in-the-Loop Feedback |
| `own_tiktok_username` | String | Tu username TikTok | Human-in-the-Loop Feedback |
| `discovery` | Object | hashtags, location, niche keywords | Discovery Script |
| `light_mode_config` | Object | whisper_model, ocr_max_frames | Multimodal Processing |

#### API Endpoints

```bash
# Obtener config actual (crea con defaults si no existe)
GET /api/v1/user/config?business_id=1

# Respuesta:
{
  "id": 1,
  "user_id": 1,
  "business_id": 1,
  "embedding_precision": "low",
  "embedding_dims": 128,
  "multimodal_mode": "light",
  "kpi_weights": {
    "likes": 1.0,
    "comments": 2.0,
    "shares": 10.0,
    "saves": 5.0,
    "views": 3.0
  },
  "own_instagram_username": "mi_negocio",
  "discovery": {
    "hashtags": ["inmobiliariaalmeria"],
    "location_keywords": ["almeria"]
  },
  "is_active": true
}

# Guardar/actualizar config
POST /api/v1/user/config?business_id=1
{
  "embedding_precision": "medium",
  "kpi_weights": {
    "likes": 1.0,
    "comments": 3.0,
    "shares": 10.0,
    "saves": 12.0,
    "views": 2.0
  },
  "own_instagram_username": "mi_inmobiliaria"
}

# Obtener opciones disponibles (para dropdowns)
GET /api/v1/user/config/info

# Respuesta:
{
  "precision_options": [
    {"value": "ultra_low", "label": "Ultra Baja (64 dims)", "dimensions": 64, ...},
    {"value": "low", "label": "Baja (128 dims) - Recomendada", "is_recommended": true},
    ...
  ],
  "multimodal_options": [...],
  "kpi_templates": [
    {"name": "brand_awareness", "display_name": "Brand Awareness", ...},
    {"name": "leads", "display_name": "Generación de Leads", ...},
    ...
  ]
}

# Validar config sin guardar (preview)
POST /api/v1/user/config/validate
{
  "embedding_precision": "max",
  "kpi_weights": {"likes": 0, "comments": 0, "shares": 0, "saves": 0, "views": 0}
}

# Respuesta:
{
  "valid": false,
  "warnings": ["Todos los pesos KPI son 0 - RPI será siempre 0"],
  "info": ["Precisión 'max' requiere más RAM (~2GB)"]
}

# Resetear a defaults
DELETE /api/v1/user/config?business_id=1

# Config para pipelines (uso interno)
GET /api/v1/user/config/pipeline?business_id=1
```

#### Uso en Pipelines (Backend)

```python
# En cualquier pipeline: ingest.py, ml_service.py, etc.
from app.services.user_config_service import get_pipeline_config

# Cargar config del usuario
config = await get_pipeline_config(user_id=1, business_id=1, db=session)

# Log crítico (siempre presente)
logger.info(f"User config loaded: precision={config.embedding_precision}, "
            f"multimodal={config.multimodal_mode}, own=@{config.own_instagram_username}")

# Usar en embeddings
from ml.features_embeddings import EmbeddingExtractor
extractor = EmbeddingExtractor(precision=config.embedding_precision)

# Usar en RPI
rpi = config.get_weighted_rpi(
    likes=100, comments=20, shares=5, saves=10, views=1000
)

# Detectar perfil propio (Human-in-the-Loop)
is_own = config.is_own_profile("mi_negocio", platform="instagram")
if is_own:
    # Añadir a dataset de feedback para reentrenamiento
    mark_as_feedback_sample(post)
```

#### PipelineConfig Schema

El schema `PipelineConfig` proporciona acceso simplificado a la config en pipelines:

```python
class PipelineConfig:
    # Identificación
    user_id: int
    business_id: int

    # Embeddings
    embedding_precision: str  # "low", "medium", "high", "max"
    embedding_dims: int       # 64, 128, 256, 384

    # Multimodal
    multimodal_mode: str      # "light", "full"
    light_mode_enabled: bool
    whisper_model: str        # "tiny", "base"
    ocr_max_frames: int

    # KPI Weights
    kpi_weights: Dict[str, float]            # Raw weights
    kpi_weights_normalized: Dict[str, float]  # Suma = 1.0

    # Own profile (para feedback loop)
    own_instagram_username: Optional[str]
    own_tiktok_username: Optional[str]

    # Discovery
    discovery_hashtags: List[str]
    discovery_niche_keywords: List[str]

    # Helper methods
    def is_own_profile(self, username: str, platform: str) -> bool
    def get_weighted_rpi(self, likes, comments, shares, saves, views) -> float
```

#### Componente Frontend

```tsx
import { UnifiedConfiguration } from './components/UnifiedConfiguration'

<UnifiedConfiguration
  businessId={business.id}
  userId={currentUser.id}
  onConfigSaved={(config) => {
    console.log('Config guardada:', config)
    // Todos los pipelines ahora usan la nueva config
  }}
/>
```

El componente `UnifiedConfiguration.tsx` incluye:
- **Sección Perfil Propio**: Input para @username de Instagram/TikTok
- **Sección Embedding**: Radio buttons con dims y estimaciones de RAM
- **Sección KPI Weights**: Sliders 0-20 + templates predefinidos
- **Sección Multimodal**: Toggle light/full con tiempos estimados
- **Sección Discovery**: Tags para hashtags, location, niche keywords
- **Validación en tiempo real** antes de guardar
- **Indicadores visuales** de config actual vs cambios pendientes

#### Human-in-the-Loop Feedback

El campo `own_instagram_username` habilita el feedback loop automático:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HUMAN-IN-THE-LOOP FEEDBACK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Usuario configura @mi_negocio en UnifiedConfiguration                   │
│                                                                              │
│  2. Ingest Pipeline detecta automáticamente posts propios:                  │
│     if config.is_own_profile(post.username):                               │
│         post.is_feedback_sample = True                                      │
│                                                                              │
│  3. Sistema compara predicción vs engagement real:                          │
│     predicted_rpi = 72.5                                                    │
│     actual_rpi = 85.3  ← calculado con métricas reales                     │
│     delta = +17.7%                                                          │
│                                                                              │
│  4. Si delta > 10%, sample se prioriza para reentrenamiento:                │
│     - Online learning: update inmediato (River)                            │
│     - Batch: cola para próximo retrain (XGBoost)                           │
│                                                                              │
│  5. Modelo mejora con datos REALES del propio usuario                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Cache y Rendimiento

El servicio incluye cache en memoria con TTL de 60 segundos:

```python
# Cache automático para configs frecuentes
class UserConfigService:
    _config_cache: Dict[str, tuple] = {}
    _cache_ttl_seconds = 60

    # Se invalida automáticamente cuando se guarda nueva config
    def _invalidate_cache(self, user_id, business_id):
        ...
```

Esto permite:
- Múltiples pipelines leyendo la misma config sin queries repetidas
- Latencia <1ms para lecturas cacheadas
- Consistencia garantizada (cache se invalida en cada update)

### 📊 Análisis de Competidores
- Scraping automático de Instagram, TikTok y LinkedIn via **Apify API**.
- Extracción de patrones ganadores: Hooks, CTAs, Pilares de contenido.
- Identificación de tendencias de audio y hashtags.

### 📅 Generador de Calendarios Inteligente
- Calendarios mensuales optimizados por IA.
- Scripts detallados con guía de filmación paso a paso.
- **Predicción de Engagement**: Score 0-100 para cada pieza antes de publicarla.

### 🔥 Scanner Viral
- Búsqueda de tendencias en tiempo real.
- Adaptación de trends globales a negocios locales.
- Generación de ideas reactivas (POVs, newsjacking).

## Tipos de Negocio Soportados

- Inmobiliarias (Real Estate)
- Floristerías
- Cafeterías
- Peluquerías
- Tiendas Locales
- Restaurantes
- Gimnasios
- Clínicas
- Y más...

## Instagram Discovery Script

> **Descubre perfiles de Instagram relevantes para tu nicho local**

El script `apify_instagram_discovery.py` permite descubrir competidores y perfiles similares en Instagram buscando por hashtags, ubicación y nicho. Genera una lista CSV/JSON de perfiles para que la usuaria los revise manualmente.

### Flujo de Trabajo

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Discovery     │     │   Elena Bridge  │     │   BrandPulse    │
│   Script        │────▶│   Extension     │────▶│   ML Analysis   │
│   (perfiles)    │     │   (contenido)   │     │   (predicción)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Discovery Script**: Encuentra perfiles relevantes por hashtags
2. **Elena Bridge**: Extrae posts/reels de los perfiles que te interesen
3. **BrandPulse ML**: Analiza el contenido y predice engagement

### Modos de Uso

```bash
# 1. Modo interactivo (te pregunta qué buscar)
python backend/scripts/apify_instagram_discovery.py

# 2. Con archivo de configuración guardado
python backend/scripts/apify_instagram_discovery.py --config mi_busqueda.yaml

# 3. Con argumentos directos
python backend/scripts/apify_instagram_discovery.py \
    --hashtags "inmobiliariaalmeria,casasalmeria" \
    --location "almería,roquetas,aguadulce" \
    --niche "inmobiliaria,casas,pisos,alquiler"

# 4. Guardar configuración para reutilizar
python backend/scripts/apify_instagram_discovery.py --save-config floristerias.yaml
```

### Opciones CLI

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--hashtags` | Hashtags a buscar (separados por comas) | - |
| `--location` | Keywords de ubicación | - |
| `--niche` | Keywords del nicho de negocio | - |
| `--min-followers` | Mínimo de seguidores | 100 |
| `--max-followers` | Máximo de seguidores | 50000 |
| `--max-posts` | Posts a buscar por hashtag | 150 |
| `--max-profiles` | Máximo perfiles en output | 150 |
| `--no-enrich` | Desactivar enriquecimiento | False |
| `--config` | Cargar desde archivo YAML/JSON | - |
| `--save-config` | Guardar config para reutilizar | - |
| `-i, --interactive` | Forzar modo interactivo | False |

### Archivo de Configuración (YAML)

```yaml
# mi_busqueda.yaml
hashtags:
  - inmobiliariaalmeria
  - casasalmeria
  - pisosalmeria

location_keywords:
  - almería
  - roquetas
  - aguadulce

niche_keywords:
  - inmobiliaria
  - casas
  - pisos
  - alquiler

min_followers: 100
max_followers: 50000
max_posts_per_hashtag: 150
max_profiles_output: 150
enrich_profiles: true
```

### Output

El script genera:
- `discovery_output/instagram_discovery_YYYYMMDD_HHMMSS.csv`
- `discovery_output/instagram_discovery_YYYYMMDD_HHMMSS.json`

Con columnas: `username`, `followers`, `bio`, `category`, `relevance_score`, etc.

---

## Elena Bridge - Chrome Extension

> **Extensión de Chrome para guardar contenido de Instagram y TikTok directamente al backend**

Elena Bridge permite a tu clienta navegar por Instagram y TikTok, encontrar contenido que le guste, y guardarlo con un solo clic para análisis posterior.

### Funcionalidades

- **Detección automática** de posts, reels y videos
- **Botón flotante contextual**:
  - "Guardar Post" en posts de Instagram
  - "Guardar Reel" en reels de Instagram
  - "Guardar Video" en videos de TikTok
  - "Analizar Perfil" en páginas de perfil
- **Extracción "stealth"**: Lee solo el DOM visible, sin hacer peticiones a APIs de redes sociales
- **Datos hidratados**: Prioriza datos del servidor (window._sharedData, SIGI_STATE) para máxima precisión

### Datos Extraídos

Para cada contenido se extrae:
- **Autor**: username, nombre, foto de perfil, verificación
- **Contenido**: caption, hashtags, menciones
- **Métricas**: likes, comentarios, vistas, shares, guardados
- **Media**: URLs de imágenes/videos, thumbnails, duración
- **Audio**: título, artista, si es original (para reels/videos)
- **Timestamp**: fecha de publicación

### Instalación de la Extensión

```bash
# 1. Instalar dependencias
cd extension
npm install

# 2. Compilar para desarrollo (con hot reload)
npm run dev

# 3. O compilar para producción
npm run build
```

**Cargar en Chrome:**
1. Ir a `chrome://extensions`
2. Activar "Modo desarrollador" (esquina superior derecha)
3. Click en "Cargar descomprimida"
4. Seleccionar la carpeta `extension/dist`

### Estructura de la Extensión

```
extension/
├── manifest.json              # Manifest V3
├── package.json               # Vite + React + TypeScript + CRXJS
├── vite.config.ts
├── src/
│   ├── content/
│   │   ├── injector.ts        # Content script (inyecta botón)
│   │   └── styles.css         # Estilos del botón flotante
│   ├── background/
│   │   └── service-worker.ts  # Comunicación con API backend
│   ├── popup/
│   │   ├── Popup.tsx          # UI del popup
│   │   └── styles.css
│   ├── utils/
│   │   ├── instagram-extractor.ts         # Perfiles IG
│   │   ├── instagram-content-extractor.ts # Posts/Reels IG
│   │   ├── tiktok-extractor.ts           # Perfiles TikTok
│   │   └── tiktok-content-extractor.ts   # Videos TikTok
│   └── types/
│       └── index.ts           # Tipos TypeScript
└── public/icons/              # Iconos de la extensión
```

### API Endpoint para la Extensión

La extensión envía datos a `POST /api/ingest/raw`:

```json
{
  "source": "elena_bridge_extension",
  "version": "1.0.0",
  "type": "content",
  "content": {
    "platform": "instagram",
    "contentType": "reel",
    "contentId": "ABC123",
    "author": { "username": "ejemplo", "isVerified": true },
    "caption": "Mi contenido viral...",
    "hashtags": ["#viral", "#trending"],
    "metrics": { "likes": 50000, "comments": 1200 },
    "audio": { "title": "Trending Sound", "artist": "Artist" }
  }
}
```

## Tech Stack

### Backend
- **Python 3.11** (Requerido)
- **FastAPI** - Framework web async
- **SQLAlchemy 2.0** - ORM con soporte async
- **SQLite/PostgreSQL** - Base de datos
- **XGBoost** - Modelo de predicción batch
- **River** - Online/incremental learning (AdaptiveRandomForest)
- **SHAP** - Explicabilidad de predicciones
- **Grok (xAI)** - Motor LLM principal para análisis creativo y generación de contenido
- **sentence-transformers** - Embeddings semánticos (all-MiniLM-L6-v2)
- **NLTK VADER** - Análisis de sentimiento
- **Apify Client** - Scraping de redes sociales
- **NumPy/Pandas/Scikit-learn** - Procesamiento de datos y PCA

### Frontend
- **React 18** con TypeScript
- **Vite** - Build tool
- **Tailwind CSS** - Estilos
- **Zustand** - State management
- **React Query** - Data fetching

### Extension (Elena Bridge)
- **Vite + CRXJS** - Build con hot reload para Chrome extensions
- **React 18 + TypeScript** - UI del popup
- **Manifest V3** - Última versión del manifiesto de Chrome
- **Content Scripts** - Inyección de botón flotante
- **Service Worker** - Comunicación con backend API

## Instalación

### Prerrequisitos

- **Python 3.11** (Estrictamente requerido)
- Node.js 18+
- npm o yarn

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/brandpulse-ai.git
cd brandpulse-ai
```

### 2. Configurar el Backend

```bash
cd backend

# Crear entorno virtual con Python 3.11
# Asegúrate de tener python 3.11 instalado
py -3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# o en Windows: venv\Scripts\activate

# Instalar dependencias (incluye librerías de ML pesadas)
pip install -r requirements.txt

# Descargar datos de NLTK para VADER
python -c "import nltk; nltk.download('vader_lexicon')"

# Configurar variables de entorno
cp .env.example .env
# Edita .env con tus claves API
```

### 3. Configurar el Frontend

```bash
cd frontend

# Instalar dependencias
npm install

# Configurar variables de entorno
cp .env.example .env.local
```

### 4. Configurar las Claves API

Edita `backend/.env`:

```env
# REQUERIDO: Grok (xAI) - Motor LLM principal
# Obtén tu clave en https://console.x.ai
GROK_API_KEY=xai-tu-clave-aqui
GROK_MODEL=grok-4-1-fast-reasoning

# REQUERIDO: Apify para scraping
APIFY_API_KEY=apify_api_tu-clave-aqui

# Base de Datos
DATABASE_URL=sqlite+aiosqlite:///./brandpulse.db

# Seguridad
SECRET_KEY=tu-clave-secreta-de-32-caracteres
```

## ▶️ Ejecutar la Aplicación

### Iniciar el Backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

El backend estará disponible en: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Iniciar el Frontend

```bash
cd frontend
npm run dev
```

El frontend estará disponible en: http://localhost:5173

## Guía de Uso

### 1. Onboarding y Competidores
Configura tu negocio y añade competidores. Acepta el consentimiento GDPR para el procesamiento de datos. El sistema iniciará el scraping y el **Pattern Extractor** analizará miles de posts para encontrar qué funciona en tu nicho.

### 2. Análisis de Drafts
Usa la herramienta de "Analizar Draft" para pasar tu idea por el **Growth Prediction Engine**.
- Obtendrás un **RPI Score** predicho.
- Verás explicaciones SHAP detalladas con Top 5 factores: *"RPI alto por: pregunta en caption (+22%), hora 20:00 (+18%), formato Reel (+15%)"*.

### 3. Generación de Contenido
Genera calendarios completos donde cada post ha sido optimizado por **Grok** siguiendo los patrones detectados y validado por los modelos de ML.

### 4. A/B Testing
Registra los resultados reales de tus publicaciones para mejorar las predicciones:
- El sistema compara predicciones vs engagement real.
- Muestras con >20% de diferencia se priorizan para reentrenamiento.
- Las predicciones mejoran continuamente con datos reales.

### 5. Viral Scanner
Detecta tendencias emergentes y genera scripts adaptados a tu negocio usando la inteligencia semántica del sistema.

## 🔌 API Endpoints Principales

### ML & Growth
- `POST /api/v1/growth/predict` - Predicción de RPI Score con explicación SHAP
- `POST /api/v1/ml/analyze-draft` - Análisis completo de borrador (Score + Roadmap de mejora)
- `POST /api/v1/ml/predict/format` - Recomendación de formato (Reel vs Carousel)

### A/B Testing & Feedback
- `POST /api/v1/abtest/log-result` - Registrar resultado real de contenido publicado
- `POST /api/v1/abtest/feedback` - Feedback con online learning automático (River)
- `GET /api/v1/abtest/online-status` - Estado del modelo online por nicho
- `GET /api/v1/abtest/stats` - Estadísticas de predicciones vs realidad

### Contenido & Análisis
- `POST /api/v1/content/{business_id}/generate-calendar` - Generar calendario con IA
- `GET /api/v1/competitors/{business_id}/{competitor_id}/analysis` - Ver patrones extraídos

### Elena Bridge (Extensión)
- `POST /api/ingest/raw` - Ingestar contenido desde la extensión (posts, reels, videos, perfiles)
- `GET /api/ingest/status` - Verificar estado del endpoint de ingesta

### User Config (Configuración Unificada)
- `GET /api/v1/user/config` - Obtener config actual del usuario (crea defaults si no existe)
- `POST /api/v1/user/config` - Guardar/actualizar configuración
- `GET /api/v1/user/config/info` - Obtener opciones disponibles (precision, multimodal, templates)
- `GET /api/v1/user/config/pipeline` - Config optimizada para pipelines (uso interno)
- `POST /api/v1/user/config/validate` - Validar config sin guardar (preview)
- `DELETE /api/v1/user/config` - Resetear a valores por defecto

### KPI Weights (Multi-Objetivo)
- `GET /api/v1/kpi/weights` - Obtener pesos configurados para un negocio
- `POST /api/v1/kpi/weights` - Guardar/actualizar pesos personalizados
- `GET /api/v1/kpi/templates` - Listar templates disponibles (brand_awareness, leads, etc.)
- `POST /api/v1/kpi/preview` - Preview del impacto de pesos en RPI
- `POST /api/v1/kpi/weights/from-template` - Crear configuración desde template

### Multi-Output Predictions
- `POST /api/v1/multi-output/predict` - Predicción multi-métrica con pesos configurables
- `POST /api/v1/multi-output/predict/batch` - Predicciones batch
- `GET /api/v1/multi-output/status` - Estado del modelo multi-output
- `POST /api/v1/multi-output/train` - Entrenar modelo multi-output
- `POST /api/v1/multi-output/compare-weights` - Comparar diferentes configuraciones de pesos

### Model Health (Nuevo)
- `GET /api/v1/ml/health/summary` - Resumen de salud del modelo (para dashboard)
- `GET /api/v1/ml/health` - Métricas detalladas por niche
- `GET /api/v1/ml/health?niche=X` - Métricas específicas de un niche con historial

### Autenticación & Negocio
- `POST /api/v1/auth/login`
- `POST /api/v1/business/onboard`

## 🧪 Modelos ML

El sistema utiliza modelos entrenados específicamente para redes sociales:
- **Engagement Model (XGBoost)**: Predice likes/comments/shares con métricas R², RMSE, MAE.
- **Growth Model (XGBoost)**: Predice RPI (Relative Performance Index).
- **Format Classifier (Random Forest)**: Recomienda el mejor formato con ROC-AUC para clasificación binaria.
- **Sentiment Analyzer (VADER)**: Detecta tono emocional para optimizar engagement.

### Cold Start Handling

El sistema implementa una estrategia robusta para nichos nuevos con pocos datos:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ESTRATEGIA COLD START                            │
├─────────────────────────────────────────────────────────────────────┤
│  Samples < 30      → Usar modelo base directamente (no fine-tune)   │
│  Samples 30-299    → Fine-tune desde modelo base (cold start)       │
│  Samples >= 300    → Entrenar from scratch (datos suficientes)      │
└─────────────────────────────────────────────────────────────────────┘
```

**Flujo de modelos:**
1. **Modelo Base**: Preentrenado en 10k samples sintéticos cubriendo todos los nichos
2. **Modelos por Nicho**: Fine-tuned con datos específicos del vertical
3. **Fallback automático**: Si no hay modelo de nicho, usa el base

**Prioridad de selección:**
```python
1. Modelo específico del nicho (niche_restaurante.pkl)
2. Modelo principal entrenado (engagement_model.joblib)
3. Modelo base preentrenado (base_xgboost.pkl)  # Cold start
4. Predicción heurística (reglas básicas)
```

### Scripts ML (/ml)

#### 0. Módulo de Embeddings Semánticos (Nuevo)

El archivo `ml/features_embeddings.py` proporciona embeddings modernos basados en NLP:

```python
from ml.features_embeddings import (
    get_caption_embedding,      # Raw embedding (dims segun precision)
    get_embedding_features,     # Dict para ML (dims segun precision)
    EmbeddingExtractor          # Clase completa
)

# Uso simple - IMPORTANTE: precision es REQUERIDO
embedding = get_caption_embedding("Nuevo apartamento en Triana!", precision="low")
# numpy array (128,) con precision="low"

features = get_embedding_features("Nuevo apartamento en Triana!", precision="low")
# {"embedding_0": 0.123, "embedding_1": -0.456, ..., "embedding_127": 0.789}
# 0-based indexing: embedding_0 a embedding_{N-1}
```

**Configuración del modelo:**
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Modelo | `all-MiniLM-L6-v2` | Lightweight, ~80MB |
| Dimensión raw | 384 | Output del transformer |
| Dimensión configurable | 64-384 | Según precision (ultra_low=64, low=128, medium=256, high=384, max=384 raw) |
| Max tokens | 256 | Truncamiento automático |
| Default recomendado | `low` (128) | Sobremesa normal Almería |

**Ajuste de TruncatedSVD en corpus de training:**
```python
from ml.features_embeddings import EmbeddingExtractor

# IMPORTANTE: precision es REQUERIDO
extractor = EmbeddingExtractor(precision="low")

# Opción 1: Desde textos
extractor.fit_reducer_from_texts(["caption1", "caption2", ...], save=True)

# Opción 2: Desde embeddings raw
embeddings = extractor.get_raw_embeddings_batch(texts)
extractor.fit_reducer(embeddings, save=True)

# El modelo se guarda en: models/embedding_svd_128.pkl (para precision="low")
```

**Integración automática:**
- `FeatureExtractor.extract_features()` añade automáticamente `embedding_0` a `embedding_{N-1}`
- 0-based indexing para consistencia con XGBoost
- `train.py` genera embeddings con precision configurable
- `pretrain_base_model.py` genera embeddings sintéticos para pretraining

#### 1. Preentrenar Modelo Base

```bash
# Generar datos sintéticos y entrenar modelo base (una vez)
python ml/pretrain_base_model.py

# Con más samples para mejor generalización
python ml/pretrain_base_model.py --samples 20000

# Output: models/base_xgboost.pkl
```

**Features del dataset sintético (variable según precision):**

| Categoría | Features | Count (low=128) |
|-----------|----------|-----------------|
| **Embeddings** | `embedding_0` a `embedding_{N-1}` (0-based indexing) | 128 (o 64/256/384 según precision) |
| **Caption** | `caption_length`, `caption_words`, `emoji_count`, `hashtag_count`, `lexical_richness` | 10 |
| **Hooks** | `hook_question`, `hook_pov`, `hook_number`, `hook_bold_claim`, `hook_story` | 7 |
| **Triggers** | `trigger_urgency`, `trigger_curiosity`, `trigger_action`, `trigger_emotion` | 8 |
| **CTAs** | `cta_comment`, `cta_save`, `cta_share`, `cta_follow`, `cta_dm`, `cta_link` | 7 |
| **Formato** | `is_reel`, `is_carousel`, `is_static`, `video_duration`, `has_audio` | 6 |
| **Timing** | `hour_of_day`, `day_of_week`, `is_prime_time`, `is_weekend` | 4 |
| **Nicho** | `niche_inmobiliaria`, `niche_cafeteria`, `niche_restaurante`, etc. | 8 |
| **Sentiment** | `sentiment_compound`, `sentiment_positive`, `sentiment_negative` | 4 |

> **Total features** varía según precision: ~182 (ultra_low), ~246 (low), ~374 (medium), ~502 (high/max)

- `engagement_rate` (target): distribución log-normal realista

#### 2. Entrenar Modelo por Nicho

```bash
# Entrenar/fine-tune para un nicho específico
python ml/train.py --niche restaurante --data-file data/restaurante_posts.csv

# Fine-tune forzado (incluso con suficientes datos)
python ml/train.py --niche cafeteria --data-file data.csv --force-finetune

# Threshold personalizado
python ml/train.py --niche floristeria --data-file data.csv --threshold 200
```

#### 3. Entrenar Modelo Multi-Output (Nuevo)

```bash
# Entrenar modelo multi-output para predicción de múltiples métricas
python ml/train_multi_output.py --niche restaurante --data-file data/posts.csv

# Entrenar modelo base multi-output (sin filtro de nicho)
python ml/train_multi_output.py --base-model --data-file data/all_posts.csv
```

**Targets del modelo multi-output:**
- `log_likes` - log(likes + 1)
- `log_comments` - log(comments + 1)
- `log_shares` - log(shares + 1)
- `log_saves` - log(saves + 1)
- `log_views` - log(views + 1)

**Métricas por target:**
```
PER-TARGET METRICS
--------------------------------------------------
likes       : RMSE=0.4521, R2=0.7823, MAE=0.3211
comments    : RMSE=0.5123, R2=0.7156, MAE=0.3892
shares      : RMSE=0.6234, R2=0.6543, MAE=0.4521
saves       : RMSE=0.4892, R2=0.7412, MAE=0.3567
views       : RMSE=0.5567, R2=0.6891, MAE=0.4123
--------------------------------------------------
COMBINED    : RMSE=0.5267, R2=0.7165
```

**CSV requerido para multi-output:**
```csv
caption,likes,comments,shares,saves,views,business_type
"Mi Reel viral",1500,45,12,89,15000,restaurante
"Nuevo plato del día",800,22,5,34,8000,restaurante
```

**Fine-tuning (cold start):**
- Learning rate bajo: 0.01 (vs 0.1 normal)
- Solo 10-20 rounds adicionales
- Early stopping para evitar overfitting
- Preserva conocimiento del modelo base

### Datasets Externos (Kaggle)

Como alternativa a los datos sintéticos, puedes usar datasets reales de Kaggle:

| Dataset | Descripción | URL |
|---------|-------------|-----|
| Instagram Analytics Dataset | Métricas de posts de IG | [Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset) |
| Instagram Reach Forecasting | Análisis de alcance | [Kaggle](https://www.kaggle.com/datasets/rahulchavan99/instagram-reach-forecasting) |
| Social Media Engagement Metrics | Engagement multi-plataforma | [Kaggle](https://www.kaggle.com/datasets/purnisharma/social-media-engagement-metrics) |
| Instagram Analysis | Interacciones de usuarios | [Kaggle](https://www.kaggle.com/datasets/shubhamsadawarti/instagram-analysis) |

**Nota:** Los datasets de Kaggle pueden requerir preprocesamiento para alinear features con nuestro sistema. El dataset sintético está diseñado específicamente para features de SMBs locales en español (triggers, CTAs, hooks) que no suelen estar en datasets públicos.

Para usar un dataset de Kaggle:
```bash
# Descargar y preparar
kaggle datasets download -d kundanbedmutha/instagram-analytics-dataset
python scripts/prepare_kaggle_data.py --input instagram_data.csv --output prepared_data.csv

# Entrenar con datos reales
python ml/pretrain_base_model.py --data-file prepared_data.csv
```

### Reentrenamiento
- Time-based train/test split para evitar data leakage.
- Evaluación vs baseline (media histórica) para medir mejora real.
- Feedback loop con Human-in-the-Loop para aprendizaje continuo.

### 🔄 Online/Incremental Learning con River (Nuevo)

Sistema de aprendizaje incremental que actualiza el modelo en tiempo real cuando llega feedback de engagement sin necesidad de reentrenamiento batch completo.

#### Arquitectura Online Learning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE LEARNING PIPELINE (River)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌─────────────────────┐     ┌────────────────────┐   │
│  │   A/B Test   │     │    Online Model     │     │    Batch XGBoost   │   │
│  │   Feedback   │────►│  (River ARF)        │────►│    (Si mejora >5%) │   │
│  │   Endpoint   │     │  AdaptiveRandomForest│     │    Trigger Retrain │   │
│  └──────────────┘     └─────────────────────┘     └────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│         ┌─────────────────────────────────────────┐                        │
│         │  Evaluation cada 15 samples:            │                        │
│         │  - MAE / R² tracking                    │                        │
│         │  - Drift detection                      │                        │
│         │  - Improvement signal (>5% MAE mejor)   │                        │
│         └─────────────────────────────────────────┘                        │
│                                                                              │
│  Inference Priority:                                                         │
│  1. Niche-specific XGBoost (si entrenado)                                   │
│  2. Main trained XGBoost                                                    │
│  3. Online River model (si drift detectado o cold start)                    │
│  4. Base model / Heuristics                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### ¿Por qué Online Learning?

| Aspecto | Ventaja |
|---------|---------|
| **Latencia** | Updates en <1s por sample (vs minutos/horas para batch retrain) |
| **Adaptación** | Responde inmediatamente a cambios en engagement patterns |
| **Cold Start** | Funciona con 1 sample (no necesita dataset mínimo) |
| **Drift Detection** | Detecta cuando batch model está desactualizado |
| **Memoria** | ~10KB por modelo de nicho (vs MBs para XGBoost) |

#### Componente Principal

**Ubicación:** `backend/ml/online_update.py`

```python
from backend.ml.online_update import (
    # Funciones principales
    online_update,           # Update batch incremental
    online_update_single,    # Update single sample
    get_online_predictor,    # Obtener predictor por nicho
    reset_online_predictor,  # Reset modelo
    detect_drift,            # Detectar drift

    # Clases
    OnlineEngagementPredictor,  # Wrapper River
    OnlineUpdateResult,         # Resultado de update

    # Configuración
    EVALUATION_INTERVAL,     # 15 samples
    IMPROVEMENT_THRESHOLD,   # 5%
)
```

#### Uso Básico

```python
import pandas as pd
import numpy as np
from backend.ml.online_update import online_update

# Cuando llega feedback real de A/B test
features_df = pd.DataFrame([extracted_features])
targets_df = pd.DataFrame({
    "log_likes": [np.log1p(500)],      # 500 likes reales
    "log_comments": [np.log1p(45)],    # 45 comments reales
    "log_shares": [np.log1p(12)],
    "log_saves": [np.log1p(80)],
    "log_views": [np.log1p(5000)],
})

# Actualizar modelo online
result = online_update(
    niche="inmobiliaria",
    new_features=features_df,
    new_targets=targets_df,
    save_model=True
)

print(f"Samples procesados: {result.samples_processed}")
print(f"Total acumulado: {result.total_samples}")
print(f"MAE actual: {result.current_mae:.4f}")
print(f"¿Mejora detectada?: {result.improvement_detected}")
print(f"¿Trigger retrain XGBoost?: {result.trigger_full_retrain}")
```

#### API Endpoints

```bash
# Feedback con online learning automático
POST /api/v1/abtest/feedback
{
  "post_id": 123,
  "actual_likes": 500,
  "actual_comments": 45,
  "actual_saves": 80,
  "actual_shares": 12,
  "actual_views": 5000,
  "niche": "inmobiliaria"  # Opcional
}

# Respuesta:
{
  "post_id": 123,
  "online_learning_enabled": true,
  "samples_processed": 1,
  "total_samples": 47,
  "current_mae": 0.3421,
  "improvement_detected": false,
  "trigger_full_retrain": false,
  "update_time_ms": 12.5,
  "message": "Online learning updated for niche 'inmobiliaria'. Total samples: 47"
}

# Obtener status del modelo online
GET /api/v1/abtest/online-status?niche=inmobiliaria

# Respuesta:
{
  "niche": "inmobiliaria",
  "is_initialized": true,
  "samples_seen": 47,
  "current_mae": 0.3421,
  "best_mae": 0.3156,
  "feature_count": 136,
  "targets": ["log_likes", "log_comments", "log_shares", "log_saves", "log_views"],
  "online_learning_available": true
}
```

#### Inference con Fallback Online

```python
from app.services.ml_service import MLPredictor

predictor = MLPredictor()

# Predicción con fallback automático a online model
result = predictor.predict_with_online_fallback(
    content={"caption": "...", "business_type": "inmobiliaria"},
    use_drift_detection=True,
    recent_predictions=[72.5, 68.3, 75.1],  # Últimas predicciones batch
    recent_actuals=[45.2, 82.1, 55.3]       # Resultados reales
)

# Si drift detectado (MAE > 30%), usa online model
print(result["model_source"])  # "online_river" o "trained" o "base"
print(result["drift_detected"])  # True/False
```

#### Modelo River: AdaptiveRandomForestRegressor

El modelo online usa River's `AdaptiveRandomForestRegressor`:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `n_models` | 10 | Ensemble de 10 árboles |
| `max_depth` | 6 | Profundidad similar a XGBoost |
| `grace_period` | 50 | Samples antes del primer split |
| `split_confidence` | 0.01 | Confidence para splits |

**Características:**
- Maneja concept drift automáticamente
- Funciona con 1 sample (incremental puro)
- Lightweight: ~10KB por modelo de nicho
- Multi-output: modelo separado por métrica (likes, comments, etc.)

#### Persistencia de Modelos Online

```
models/
├── online_inmobiliaria.pkl    # Modelo online para nicho
├── online_floristeria.pkl
├── online_cafeteria.pkl
└── ...
```

Los modelos se guardan automáticamente después de cada update.

#### Evaluación y Señales de Mejora

Cada 15 samples, el sistema evalúa:

```python
# Metrics tracking
- MAE (Mean Absolute Error)
- R² (Coefficient of determination)

# Improvement detection
if (previous_mae - current_mae) / previous_mae > 0.05:  # >5% mejora
    trigger_full_retrain = True
    # Señal para reentrenar XGBoost batch con datos acumulados
```

#### Tests

```bash
# Ejecutar tests de online learning
pytest backend/tests/test_online_learning.py -v

# Tests incluidos:
# - test_50_incremental_updates (simula 50 batches, confirma mejora)
# - test_partial_fit_single
# - test_online_update_function
# - test_drift_detection
# - test_save_and_load
# - test_update_time_under_1_second
# - test_memory_efficiency
```

#### Integración con Feedback Loop Existente

El sistema se integra con el endpoint existente `/abtest/log-result`:

1. Usuario publica contenido → Predicción batch (XGBoost)
2. 24-48h después → Feedback real llega a `/abtest/feedback`
3. Online model se actualiza inmediatamente
4. Si mejora >5%, se señala retrain batch
5. Próximas predicciones: si drift, usa online; si no, usa batch

```
Timeline:
─────────────────────────────────────────────────────────────────
  Día 1             Día 2-3              Día 4+
  [Predicción]      [Feedback Real]      [Modelo Mejorado]
  XGBoost batch  →  Online update  →    XGBoost retrained
                                    o   Online fallback si drift
```

### 🔬 Evaluación Granular con Drift Detection (Nuevo)

Sistema de evaluación de modelos ML que proporciona métricas granulares por segmento y detección automática de drift para mantener la salud del modelo.

#### Arquitectura de Evaluación

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML EVALUATION PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │ Post-Train   │──► evaluate() ──────────────────────────────────────┐     │
│  │ Hook         │                                                      │     │
│  └──────────────┘                                                      │     │
│                                                                        ▼     │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │ Online Update│──►  │            GRANULAR EVALUATION                   │   │
│  │ (cada 50     │     │                                                  │   │
│  │  samples)    │     │  1. Global Metrics (MAE, R2, RMSE por target)   │   │
│  └──────────────┘     │  2. Format Metrics (Reel/Carousel/Image)        │   │
│                       │  3. Time Metrics (hora, día, prime time)         │   │
│                       │  4. RPI Metrics (weighted engagement)            │   │
│                       │  5. Drift Detection (KS test + MAE baseline)     │   │
│                       │  6. Calibration Analysis                         │   │
│                       │  7. Insight Generation                           │   │
│                       └──────────────────────────────────────────────────┘   │
│                                        │                                     │
│                                        ▼                                     │
│                       ┌─────────────────────────────────────────────────┐   │
│                       │  /logs/eval_{niche}.json                         │   │
│                       │  - Historial de evaluaciones                     │   │
│                       │  - Tendencias de MAE/drift                       │   │
│                       │  - Alertas automáticas                           │   │
│                       └─────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Métricas Disponibles

| Categoría | Métricas | Descripción |
|-----------|----------|-------------|
| **Global** | MAE, RMSE, R², MAPE | Por cada target (likes, comments, shares, saves, views) |
| **Formato** | MAE por formato | Reel, Carousel, Static Image, Story |
| **Tiempo** | MAE por segmento | Morning, Afternoon, Evening, Night, Weekday, Weekend, Prime Time |
| **RPI** | RPI MAE, Correlation | Métricas del índice de rendimiento ponderado |
| **Drift** | Score (0-1), KS stat | Detección de cambio en distribución de predicciones |
| **Calibration** | Error (%), Well-calibrated | Comparación expected vs observed percentiles |

#### Drift Detection

El sistema combina dos métodos para detectar drift:

1. **KS Test (Kolmogorov-Smirnov)**: Detecta cambios en la distribución de predicciones vs histórico
2. **MAE vs Baseline**: Compara el error actual con el promedio histórico

```python
# Score combinado (0-1)
drift_score = 0.4 * ks_score + 0.6 * mae_score

# Alerta si drift_score > 0.3
if drift_score > DRIFT_ALERT_THRESHOLD:
    # Trigger reentrenamiento
    # Log alert en dashboard
```

#### Componente Principal

**Ubicación:** `backend/ml/evaluate_model.py`

```python
from backend.ml.evaluate_model import (
    # Función principal
    evaluate,

    # Métricas individuales
    evaluate_global_metrics,
    evaluate_by_format,
    evaluate_by_time,
    evaluate_rpi,

    # Drift detection
    compute_drift_score,
    detect_drift_ks_test,
    detect_drift_mae,

    # Calibration
    analyze_calibration,

    # Insights
    generate_insights,

    # Storage
    save_evaluation_results,
    load_evaluation_history,
    get_baseline_mae,

    # Online trigger
    should_evaluate_online,
    reset_online_counter,
)
```

#### Uso Básico

```python
from backend.ml.evaluate_model import evaluate, get_baseline_mae

# Después de entrenar
result = evaluate(
    niche="restaurante",
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    metadata=metadata_df,  # con content_format, hour_of_day, day_of_week
    baseline_mae=get_baseline_mae("restaurante"),
    evaluation_type="retrain"
)

# Acceder a resultados
print(f"MAE Agregado: {result.global_metrics['_aggregate']['mae']}")
print(f"Drift Score: {result.drift.drift_score}")
print(f"Drift Detectado: {result.drift.drift_detected}")

# Ver insights
for insight in result.insights:
    print(f"  - {insight}")

# Métricas por formato
for fmt, metrics in result.format_metrics.items():
    print(f"{fmt}: MAE={metrics['mae']:.4f}, n={metrics['n_samples']}")
```

#### Integración Automática

**Post-Train (ml/train.py):**
```python
# Evaluación automática después de entrenar
model, metrics = train_niche_model(niche, df)

# metrics ahora incluye:
# {
#   "evaluation": {
#     "aggregate_mae": 0.2345,
#     "drift_score": 0.12,
#     "drift_detected": False,
#     "insights": ["Modelo estable, sin drift significativo."]
#   }
# }
```

**Online Update (cada 50 samples):**
```python
# Evaluación periódica durante online learning
from backend.ml.online_update import online_update

result = online_update(niche, features, targets)

# Si drift detectado, trigger_full_retrain = True
if result.trigger_full_retrain:
    retrain_xgboost(niche)
```

#### API Endpoints

```bash
# Obtener health summary (optimizado para dashboard)
GET /api/v1/ml/health/summary

# Respuesta:
{
  "total_niches": 5,
  "healthy_count": 4,
  "warning_count": 1,
  "overall_status": "warning",
  "avg_mae": 0.2456,
  "avg_drift_score": 0.15,
  "last_updated": "2025-01-18T10:30:00Z",
  "alerts": [
    {
      "niche": "inmobiliaria",
      "message": "Drift en inmobiliaria",
      "score": 0.42
    }
  ]
}

# Obtener métricas detalladas por niche
GET /api/v1/ml/health?niche=restaurante

# Respuesta:
{
  "status": "healthy",
  "niches": {
    "restaurante": {
      "niche": "restaurante",
      "last_evaluated": "2025-01-18T10:30:00Z",
      "evaluation_type": "retrain",
      "n_samples": 200,
      "metrics": {
        "aggregate_mae": 0.2134,
        "aggregate_r2": 0.8521,
        "aggregate_rmse": 0.2567
      },
      "per_target": {
        "log_likes": {"mae": 0.18, "r2": 0.89, "rmse": 0.22},
        "log_comments": {"mae": 0.25, "r2": 0.78, "rmse": 0.31}
      },
      "drift": {
        "detected": false,
        "score": 0.12,
        "alert": null
      },
      "format_metrics": {
        "reel": {"mae": 0.21, "n_samples": 80},
        "carousel": {"mae": 0.23, "n_samples": 50},
        "static_image": {"mae": 0.19, "n_samples": 70}
      },
      "time_metrics": {
        "hour_evening": {"mae": 0.25, "n_samples": 45},
        "prime_time": {"mae": 0.22, "n_samples": 30}
      },
      "insights": [
        "Modelo estable, sin drift significativo detectado."
      ],
      "history": [
        {"timestamp": "2025-01-17T10:00:00Z", "mae": 0.22, "drift_score": 0.10},
        {"timestamp": "2025-01-18T10:30:00Z", "mae": 0.21, "drift_score": 0.12}
      ]
    }
  }
}
```

#### Dashboard - Sección "Salud del Modelo"

El dashboard muestra automáticamente una sección de salud con:

- **Estado general**: Estable (verde) o Drift Detectado (amber)
- **Métricas clave**: MAE promedio, Drift Score, Nichos OK/Con Alertas
- **Alertas activas**: Lista de nichos con drift detectado y su score
- **Última actualización**: Timestamp de la última evaluación

```tsx
// Frontend: Se carga automáticamente en Dashboard.tsx
const [modelHealth] = await mlApi.getModelHealthSummary()

// Componente muestra:
// - Shield verde si overall_status === 'healthy'
// - AlertTriangle amber si overall_status === 'warning'
// - Grid con MAE, Drift Score, Nichos OK, Con Alertas
// - Lista de alertas si las hay
```

#### Calibration Plot (Text-Based)

El sistema genera un plot de calibración en logs:

```
==================================================
CALIBRATION PLOT - RESTAURANTE
==================================================
Expected %     Observed %     Delta      Visual
--------------------------------------------------
5.0            7.2            +2.2       [++        ]
15.0           13.8           -1.2       [-         ]
25.0           26.5           +1.5       [+         ]
35.0           33.1           -1.9       [-         ]
45.0           47.2           +2.2       [++        ]
55.0           53.8           -1.2       [-         ]
65.0           67.5           +2.5       [++        ]
75.0           72.1           -2.9       [--        ]
85.0           86.3           +1.3       [+         ]
95.0           93.5           -1.5       [-         ]
--------------------------------------------------
Calibration Error: 1.85%
Well Calibrated: Yes
==================================================
```

#### Insight Generation

El sistema genera insights automáticos en español:

```python
insights = [
    "ALERTA: Drift detectado (score=0.45). Considerar reentrenamiento.",
    "Comments: MAE alto (0.72). Revisar features o datos de comments.",
    "Reel: MAE elevado (0.65, n=50). Posible subrepresentación o patrón diferente.",
    "Prime time (18-21h): MAE alto (0.58). Mayor variabilidad en horas pico.",
    "Weekend: MAE 35% mayor que weekday. Comportamiento diferente en fines de semana."
]
```

#### Tests

```bash
# Ejecutar tests del módulo de evaluación
pytest tests/test_evaluate_model.py -v

# Tests incluidos:
# - test_compute_mae, test_compute_rmse, test_compute_r2
# - test_evaluate_global_metrics, test_evaluate_by_format, test_evaluate_by_time
# - test_detect_drift_mae, test_detect_drift_ks_test, test_compute_drift_score
# - test_analyze_calibration
# - test_generate_insights
# - test_should_evaluate_online, test_reset_online_counter
# - test_evaluate_basic, test_evaluate_with_drift_baseline
# - test_save_and_load_evaluation, test_get_baseline_mae
```

#### Archivos de Log

```
logs/
├── eval_restaurante.json
├── eval_inmobiliaria.json
├── eval_cafeteria.json
└── ...

# Estructura:
{
  "niche": "restaurante",
  "last_updated": "2025-01-18T10:30:00Z",
  "latest": { ... },  # Última evaluación completa
  "history": [ ... ]   # Últimas 100 evaluaciones
}
```

### Logging de Cold Start

El sistema genera logs claros cuando usa el modelo base:

```
INFO: Usando modelo base + fine-tune por cold start (niche: floristeria)
INFO: Predicción con modelo base (cold start) - Niche: floristeria, Score: 72.3
```

La respuesta de predicción incluye metadata sobre la fuente del modelo:
```json
{
  "score": 72.3,
  "confidence": 68.0,
  "model_source": "base",
  "cold_start": true,
  "niche": "floristeria",
  "explanation": {
    "explanation_text": "Usando modelo base preentrenado para cold start..."
  }
}
```

## 🔒 Privacidad y GDPR

- **Consentimiento explícito**: Checkbox obligatorio en onboarding.
- **Anonimización**: Usernames hasheados en logs de predicción.
- **Retención de datos**: Configurable según normativa local.
- **Derecho al olvido**: Endpoint para eliminación de datos de usuario.

## Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

---

**Hecho con ❤️ para negocios locales** | Powered by **Grok (xAI)** & **BrandPulse ML Engine**

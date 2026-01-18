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

#### 🧬 Semantic Embeddings (Nuevo - Priorizado)
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB, multilingual)
- **Output**: 384-dim embeddings → PCA reducido a 30-dim (`embedding_1` a `embedding_30`)
- **Ventajas**: Captura significado semántico que las heurísticas no pueden detectar
- **Soporte**: Español + Inglés nativamente
- **Descarga**: Automática en primera ejecución desde HuggingFace

#### 📝 Features Manuales (Backup)
- **Análisis de Caption**: longitud, emoji_count, hashtag_count, has_question (regex), has_strong_cta (Comenta, Guarda, DM, Visita, Taggea), lexical_richness.
- **Análisis de Sentimiento**: VADER (nltk) para score emocional (-1 a +1). Contenido emocional/positivo impulsa engagement.
- **Features Temporales**: post_hour, post_day_of_week, is_weekend, is_prime_time.
- **Formato**: One-hot encoding (Reel, Carousel, Static, TikTok).
- **Niche Flags**: Detección de keywords por vertical (inmobiliaria: "casa", "tour", "Triana"; floristería: "flores", "arreglo", "ramo").

> **Arquitectura de Features**: El sistema prioriza embeddings semánticos (30 features) + multimodales (48 features) y mantiene heurísticas manuales (~58 features), resultando en **~136 features totales** para XGBoost.

### 🎬 Multimodal Late Fusion (Nuevo)

Sistema robusto de fusión multimodal para maximizar predicción de engagement en Reels/TikTok. Combina información de múltiples fuentes (texto, audio transcrito, texto visual) para predicciones más precisas.

#### Arquitectura Late Fusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTIMODAL LATE FUSION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │ Caption Text │ ──► MiniLM-L6-v2 ──► 384-dim ──► PCA ──► 30-dim ─────┐   │
│  └──────────────┘                                                       │   │
│                                                                         │   │
│  ┌──────────────┐                                                       │   │
│  │   Whisper    │ ──► MiniLM-L6-v2 ──► 384-dim ──► PCA ──► 20-dim ─────┼──►│CONCAT│──► XGBoost
│  │  Transcript  │                                                       │   │       │     │
│  └──────────────┘                                                       │   │       │     │
│                                                                         │   │       │     ▼
│  ┌──────────────┐                                                       │   │       │   Score
│  │   EasyOCR    │ ──► MiniLM-L6-v2 ──► 384-dim ──► PCA ──► 20-dim ─────┤   │       │   0-100
│  │  Visual Text │                                                       │   │       │     +
│  └──────────────┘                                                       │   │       │   SHAP
│                                                                         │   │
│  ┌──────────────┐                                                       │   │
│  │ Heuristics   │ ──► ~58 features (hooks, CTAs, timing, etc.) ────────┤   │
│  └──────────────┘                                                       │   │
│                                                                         │   │
│  ┌──────────────┐                                                       │   │
│  │ Interactions │ ──► 8 cross-modal synergy features ──────────────────┘   │
│  └──────────────┘                                                           │
│                                                                              │
│  Total: 30 + 20 + 20 + 58 + 8 = ~136 features                               │
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

#### Features Generados (48 total)

**1. Transcript Embeddings (20 dims) - Audio transcrito via Whisper:**
```
transcript_emb_1, transcript_emb_2, ..., transcript_emb_20
```

**2. OCR Embeddings (20 dims) - Texto visual via EasyOCR:**
```
ocr_emb_1, ocr_emb_2, ..., ocr_emb_20
```

**3. Cross-Modal Interaction Features (8 dims):**

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
├── embedding_pca_30.pkl      # PCA para caption (30 dims)
├── transcript_pca_20.pkl     # PCA para transcript (20 dims)
├── ocr_pca_20.pkl            # PCA para OCR (20 dims)
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
- **XGBoost** - Modelo de predicción
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
- `GET /api/v1/abtest/stats` - Estadísticas de predicciones vs realidad

### Contenido & Análisis
- `POST /api/v1/content/{business_id}/generate-calendar` - Generar calendario con IA
- `GET /api/v1/competitors/{business_id}/{competitor_id}/analysis` - Ver patrones extraídos

### Elena Bridge (Extensión)
- `POST /api/ingest/raw` - Ingestar contenido desde la extensión (posts, reels, videos, perfiles)
- `GET /api/ingest/status` - Verificar estado del endpoint de ingesta

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
    get_caption_embedding,      # 384-dim raw embedding
    get_embedding_features,     # 30-dim dict para ML
    EmbeddingExtractor          # Clase completa
)

# Uso simple
embedding_384 = get_caption_embedding("Nuevo apartamento en Triana!")
# numpy array (384,)

features_30 = get_embedding_features("Nuevo apartamento en Triana!")
# {"embedding_1": 0.123, "embedding_2": -0.456, ..., "embedding_30": 0.789}
```

**Configuración del modelo:**
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Modelo | `all-MiniLM-L6-v2` | Lightweight, ~80MB |
| Dimensión raw | 384 | Output del transformer |
| Dimensión PCA | 30 | Reducido para XGBoost |
| Max tokens | 256 | Truncamiento automático |

**Ajuste de PCA en corpus de training:**
```python
from ml.features_embeddings import EmbeddingExtractor

extractor = EmbeddingExtractor()

# Opción 1: Desde textos
extractor.fit_pca_from_texts(["caption1", "caption2", ...], save=True)

# Opción 2: Desde embeddings raw
embeddings = extractor.get_caption_embeddings_batch(texts)
extractor.fit_pca(embeddings, save=True)

# El modelo PCA se guarda en: models/embedding_pca_30.pkl
```

**Integración automática:**
- `FeatureExtractor.extract_features()` añade automáticamente `embedding_1` a `embedding_30`
- `train.py` genera embeddings para datos de entrenamiento
- `pretrain_base_model.py` genera embeddings sintéticos para pretraining

#### 1. Preentrenar Modelo Base

```bash
# Generar datos sintéticos y entrenar modelo base (una vez)
python ml/pretrain_base_model.py

# Con más samples para mejor generalización
python ml/pretrain_base_model.py --samples 20000

# Output: models/base_xgboost.pkl
```

**Features del dataset sintético (73 total):**

| Categoría | Features | Count |
|-----------|----------|-------|
| **Embeddings** | `embedding_1` a `embedding_30` (PCA sintético) | 30 |
| **Caption** | `caption_length`, `caption_words`, `emoji_count`, `hashtag_count`, `lexical_richness` | 10 |
| **Hooks** | `hook_question`, `hook_pov`, `hook_number`, `hook_bold_claim`, `hook_story` | 7 |
| **Triggers** | `trigger_urgency`, `trigger_curiosity`, `trigger_action`, `trigger_emotion` | 8 |
| **CTAs** | `cta_comment`, `cta_save`, `cta_share`, `cta_follow`, `cta_dm`, `cta_link` | 7 |
| **Formato** | `is_reel`, `is_carousel`, `is_static`, `video_duration`, `has_audio` | 6 |
| **Timing** | `hour_of_day`, `day_of_week`, `is_prime_time`, `is_weekend` | 4 |
| **Nicho** | `niche_inmobiliaria`, `niche_cafeteria`, `niche_restaurante`, etc. | 8 |
| **Sentiment** | `sentiment_compound`, `sentiment_positive`, `sentiment_negative` | 4 |

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

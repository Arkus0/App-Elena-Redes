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
- **Análisis de Caption**: longitud, emoji_count, hashtag_count, has_question (regex), has_strong_cta (Comenta, Guarda, DM, Visita, Taggea), lexical_richness.
- **Análisis de Sentimiento**: VADER (nltk) para score emocional (-1 a +1). Contenido emocional/positivo impulsa engagement.
- **Features Temporales**: post_hour, post_day_of_week, is_weekend, is_prime_time.
- **Formato**: One-hot encoding (Reel, Carousel, Static, TikTok).
- **Niche Flags**: Detección de keywords por vertical (inmobiliaria: "casa", "tour", "Triana"; floristería: "flores", "arreglo", "ramo").

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
- **NLTK VADER** - Análisis de sentimiento
- **Apify Client** - Scraping de redes sociales
- **NumPy/Pandas** - Procesamiento de datos

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

#### 1. Preentrenar Modelo Base

```bash
# Generar datos sintéticos y entrenar modelo base (una vez)
python ml/pretrain_base_model.py

# Con más samples para mejor generalización
python ml/pretrain_base_model.py --samples 20000

# Output: models/base_xgboost.pkl
```

**Features del dataset sintético:**
- `caption_length`, `emoji_count`, `hashtag_count`, `cta_count`
- `hook_question`, `hook_pov`, `hook_number` (tipos de gancho)
- `trigger_urgency`, `trigger_curiosity`, `trigger_action`
- `is_reel`, `is_carousel`, `is_static` (formato)
- `hour_of_day`, `is_prime_time`, `is_weekend` (timing)
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

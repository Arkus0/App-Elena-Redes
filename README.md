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

### Autenticación & Negocio
- `POST /api/v1/auth/login`
- `POST /api/v1/business/onboard`

## 🧪 Modelos ML

El sistema utiliza modelos entrenados específicamente para redes sociales:
- **Engagement Model (XGBoost)**: Predice likes/comments/shares con métricas R², RMSE, MAE.
- **Growth Model (XGBoost)**: Predice RPI (Relative Performance Index).
- **Format Classifier (Random Forest)**: Recomienda el mejor formato con ROC-AUC para clasificación binaria.
- **Sentiment Analyzer (VADER)**: Detecta tono emocional para optimizar engagement.

### Reentrenamiento
- Time-based train/test split para evitar data leakage.
- Evaluación vs baseline (media histórica) para medir mejora real.
- Feedback loop con Human-in-the-Loop para aprendizaje continuo.

## 🔒 Privacidad y GDPR

- **Consentimiento explícito**: Checkbox obligatorio en onboarding.
- **Anonimización**: Usernames hasheados en logs de predicción.
- **Retención de datos**: Configurable según normativa local.
- **Derecho al olvido**: Endpoint para eliminación de datos de usuario.

## Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

---

**Hecho con ❤️ para negocios locales** | Powered by **Grok (xAI)** & **BrandPulse ML Engine**

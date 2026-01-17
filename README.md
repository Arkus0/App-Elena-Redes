# BrandPulse AI

> **El co-piloto definitivo para generar contenido de ALTO ENGAGEMENT para negocios locales**

BrandPulse AI analiza los posts más exitosos de tus competidores y genera contenido optimizado para Instagram, TikTok y LinkedIn. Basado en patrones reales que funcionan, no en ideas genéricas.

## Características Principales

### Analisis de Competidores
- Scraping automático de Instagram, TikTok y LinkedIn via **Apify API**
- Extracción de métricas de engagement (likes, comentarios, guardados, compartidos)
- Identificación de patrones ganadores con IA
- **Balanced Sampling**: Recolección de posts exitosos Y fallidos para eliminar Survivor Bias

### Patrones de Engagement
- **Hooks**: Qué primeras líneas/frames funcionan mejor
- **CTAs**: Qué llamadas a la acción generan más interacción
- **Formatos**: Reels vs Carousels vs Estático
- **Timing**: Mejores días y horas para publicar
- **Hashtags**: Estrategia óptima de hashtags

### Generador de Calendarios
- Calendarios mensuales completos (15-40 piezas)
- Scripts de video con estructura temporal
- Guías de filmación detalladas
- Prompts para generación de imágenes con IA

### Motor de Predicción XGBoost (GrowthPredictionEngine)
- **Modelo XGBoost** entrenado con Hook Theory
- **SHAP Explainability**: Explicación detallada de cada predicción
- **17 features** incluyendo:
  - Hook Energy (primeros 3 segundos) - CRÍTICO
  - Retention Energy (post-hook)
  - Cut Rate (cortes por minuto)
  - Face in Hook (presencia de cara)
  - Tempo (BPM del audio)
  - 10 componentes PCA semánticos
- Validación cruzada K-Fold
- Métricas: RMSE, MAE, R²

### Account Health Scoring (NUEVO)
Sistema de evaluación de salud de cuenta antes de generar predicciones:

| Estado | Umbral | Factor de Calibración |
|--------|--------|----------------------|
| **HEALTHY** | Views >= 10% de seguidores | 1.0x (sin penalización) |
| **LOW_AUTHORITY** | Views < 10% de seguidores | 0.3x |
| **POSSIBLE_SHADOWBAN** | Views < 2% de seguidores | 0.1x |

**Lógica:**
1. Analiza los últimos 10 posts de la cuenta del usuario
2. Calcula media de views y desviación estándar
3. Compara con número de seguidores
4. Aplica factor de penalización a predicciones
5. Genera mensaje de advertencia si hay problemas

**Mensaje para usuarios Low Authority:**
> "Tu cuenta tiene baja tracción actualmente. Este video está optimizado, pero necesitarás subir 5-10 así de constantes para reactivar el algoritmo."

### Trend Velocity (NUEVO)
Sistema de verificación de frescura de tendencias en tiempo real:

| Estado | Umbral | Recomendación |
|--------|--------|---------------|
| **TRENDING** | >= 50% en últimas 48h | USAR - Muy fresco |
| **RISING** | >= 30% en última semana | USAR - En crecimiento |
| **STABLE** | Distribución normal | PRECAUCIÓN |
| **STALE** | >= 80% hace > 2 semanas | NO USAR - Caducado |

**Safety Check para Recetas:**
- Antes de recomendar un Audio o Hashtag, verifica los últimos 50 videos que lo usaron
- Calcula la "velocidad" de la tendencia basándose en fechas de publicación
- Filtra automáticamente elementos STALE aunque tengan muchos likes históricos
- Ordena recomendaciones por velocity_score

### Scanner Viral
- Búsqueda de tendencias por keyword
- Ideas reactivas listas para filmar
- Generador de ideas rápidas (POV, transformaciones, tips, storytime)

### Variaciones A/B
- Genera múltiples versiones de cada contenido
- Diferentes hooks, CTAs y audios
- Compara scores de engagement predichos

### Exportación
- CSV para hojas de cálculo
- JSON para integraciones
- Incluye scripts y guías de filmación

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
- **Python 3.11+**
- **FastAPI** - Framework web async
- **SQLAlchemy 2.0** - ORM con soporte async
- **SQLite/PostgreSQL** - Base de datos
- **XGBoost** - Modelo de predicción
- **SHAP** - Explicabilidad de predicciones
- **Anthropic Claude** - IA para análisis y generación
- **Apify Client** - Scraping de redes sociales
- **NumPy/Pandas** - Procesamiento de datos

### Frontend
- **React 18** con TypeScript
- **Vite** - Build tool
- **Tailwind CSS** - Estilos
- **Zustand** - State management
- **React Query** - Data fetching
- **React Router** - Navegación

## Instalación

### Prerrequisitos

- Python 3.11+
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

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

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
# REQUERIDO: Anthropic Claude para IA
ANTHROPIC_API_KEY=sk-ant-tu-clave-aqui

# REQUERIDO: Apify para scraping
APIFY_API_KEY=apify_api_tu-clave-aqui

# OPCIONAL: Generar clave segura
SECRET_KEY=tu-clave-secreta-de-32-caracteres
```

**Obtener claves:**
- **Anthropic**: https://console.anthropic.com/
- **Apify**: https://apify.com/ (plan gratuito disponible)

## Ejecutar la Aplicación

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

### 1. Registro e Inicio de Sesión

1. Abre http://localhost:5173
2. Haz click en "Registrarse" o usa el **modo demo** para probar sin cuenta

### 2. Onboarding (Configuración Inicial)

1. **Tu Negocio**: Nombre, tipo (floristería, inmobiliaria, etc.), ubicación
2. **Redes Sociales**: Tus handles de Instagram y TikTok
3. **Competidores**: Añade 3-7 competidores para analizar (locales o de referencia)
4. **Objetivos**: Qué quieres lograr (engagement, leads, visitas, ventas)

### 3. Análisis de Competidores

- Ve a "Competidores" en el menú
- Espera a que se complete el análisis (indicador de progreso)
- Haz click en "Ver análisis" para ver:
  - Posts con mejor engagement
  - Patrones ganadores identificados
  - Hooks y CTAs recomendados
  - Hashtags efectivos

### 4. Generar Calendario de Contenido

1. Ve a "Generar Contenido"
2. Selecciona mes y año
3. Elige cantidad de posts (10-30)
4. Define objetivo principal
5. Ajusta el mix de contenido (% Reels, Carousels, Estático)
6. Click en "Generar Calendario"

### 5. Revisar y Usar el Contenido

Cada pieza de contenido incluye:
- **Hook**: Primera línea/frame que engancha
- **Caption**: Texto completo con emojis y formato
- **Hashtags**: Optimizados para tu nicho (verificados por TrendVelocity)
- **Script**: Estructura temporal del video
- **Guía de Filmación**: Ángulos, equipo, tips de edición
- **Audio**: Recomendación de audio trending (verificado como FRESH)
- **Score de Engagement**: Predicción calibrada según salud de tu cuenta

### 6. Scanner Viral

- Busca tendencias por keyword
- Genera ideas rápidas (POV, transformaciones, tips, storytime)
- Obtén contenido reactivo listo para filmar en minutos

## Estructura del Proyecto

```
brandpulse-ai/
├── backend/
│   ├── app/
│   │   ├── api/                    # Endpoints de la API
│   │   │   ├── auth.py            # Autenticación
│   │   │   ├── business.py        # Gestión de negocios
│   │   │   ├── competitors.py     # Competidores
│   │   │   ├── content.py         # Generación de contenido
│   │   │   ├── viral.py           # Scanner viral
│   │   │   ├── ml.py              # Predicciones ML
│   │   │   ├── growth.py          # Growth Prediction + Account Health
│   │   │   └── trends.py          # Trend Velocity
│   │   ├── core/                   # Configuración y seguridad
│   │   ├── models/                 # Modelos de base de datos
│   │   ├── schemas/                # Schemas Pydantic
│   │   │   ├── growth.py          # Schemas de predicción
│   │   │   ├── account_health.py  # Schemas de salud de cuenta
│   │   │   └── trend_velocity.py  # Schemas de velocidad de tendencias
│   │   ├── services/               # Lógica de negocio
│   │   │   ├── growth_prediction_engine.py  # Motor XGBoost
│   │   │   ├── account_health_scoring.py    # Evaluación de salud
│   │   │   ├── trend_velocity.py            # Verificación de frescura
│   │   │   ├── analytics_engine.py          # Extracción de features
│   │   │   ├── text_intelligence.py         # NLP pipeline
│   │   │   └── apify_service.py             # Scraping
│   │   └── main.py                 # Aplicación FastAPI
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/             # Componentes React
│   │   ├── pages/                  # Páginas
│   │   ├── stores/                 # Zustand stores
│   │   ├── services/               # API client
│   │   ├── types/                  # TypeScript types
│   │   └── App.tsx                 # Componente principal
│   ├── package.json
│   └── .env.example
└── README.md
```

## API Endpoints

### Autenticación
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Registrar usuario |
| POST | `/api/v1/auth/login` | Iniciar sesión |

### Negocios
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/business/onboard` | Configurar negocio |
| GET | `/api/v1/business/me` | Obtener mis negocios |
| GET | `/api/v1/business/{id}/status` | Estado del análisis |

### Competidores
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/v1/competitors/{business_id}` | Listar competidores |
| POST | `/api/v1/competitors/{business_id}/add` | Añadir competidor |
| GET | `/api/v1/competitors/{business_id}/{competitor_id}/analysis` | Análisis detallado |

### Contenido
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/content/{business_id}/generate-calendar` | Generar calendario |
| GET | `/api/v1/content/{business_id}/calendars` | Listar calendarios |
| POST | `/api/v1/content/{business_id}/content/{id}/variations` | Generar variaciones A/B |
| POST | `/api/v1/content/{business_id}/export` | Exportar contenido |

### Scanner Viral
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/viral/{business_id}/scan` | Buscar tendencias |
| GET | `/api/v1/viral/{business_id}/trending` | Tendencias del nicho |
| POST | `/api/v1/viral/{business_id}/quick-idea` | Idea rápida |

### Growth Prediction (XGBoost)
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/growth/predict` | Predicción con SHAP |
| POST | `/api/v1/growth/predict/batch` | Predicción en batch |
| POST | `/api/v1/growth/predict/calibrated` | Predicción calibrada con salud de cuenta |
| POST | `/api/v1/growth/train` | Entrenar modelo |
| GET | `/api/v1/growth/status` | Estado del modelo |
| GET | `/api/v1/growth/features/importance` | Importancia de features |

### Account Health
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/growth/account-health` | Evaluar salud de cuenta |
| GET | `/api/v1/growth/account-health/summary/{status}` | Descripción de estado |

### Trend Velocity
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/trends/check` | Verificar frescura de tendencia |
| POST | `/api/v1/trends/check/batch` | Verificar múltiples tendencias |
| POST | `/api/v1/trends/validate-recipe` | Validar elementos de receta |
| GET | `/api/v1/trends/quick/{type}/{identifier}` | Verificación rápida |
| GET | `/api/v1/trends/status/{status}` | Descripción de estado |

## Flujo de Predicción Completo

```
1. Usuario solicita predicción para un video
   │
   ▼
2. AccountHealthScoring analiza últimos 10 posts del usuario
   │
   ├─ HEALTHY (>=10% views/followers) → Factor 1.0x
   ├─ LOW_AUTHORITY (<10%) → Factor 0.3x + Warning
   └─ POSSIBLE_SHADOWBAN (<2%) → Factor 0.1x + Alerta crítica
   │
   ▼
3. GrowthPredictionEngine extrae features del contenido
   │
   ├─ Hook Energy (0-3 segundos) ← CRÍTICO
   ├─ Retention Energy (3s+)
   ├─ Cut Rate, Face in Hook, Tempo
   └─ 10 PCA semánticos
   │
   ▼
4. XGBoost genera predicción base
   │
   ▼
5. SHAP explica qué features impactan más
   │
   ▼
6. TrendVelocity verifica audios y hashtags recomendados
   │
   ├─ TRENDING (>50% en 48h) → Incluir
   ├─ RISING (>30% en semana) → Incluir
   ├─ STABLE → Incluir con precaución
   └─ STALE (>80% hace >2 semanas) → FILTRAR
   │
   ▼
7. Predicción calibrada + Elementos frescos → Usuario
```

## Demo Data

Si no tienes claves API configuradas, la aplicación usa datos de demostración realistas para:
- Perfiles de competidores con métricas
- Posts con engagement scores
- Patrones de contenido extraídos
- Contenido generado de ejemplo
- Simulación de TrendVelocity con patrones predefinidos

Perfecto para probar la interfaz y entender el flujo.

## Seguridad

- Contraseñas hasheadas con bcrypt
- Autenticación JWT
- Tokens con expiración configurable
- CORS configurado para desarrollo local

## Variables de Entorno

### Backend (.env)

| Variable | Descripción | Requerido |
|----------|-------------|-----------|
| `SECRET_KEY` | Clave para JWT | Sí |
| `ANTHROPIC_API_KEY` | API key de Claude | Sí* |
| `APIFY_API_KEY` | API key de Apify | Sí* |
| `DATABASE_URL` | URL de base de datos | No (default: SQLite) |

*Sin estas claves, se usan datos de demostración.

## Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit tus cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## Agradecimientos

- [Anthropic](https://anthropic.com) por Claude AI
- [Apify](https://apify.com) por la infraestructura de scraping
- [XGBoost](https://xgboost.readthedocs.io/) por el framework de ML
- [SHAP](https://shap.readthedocs.io/) por la explicabilidad
- Todos los negocios locales que nos inspiran a crear mejores herramientas

---

**Hecho con amor para negocios locales que quieren crecer en redes sociales**

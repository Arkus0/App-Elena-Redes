# BrandPulse AI 🚀

> **El co-piloto definitivo para generar contenido de ALTO ENGAGEMENT para negocios locales**

BrandPulse AI analiza los posts más exitosos de tus competidores y genera contenido optimizado para Instagram, TikTok y LinkedIn. Basado en patrones reales que funcionan, no en ideas genéricas.

![BrandPulse AI](https://via.placeholder.com/800x400/1f2937/0ea5e9?text=BrandPulse+AI)

## ✨ Características Principales

### 📊 Análisis de Competidores
- Scraping automático de Instagram, TikTok y LinkedIn via **Apify API**
- Extracción de métricas de engagement (likes, comentarios, guardados, compartidos)
- Identificación de patrones ganadores con IA

### 🎯 Patrones de Engagement
- **Hooks**: Qué primeras líneas/frames funcionan mejor
- **CTAs**: Qué llamadas a la acción generan más interacción
- **Formatos**: Reels vs Carousels vs Estático
- **Timing**: Mejores días y horas para publicar
- **Hashtags**: Estrategia óptima de hashtags

### 📅 Generador de Calendarios
- Calendarios mensuales completos (15-40 piezas)
- Scripts de video con estructura temporal
- Guías de filmación detalladas
- Prompts para generación de imágenes con IA

### 📈 Predicción de Engagement
- Score 0-100 para cada contenido
- Explicación detallada de por qué funcionará
- Comparación con posts virales de competidores
- Métricas predichas (likes, comentarios, guardados)

### 🔥 Scanner Viral
- Búsqueda de tendencias por keyword
- Ideas reactivas listas para filmar
- Generador de ideas rápidas (POV, transformaciones, tips, storytime)

### 🔄 Variaciones A/B
- Genera múltiples versiones de cada contenido
- Diferentes hooks, CTAs y audios
- Compara scores de engagement predichos

### 📤 Exportación
- CSV para hojas de cálculo
- JSON para integraciones
- Incluye scripts y guías de filmación

## 🏢 Tipos de Negocio Soportados

- 🏠 Inmobiliarias (Real Estate)
- 💐 Floristerías
- ☕ Cafeterías
- 💇 Peluquerías
- 🛍️ Tiendas Locales
- 🍽️ Restaurantes
- 💪 Gimnasios
- 🏥 Clínicas
- ✨ Y más...

## 🛠️ Tech Stack

### Backend
- **Python 3.11+**
- **FastAPI** - Framework web async
- **SQLAlchemy 2.0** - ORM con soporte async
- **SQLite/PostgreSQL** - Base de datos
- **Anthropic Claude** - IA para análisis y generación
- **Apify Client** - Scraping de redes sociales

### Frontend
- **React 18** con TypeScript
- **Vite** - Build tool
- **Tailwind CSS** - Estilos
- **Zustand** - State management
- **React Query** - Data fetching
- **React Router** - Navegación

## 🚀 Instalación

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

## 📖 Guía de Uso

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
- **Hashtags**: Optimizados para tu nicho
- **Script**: Estructura temporal del video
- **Guía de Filmación**: Ángulos, equipo, tips de edición
- **Audio**: Recomendación de audio trending
- **Score de Engagement**: Predicción 0-100 con explicación

### 6. Scanner Viral

- Busca tendencias por keyword
- Genera ideas rápidas (POV, transformaciones, tips, storytime)
- Obtén contenido reactivo listo para filmar en minutos

## 📁 Estructura del Proyecto

```
brandpulse-ai/
├── backend/
│   ├── app/
│   │   ├── api/          # Endpoints de la API
│   │   ├── core/         # Configuración y seguridad
│   │   ├── models/       # Modelos de base de datos
│   │   ├── schemas/      # Schemas Pydantic
│   │   ├── services/     # Lógica de negocio
│   │   └── main.py       # Aplicación FastAPI
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/   # Componentes React
│   │   ├── pages/        # Páginas
│   │   ├── stores/       # Zustand stores
│   │   ├── services/     # API client
│   │   ├── types/        # TypeScript types
│   │   └── App.tsx       # Componente principal
│   ├── package.json
│   └── .env.example
└── README.md
```

## 🔌 API Endpoints

### Autenticación
- `POST /api/v1/auth/register` - Registrar usuario
- `POST /api/v1/auth/login` - Iniciar sesión

### Negocios
- `POST /api/v1/business/onboard` - Configurar negocio
- `GET /api/v1/business/me` - Obtener mis negocios
- `GET /api/v1/business/{id}/status` - Estado del análisis

### Competidores
- `GET /api/v1/competitors/{business_id}` - Listar competidores
- `POST /api/v1/competitors/{business_id}/add` - Añadir competidor
- `GET /api/v1/competitors/{business_id}/{competitor_id}/analysis` - Análisis detallado

### Contenido
- `POST /api/v1/content/{business_id}/generate-calendar` - Generar calendario
- `GET /api/v1/content/{business_id}/calendars` - Listar calendarios
- `POST /api/v1/content/{business_id}/content/{id}/variations` - Generar variaciones A/B
- `POST /api/v1/content/{business_id}/export` - Exportar contenido

### Scanner Viral
- `POST /api/v1/viral/{business_id}/scan` - Buscar tendencias
- `GET /api/v1/viral/{business_id}/trending` - Tendencias del nicho
- `POST /api/v1/viral/{business_id}/quick-idea` - Idea rápida

## 🧪 Demo Data

Si no tienes claves API configuradas, la aplicación usa datos de demostración realistas para:
- Perfiles de competidores con métricas
- Posts con engagement scores
- Patrones de contenido extraídos
- Contenido generado de ejemplo

Perfecto para probar la interfaz y entender el flujo.

## 🔒 Seguridad

- Contraseñas hasheadas con bcrypt
- Autenticación JWT
- Tokens con expiración configurable
- CORS configurado para desarrollo local

## 📝 Variables de Entorno

### Backend (.env)

| Variable | Descripción | Requerido |
|----------|-------------|-----------|
| `SECRET_KEY` | Clave para JWT | Sí |
| `ANTHROPIC_API_KEY` | API key de Claude | Sí* |
| `APIFY_API_KEY` | API key de Apify | Sí* |
| `DATABASE_URL` | URL de base de datos | No (default: SQLite) |

*Sin estas claves, se usan datos de demostración.

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit tus cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- [Anthropic](https://anthropic.com) por Claude AI
- [Apify](https://apify.com) por la infraestructura de scraping
- Todos los negocios locales que nos inspiran a crear mejores herramientas

---

**Hecho con ❤️ para negocios locales que quieren crecer en redes sociales**

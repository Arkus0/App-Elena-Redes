# PyAV (av) Installation Guide for GitHub Codespaces

## El Problema

PyAV es un wrapper de Python para FFmpeg. Cuando se instala con `pip install av`, puede fallar con errores como:

```
error: unknown type name 'AVCodecContext'
error: 'AV_CODEC_FLAG_GLOBAL_HEADER' undeclared
error: 'AVFMT_RAWPICTURE' undeclared
fatal error: libavcodec/avcodec.h: No such file or directory
```

### ¿Por qué ocurre?

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyAV Compilation Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   pip install av                                                │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────┐    ¿Wheel disponible?                    │
│   │  Check PyPI     │────────────────────────┐                  │
│   └─────────────────┘                        │                  │
│         │ NO                                 │ SÍ               │
│         ▼                                    ▼                  │
│   ┌─────────────────┐                 ┌──────────────┐         │
│   │ Build from      │                 │ Download     │         │
│   │ source (Cython) │                 │ binary wheel │         │
│   └─────────────────┘                 └──────────────┘         │
│         │                                    │                  │
│         ▼                                    │                  │
│   ┌─────────────────┐                        │                  │
│   │ Needs FFmpeg    │                        │                  │
│   │ dev headers     │◄───── AQUÍ FALLA ─────┘                  │
│   │ (libavcodec-dev)│                                          │
│   └─────────────────┘                                          │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────┐                                          │
│   │ pkg-config      │                                          │
│   │ locate libs     │                                          │
│   └─────────────────┘                                          │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────┐                                          │
│   │ Compile C       │                                          │
│   │ extensions      │                                          │
│   └─────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Causas principales:**

1. **Faltan headers de desarrollo de FFmpeg** (`libavcodec-dev`, etc.)
2. **Versión incompatible de FFmpeg** (PyAV 10.x requiere FFmpeg 4.x o 5.x)
3. **Faltan herramientas de compilación** (`build-essential`, `pkg-config`)
4. **Python headers faltantes** (`python3-dev`)

---

## Soluciones

### Solución 1: Instalar Dependencias del Sistema (Recomendada)

```bash
# 1. Actualizar paquetes
sudo apt-get update

# 2. Instalar dependencias de FFmpeg
sudo apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev \
    pkg-config \
    build-essential \
    python3-dev

# 3. Instalar PyAV
pip install av
```

### Solución 2: Usar Wheel Pre-compilado (Más Rápida)

PyAV publica wheels para algunas plataformas. Intentar instalar solo wheels:

```bash
# Intentar instalar solo wheel pre-compilado
pip install av --only-binary=:all:

# Si falla, especificar versión con wheel conocido
pip install av==10.0.0 --only-binary=:all:
```

**Nota:** Los wheels de PyAV incluyen FFmpeg estáticamente enlazado, no requieren FFmpeg del sistema.

### Solución 3: Usar Conda (Más Confiable)

Conda maneja dependencias nativas automáticamente:

```bash
# Crear entorno con conda
conda create -n brandpulse python=3.11
conda activate brandpulse

# Instalar PyAV desde conda-forge (incluye FFmpeg)
conda install -c conda-forge av

# Continuar con pip para otras dependencias
pip install -r requirements.txt
```

### Solución 4: Versión Específica Compatible

Si todo falla, usar una versión específica de PyAV:

```bash
# PyAV 10.0.0 tiene mejor compatibilidad
pip install av==10.0.0

# O versión anterior más estable
pip install av==9.2.0
```

---

## Matriz de Compatibilidad

| PyAV Version | Python | FFmpeg | Ubuntu | Wheel Available |
|--------------|--------|--------|--------|-----------------|
| 12.x | 3.9-3.12 | 6.x | 22.04+ | Linux x86_64 |
| 11.x | 3.8-3.11 | 5.x-6.x | 20.04+ | Linux x86_64 |
| 10.x | 3.8-3.11 | 4.x-5.x | 20.04+ | Linux x86_64 |
| 9.x | 3.7-3.10 | 4.x | 18.04+ | Linux x86_64 |

---

## Integración con GitHub Codespaces

### Opción A: devcontainer.json

Crear `.devcontainer/devcontainer.json`:

```json
{
  "name": "BrandPulse Dev",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/common-utils:2": {}
  },
  "postCreateCommand": "bash scripts/install_pyav_deps.sh && pip install -r backend/requirements.txt",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  }
}
```

### Opción B: Dockerfile Personalizado

Crear `.devcontainer/Dockerfile`:

```dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install FFmpeg and development libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install PyAV
RUN pip install av==10.0.0
```

Y en `devcontainer.json`:

```json
{
  "name": "BrandPulse Dev",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "postCreateCommand": "pip install -r backend/requirements.txt"
}
```

---

## Integración con GitHub Actions

### Workflow para CI

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install FFmpeg dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            ffmpeg \
            libavcodec-dev \
            libavformat-dev \
            libavutil-dev \
            libswscale-dev \
            libswresample-dev \
            pkg-config

      - name: Install Python dependencies
        run: |
          pip install --upgrade pip
          pip install -r backend/requirements.txt

      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v
```

### Workflow con Cache (Más Rápido)

```yaml
# .github/workflows/test-cached.yml
name: Tests (Cached)

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache apt packages
        uses: awalsh128/cache-apt-pkgs-action@latest
        with:
          packages: ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev pkg-config
          version: 1.0

      - name: Cache pip packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r backend/requirements.txt

      - name: Run tests
        run: pytest backend/tests/ -v
```

---

## Troubleshooting

### Error: "libavcodec.h: No such file or directory"

**Causa:** Faltan headers de desarrollo de FFmpeg.

```bash
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev
```

### Error: "pkg-config: command not found"

**Causa:** pkg-config no está instalado.

```bash
sudo apt-get install pkg-config
```

### Error: "'AV_CODEC_FLAG_GLOBAL_HEADER' undeclared"

**Causa:** Versión de FFmpeg incompatible con PyAV.

```bash
# Verificar versión de FFmpeg
ffmpeg -version

# Si es FFmpeg 6.x, usar PyAV 12.x
pip install av>=12.0.0

# Si es FFmpeg 4.x-5.x, usar PyAV 10.x
pip install av==10.0.0
```

### Error: "Could not find a version that satisfies the requirement av"

**Causa:** No hay wheel disponible para tu plataforma.

```bash
# Forzar compilación desde source
pip install av --no-binary av

# O usar conda
conda install -c conda-forge av
```

### Error en MacOS: "xcrun: error: invalid active developer path"

```bash
xcode-select --install
brew install ffmpeg pkg-config
pip install av
```

---

## Script de Verificación

Después de instalar, verificar que funciona:

```python
# test_pyav.py
import av

print(f"PyAV version: {av.__version__}")
print(f"FFmpeg version: {av.library_versions}")

# Test básico de lectura
# container = av.open("test_video.mp4")
# for frame in container.decode(video=0):
#     print(f"Frame: {frame}")
#     break

print("PyAV installed correctly!")
```

```bash
python test_pyav.py
```

---

## Resumen de Pasos

```
┌─────────────────────────────────────────────────────────┐
│           Instalación de PyAV - Flujo Recomendado       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Intentar wheel pre-compilado                        │
│     pip install av --only-binary=:all:                  │
│              │                                          │
│              ▼                                          │
│     ┌────────────────┐                                  │
│     │   ¿Funciona?   │                                  │
│     └────────────────┘                                  │
│        │ SÍ    │ NO                                     │
│        ▼       ▼                                        │
│     [DONE]  2. Instalar deps + compilar                 │
│             ./install_pyav_deps.sh                      │
│             pip install av                              │
│                    │                                    │
│                    ▼                                    │
│             ┌────────────────┐                          │
│             │   ¿Funciona?   │                          │
│             └────────────────┘                          │
│                │ SÍ    │ NO                             │
│                ▼       ▼                                │
│             [DONE]  3. Usar conda                       │
│                     conda install -c conda-forge av     │
│                            │                            │
│                            ▼                            │
│                     [DONE]                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

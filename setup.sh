#!/usr/bin/env bash
# setup.sh
# =========
# Crea el entorno virtual, instala dependencias y verifica la instalación.
#
# Uso:
#   chmod +x setup.sh
#   ./setup.sh
#
# Variables de entorno opcionales:
#   PYTHON  : Intérprete a usar (por defecto: python3)
#   VENV    : Nombre del directorio del venv (por defecto: .venv)

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
REQUIREMENTS="requirements.txt"

# ── Colores para output ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Verificar Python ──────────────────────────────────────────────────────────
info "Verificando Python..."
command -v "$PYTHON" >/dev/null 2>&1 || error "Python no encontrado. Instala Python 3.10+."

PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python detectado: $PYTHON_VERSION"

MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    error "Se requiere Python 3.10+. Versión encontrada: $PYTHON_VERSION"
fi

# ── Crear entorno virtual ─────────────────────────────────────────────────────
if [ -d "$VENV" ]; then
    warn "El directorio '$VENV' ya existe. Se omite la creación."
else
    info "Creando entorno virtual en '$VENV'..."
    "$PYTHON" -m venv "$VENV"
fi

# ── Activar entorno virtual ───────────────────────────────────────────────────
info "Activando entorno virtual..."
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ── Actualizar pip ────────────────────────────────────────────────────────────
info "Actualizando pip..."
pip install --upgrade pip --quiet

# ── Instalar dependencias ─────────────────────────────────────────────────────
if [ ! -f "$REQUIREMENTS" ]; then
    error "Archivo '$REQUIREMENTS' no encontrado. Ejecuta desde la raíz del proyecto."
fi

info "Instalando dependencias desde $REQUIREMENTS..."
pip install -r "$REQUIREMENTS" --quiet

# ── Instalar paquete en modo editable ─────────────────────────────────────────
info "Instalando paquete en modo editable (pip install -e .)..."
pip install -e . --quiet

# ── Crear directorios de salida ───────────────────────────────────────────────
info "Creando directorios de salida..."
mkdir -p outputs/figures outputs/models outputs/results notebooks

# ── Copiar .env si no existe ──────────────────────────────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    warn ".env creado desde .env.example. Revisa y ajusta las variables."
fi

# ── Verificación de instalación ───────────────────────────────────────────────
info "Verificando instalación de paquetes clave..."

PACKAGES=("numpy" "pandas" "sklearn" "tensorflow" "matplotlib" "yfinance")
ALL_OK=true

for pkg in "${PACKAGES[@]}"; do
    if "$PYTHON" -c "import $pkg" 2>/dev/null; then
        VERSION=$("$PYTHON" -c "import $pkg; print(getattr($pkg, '__version__', 'OK'))" 2>/dev/null || echo "OK")
        info "  ✓ $pkg ($VERSION)"
    else
        warn "  ✗ $pkg — no importable"
        ALL_OK=false
    fi
done

# ── Resumen ───────────────────────────────────────────────────────────────────
echo ""
if [ "$ALL_OK" = true ]; then
    info "════════════════════════════════════════════════"
    info "  Entorno configurado correctamente."
    info "  Para ejecutar el pipeline:"
    info "    source $VENV/bin/activate"
    info "    python main.py"
    info "════════════════════════════════════════════════"
else
    warn "Algunos paquetes no se importaron correctamente."
    warn "Revisa los errores anteriores."
fi

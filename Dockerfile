# 1. Imagem Base
FROM python:3.11-slim

# 2. Pasta de Trabalho
WORKDIR /app

# --- A CURA DO PROBLEMA 👇 ---
# Instala bibliotecas do Linux essenciais para ML (OpenMP e Fortran)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------

# 3. Dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia o Modelo e o Código
COPY model_prod /app/model_prod
COPY . .

# 5. Configurações
EXPOSE 8000
CMD ["python", "src/app.py"]
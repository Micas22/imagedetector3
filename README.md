# ImageDetector 3

O ImageDetector 3 é uma ferramenta para rastrear websites, extrair imagens e classificá-las como tabelas ou imagens normais utilizando machine learning.

O projecto inclui:

* Interface web em Streamlit
* Backend em FastAPI
* Suporte opcional para OCR
* Docker para facilitar a instalação

---

# Instalação

## Executar com Docker (Recomendado)

A forma mais simples de correr o projecto é através do Docker Compose.

```bash id="5dd3h8"
# Entrar na pasta do projecto
cd imagedetector3

# Fazer build e iniciar os containers
docker-compose up -d --build
```

Depois de iniciar:

* Interface Web: `http://localhost:8501`
* API: `http://localhost:8000/docs`

Este setup também:

* Guarda automaticamente a base de dados SQLite
* Faz cache dos modelos de IA
* Instala todas as dependências necessárias

---
# ImageDetector 3

[English](#english) | [Português](#portugues)

---

<a id="english"></a>
## English Version

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)

**ImageDetector 3** is a powerful, containerized web crawler and machine-learning tool built to automatically scan websites, extract images, and intelligently classify them as either **tables** or **normal** images. It provides both a rich **Streamlit Web UI** for interactive use and a **FastAPI REST API** for integrations.

---

### Key Features

- **Advanced Web Crawling:** Scan a single page, entire domain, specific URLs, or even auto-detect paginated listings (like news or events). Powered by Playwright and BeautifulSoup4.
- **ML Image Classification:** Uses PyTorch, HuggingFace Transformers, and Timm to perform robust image classification.
- **OCR Integration:** Optional PaddleOCR integration to detect text density and improve table identification.
- **Performance Presets:** Choose between `Full Precision` (maximum accuracy) or `Full Speed` (turbo mode with aggressive skipping).
- **Smart Caching:** SQLite-backed cache (`.crawler.db`) remembers previously classified images to speed up repeated scans and save compute resources.
- **Interactive Dashboard:** Beautiful Streamlit UI to configure scans, view real-time progress, browse history, and download results as CSV.
- **Docker Ready:** Includes a `docker-compose.yml` for effortless deployment, caching of HF models, and strict memory management.

---

### Architecture & Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Backend / API:** [FastAPI](https://fastapi.tiangolo.com/) & Uvicorn
- **Crawling / Scraping:** [Playwright](https://playwright.dev/python/), Requests, BeautifulSoup4
- **Machine Learning:** PyTorch, Torchvision, Transformers, Timm, Safetensors
- **OCR:** PaddleOCR & PaddlePaddle (Optional, gracefully skipped if unavailable)
- **Database:** SQLite (built-in)

---

### Getting Started

#### 1. Running with Docker (Recommended)

The easiest way to run ImageDetector 3 is via Docker Compose. This ensures all ML dependencies (like PyTorch and PaddlePaddle) and browser drivers (Playwright) are perfectly configured.

```bash
# Clone the repository and navigate into it
cd imagedetector3

# Build and start the containers
docker-compose up -d --build
```

**What this does:**
- Starts the Streamlit Web UI on `http://localhost:8501`
- Starts the FastAPI REST API on `http://localhost:8000`
- Mounts a volume to persist the SQLite database (`.crawler.db`)
- Caches HuggingFace models to avoid re-downloading on every restart

#### 2. Running Locally (Without Docker)

If you prefer to run it directly on your host machine:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Playwright browsers
playwright install

# 4. Start the application
./start.sh
```
*(The `start.sh` script handles launching both Streamlit and FastAPI simultaneously).*

---

### How to Use the UI

1. **Open the Web UI:** Navigate to `http://localhost:8501`.
2. **Select Evaluation Mode:**
   - **Scan page/site:** Enter a URL and choose the scan scope (Single page, Whole website, Specific URLs, or Paginated listing).
   - **Evaluate image(s):** Upload an image file directly or paste an image URL to test the classification model.
3. **Configure Presets:** 
   - **Full Precision:** Highest accuracy, best for complex images.
   - **Full Speed:** Uses Turbo mode and Fast OCR for rapid processing.
   - **Custom:** Manually adjust the Table Confidence Threshold and toggle Turbo/OCR modes.
4. **Run the Scan:** Hit start and watch the live progress. Once completed, you can review the results and download a detailed **CSV report**.
5. **Review History:** Switch to the `History & Cache` tab to view past runs and manage the image classification cache.

---

### Project Structure

- `webapp.py` & `webapp_*.py`: Streamlit frontend application files.
- `api.py`: FastAPI endpoints and routing logic.
- `orchestrator.py`: Core logic orchestrating the crawling and processing tasks.
- `classifier.py` & `table_transformer_adapter.py`: Machine Learning classification and model loading.
- `fetchers.py` & `parsers.py`: Playwright/BS4 web scraping and DOM parsing utilities.
- `queue_manager.py`: Handles async task queuing for the crawlers.
- `database.py`: SQLite operations, caching, and history management.
- `docker-compose.yml` & `Dockerfile`: Containerization configs.

---

### Notes & Troubleshooting

- **Memory Constraints:** The default `docker-compose.yml` sets a hard memory limit of `1g` for the container. If you encounter Out-Of-Memory (OOM) kills during heavy model inference, consider increasing this limit in `docker-compose.yml`.
- **First Run:** The first time you scan an image, the system will download the necessary HuggingFace models. This may take a few minutes depending on your internet connection. Subsequent runs will use the cached models.

---
*Built with passion for automated visual data extraction.*

<br><br>

---

<a id="portugues"></a>
## Versão em Português

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)

O **ImageDetector 3** é uma ferramenta de machine learning e web crawler conteinerizada e poderosa, construída para rastrear sites automaticamente, extrair imagens e classificá-las de forma inteligente como **tabelas** ou imagens **normais**. Ele fornece tanto uma **Interface Web em Streamlit** rica para uso interativo quanto uma **REST API em FastAPI** para integrações.

---

### Principais Funcionalidades

- **Web Crawling Avançado:** Rastreie uma única página, um domínio inteiro, URLs específicas ou até mesmo liste páginas paginadas detectadas automaticamente (como notícias ou eventos). Desenvolvido com Playwright e BeautifulSoup4.
- **Classificação de Imagens com ML:** Usa PyTorch, HuggingFace Transformers e Timm para realizar a classificação robusta de imagens.
- **Integração OCR:** Integração opcional com PaddleOCR para detectar densidade de texto e melhorar a identificação de tabelas.
- **Predefinições de Desempenho:** Escolha entre `Full Precision` (Precisão Total, para máxima precisão) ou `Full Speed` (Velocidade Máxima, modo turbo com pulos agressivos).
- **Cache Inteligente:** O cache apoiado em SQLite (`.crawler.db`) lembra imagens previamente classificadas para acelerar verificações repetidas e economizar recursos computacionais.
- **Painel Interativo:** Uma interface maravilhosa em Streamlit para configurar verificações, visualizar o progresso em tempo real, navegar pelo histórico e baixar os resultados como CSV.
- **Pronto para Docker:** Inclui um `docker-compose.yml` para implantação sem esforço, armazenamento de cache de modelos HF e gerenciamento restrito de memória.

---

### Arquitetura e Tecnologias

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Backend / API:** [FastAPI](https://fastapi.tiangolo.com/) & Uvicorn
- **Crawling / Scraping:** [Playwright](https://playwright.dev/python/), Requests, BeautifulSoup4
- **Machine Learning:** PyTorch, Torchvision, Transformers, Timm, Safetensors
- **OCR:** PaddleOCR & PaddlePaddle (Opcional, graciosamente ignorado se indisponível)
- **Banco de Dados:** SQLite (embutido)

---

### Primeiros Passos

#### 1. Executando com Docker (Recomendado)

A maneira mais fácil de executar o ImageDetector 3 é através do Docker Compose. Isso garante que todas as dependências de ML (como PyTorch e PaddlePaddle) e drivers do navegador (Playwright) estejam perfeitamente configurados.

```bash
# Clone o repositório e navegue até ele
cd imagedetector3

# Construa e inicie os contêineres
docker-compose up -d --build
```

**O que isso faz:**
- Inicia a Interface Web do Streamlit em `http://localhost:8501`
- Inicia a REST API do FastAPI em `http://localhost:8000`
- Monta um volume para persistir o banco de dados SQLite (`.crawler.db`)
- Faz o cache dos modelos HuggingFace para evitar re-downloads em cada reinicialização

#### 2. Executando Localmente (Sem Docker)

Se preferir rodar diretamente na sua máquina local:

```bash
# 1. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Instale os navegadores do Playwright
playwright install

# 4. Inicie a aplicação
./start.sh
```
*(O script `start.sh` gerencia a inicialização simultânea do Streamlit e do FastAPI).*

---

### Como usar a Interface do Usuário

1. **Abra a Interface Web:** Navegue para `http://localhost:8501`.
2. **Selecione o Modo de Avaliação:**
   - **Scan page/site (Escanear página/site):** Insira um URL e escolha o escopo de verificação (Página única, Site inteiro, URLs específicas ou Listagem paginada).
   - **Evaluate image(s) (Avaliar imagens):** Faça o upload direto de um arquivo de imagem ou cole o URL da imagem para testar o modelo de classificação.
3. **Configure as Predefinições:** 
   - **Full Precision:** Maior precisão, ideal para imagens complexas.
   - **Full Speed:** Usa o Modo Turbo e OCR rápido para processamento acelerado.
   - **Custom:** Ajuste manualmente o limite de confiança para tabelas e alterne os modos Turbo/OCR.
4. **Inicie o Escaneamento:** Pressione iniciar e observe o progresso ao vivo. Quando concluído, você pode revisar os resultados e baixar um **relatório CSV** detalhado.
5. **Revise o Histórico:** Mude para a aba `History & Cache` (Histórico e Cache) para visualizar as execuções anteriores e gerenciar o cache de classificação de imagens.

---

### Estrutura do Projeto

- `webapp.py` & `webapp_*.py`: Arquivos de aplicação do frontend em Streamlit.
- `api.py`: Endpoints do FastAPI e lógica de roteamento.
- `orchestrator.py`: Lógica central orquestrando as tarefas de rastreamento e processamento.
- `classifier.py` & `table_transformer_adapter.py`: Classificação de Machine Learning e carregamento de modelos.
- `fetchers.py` & `parsers.py`: Utilitários de raspagem de web (Playwright/BS4) e parsing de DOM.
- `queue_manager.py`: Lida com a fila de tarefas assíncronas para os rastreadores.
- `database.py`: Operações do SQLite, cache e gerenciamento de histórico.
- `docker-compose.yml` & `Dockerfile`: Configurações de containerização.

---

### Notas e Solução de Problemas

- **Restrições de Memória:** O `docker-compose.yml` padrão estabelece um limite restrito de memória de `1g` para o contêiner. Se você encontrar encerramentos por Falta de Memória (OOM) durante inferências de modelo pesadas, considere aumentar esse limite no `docker-compose.yml`.
- **Primeira Execução:** Na primeira vez que você escanear uma imagem, o sistema baixará os modelos necessários do HuggingFace. Isso pode levar alguns minutos dependendo da sua conexão de internet. As execuções subsequentes usarão os modelos em cache.

---
*Construído com paixão para extração automatizada de dados visuais.*

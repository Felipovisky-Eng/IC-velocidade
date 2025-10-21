# 🚀 IC-VELOCIDADE — Sistema de Aquisição e Análise de Velocidade

Este repositório contém o conjunto de códigos e dados utilizados no estudo experimental e na análise digital da **velocidade de fitas em movimento**, a partir de sinais capturados por um **sistema optoeletrônico** com **ESP32** e scripts em **Python**.

O projeto foi desenvolvido com finalidade acadêmica e experimental, com foco em **instrumentação eletrônica** e **processamento digital de sinais (PDS)** — oferecendo um fluxo completo: aquisição → armazenamento → pré-análise → análise avançada.

---

# Estrutura do repositório

```
IC-VELOCIDADE/
│
├── src/                 # Códigos-fonte principais (ESP32 e Python)
│   ├── Envio-ESP32-Serial.cpp
│   ├── Leitura-ESP32-Serial.py
│   ├── Grafico-simples.py
│   ├── ICPYthon_fita_fina.py
│   └── ICPYthon_fita_grossa.py
│
├── testes/              # Arquivos .txt de teste (amostras experimentais)
│   ├── exemplo_1.txt
│   ├── exemplo_2.txt
│   └── ...
│
└── README.md            # Este arquivo
```

---

# Visão geral dos arquivos

- **`src/Envio-ESP32-Serial.cpp`**  
  Código para ESP32 que realiza leituras do ADC e envia os dados pela porta Serial. Contém temporização e protocolo simples de envio compatível com o script Python de leitura.

- **`src/Leitura-ESP32-Serial.py`**  
  Script Python que lê a Serial (via `pyserial`) e salva os dados em `.csv`. Inclui salvamento com timestamp, verificação de erros e opção de gravação em binário/texto.

- **`src/Grafico-simples.py`**  
  Script para inspeção rápida do `.csv` ou `.txt`: plota ADC × número de amostras, tempo × ADC e avalia jitter/estabilidade. Útil para recortar trechos válidos.

- **`src/ICPYthon_fita_fina.py`**  
  Pipeline de análise voltado a fitas finas e velocidades aproximadamente constantes. Contém filtragem, detecção de picos/vales, cálculo de velocidade e RPM, FFT e plots informativos.

- **`src/ICPYthon_fita_grossa.py`**  
  Versão adaptada para fitas de maior espessura e velocidade variável. Utiliza filtros menos agressivos, interpolação para melhorar derivadas e rotinas de equalização de envelope.

- **`testes/`**  
  Conjunto de arquivos `.txt` com aquisições reais para testes e validação dos scripts.

---

# Fluxo de uso recomendado

1. **Aquisição**
   - Carregue `Envio-ESP32-Serial.cpp` na ESP32.
   - Execute `Leitura-ESP32-Serial.py` no computador para gravar os dados em `.csv`.

2. **Pré-análise / inspeção**
   - Rode `Grafico-simples.py` para visualizar o sinal, conferir jitter e selecionar trechos.
   - Salve o trecho desejado em `.txt` dentro de `testes/` (ou outro local de sua preferência).

3. **Análise avançada**
   - Use `ICPYthon_fita_fina.py` para trechos de fita fina e velocidade estável.
   - Use `ICPYthon_fita_grossa.py` para fitas grossas ou velocidade variável.
   - Ambos os scripts geram gráficos de velocidade, aceleração, FFT e estatísticas.

---

# Dependências (Python)

Recomenda-se criar um ambiente virtual (venv) e instalar as dependências abaixo:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install numpy scipy matplotlib pandas pyserial
```

Versão mínima recomendada do Python: **3.8+** (preferível 3.10+).

---

# Boas práticas e recomendações

- **Verifique a alimentação e aterramento** do sistema ESP32/sensor antes de coletar dados (flutuações afetam amplitude e drift).  
- **Aplique filtros analógicos** (anti-aliasing) no hardware sempre que possível; isso melhora a performance das reconstruções e diferenciações numéricas.  
- **Documente cada experimento** com parâmetros: `FS`, ganho do sensor, distância sensor→alvo, tipo de fita, e condições ambientais.  
- Ao postar resultados, inclua o arquivo de entrada `.txt`/`.csv` usado para reproduzir figuras.

---

# Objetivo acadêmico

Este projeto visa:
- Demonstrar um pipeline completo de aquisição e análise experimental com `ESP32 + Python`.
- Permitir estudo prático de técnicas de PDS: filtragem IIR/FIR, interpolação, diferenciação numérica, FFT e equalização de envelope.
- Servir como base para trabalhos de TCC e relatórios científicos.

---

# Autor

**Luis Felipe Pereira Ramos**  
Instituto Federal do Mato Grosso (IFMT)

---

# Licença

Disponibilizado para fins **acadêmicos e educacionais**. Sinta-se livre para usar e adaptar o código, desde que mantenha a referência ao autor.

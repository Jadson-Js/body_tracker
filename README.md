# 📄 **Documento Técnico — Monitor de Postura com MediaPipe + OpenCV**

---

## ⚠️ Problemas, riscos e limitações do projeto

- **Dependência da iluminação**: MediaPipe perde precisão com pouca luz ou luz forte atrás do usuário.
- **Webcam fraca → detecção instável**: câmeras com ruído ou baixa resolução prejudicam o cálculo do ângulo.
- **Falso positivo em movimentos rápidos**: o modelo não é otimizado para exercícios intensos.
- **Hardcoded**: limites de 170°/195° são genéricos; não funcionam bem para todas as anatomias.
- **Sem multithreading**: leitura da câmera + inferência + desenho ocorre em sequência; pode gerar travamentos.
- **Sem calibração inicial**: assume que a postura neutra é exatamente 180°, o que raramente é verdade.
- **Dependência total do arquivo `.task`**: se o path mudar ou o arquivo não existir, o sistema quebra.

---

# 1. 📦 Instalação e Ambiente

Este projeto requer:

- Python **3.9+**
- pip atualizado
- Webcam funcional
- Sistema com AVX/CPU moderna para rodar o MediaPipe Tasks

### 1.1. Instalação das dependências

```bash
pip install opencv-python mediapipe numpy
```

Caso o `mediapipe` falhe em máquinas Linux:

```bash
pip install mediapipe --index-url https://google.github.io/mediapipe/pypi/simple/
```

### 1.2. Estrutura de diretórios recomendada

```
seu_projeto/
│
├── main.py
├── utils.py
└── model/
    └── pose_landmarker_heavy.task
```

### 1.3. Verificação de webcam

Before running:

```bash
python - <<EOF
import cv2; print(cv2.VideoCapture(0).isOpened())
EOF
```

Se retornar `False`, a câmera não está disponível.

---

# 2. 🎯 Objetivo do Projeto

Criar um sistema de **monitoramento de postura em tempo real**, usando a webcam para:

1. Detectar o corpo via MediaPipe Pose Landmarker.
2. Calcular o ângulo entre ombro–quadril–joelho.
3. Emitir alertas visuais se o usuário estiver curvado.

A aplicação funciona como um **assistente de ergonomia**, útil para alunos, trabalhadores ou pessoas que passam muito tempo sentadas.

---

# 3. 🧠 Tecnologias Utilizadas

- **MediaPipe Tasks (Pose Landmarker)**
  Modelo de IA para detectar 33 pontos do corpo.

- **OpenCV**
  Captura da webcam, desenho, exibição.

- **NumPy**
  Cálculo de ângulos.

---

# 4. 🧩 Arquitetura do Código

O sistema é dividido em:

### 4.1. `main.py`

Controla captura de vídeo, execução da IA, cálculos e alertas.

### 4.2. `utils.py`

Contém:

- Função de desenho dos landmarks.
- Função de cálculo matemático do ângulo.
- Função especializada para o ângulo da coluna.

### 4.3. Pasta `model/`

Armazena o modelo `.task` do MediaPipe.

---

# 5. 🔍 Funcionamento Interno

### 5.1. Captura da Webcam

O OpenCV lê os frames continuamente.

### 5.2. Conversão e envio para a IA

O frame BGR é convertido para RGB → enviado ao Pose Landmarker.

### 5.3. Detecção dos pontos do corpo

A IA retorna a lista de 33 landmarks normalizados (x, y, z).

### 5.4. Cálculo do ângulo

Usa-se:

- Ombro esquerdo
- Quadril esquerdo
- Joelho esquerdo

O ângulo é calculado via trigonometria (diferença de vetores + atan2).

### 5.5. Lógica de alerta

| Condição                | Ação                             |
| ----------------------- | -------------------------------- |
| Ângulo < 170° ou > 195° | Borda vermelha + texto de alerta |
| Ângulo entre 170°–195°  | “POSTURA OK” em verde            |

---

# 6. 📈 Utilidade do Sistema

- Monitoramento ergonômico durante estudos ou trabalho.
- Base para apps de fisioterapia digital.
- Plataforma para treinar modelos de movimento.
- Extensível para análise de exercícios.

---

# 7. 🛠 Melhorias Recomendadas

- Calibração inicial do usuário (definir o “180° personalizado”).
- Filtragem temporal (ex.: filtro de Kalman) para reduzir tremores.
- Suporte a multithreading com `cv2.CAP_DSHOW` ou threading nativo.
- Exportação de métricas (CSV, API REST).
- Ajuste automático de threshold baseado em altura real.
- Interface gráfica independente da janela do OpenCV.

---

Se quiser, posso gerar:

- slides (PowerPoint ou markdown),
- versão resumida,
- versão mais técnica para professores,
- diagrama do fluxo,
- ou explicação verbal para apresentação.

Como prefere?

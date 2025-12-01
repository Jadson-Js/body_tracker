# --- IMPORTAÇÃO DE FERRAMENTAS ---
# Importa as soluções prontas do MediaPipe (ferramentas de desenho, estilos, etc).
from mediapipe import solutions

# Importa um formato de dados específico (Protocol Buffers) que o MediaPipe usa para organizar os pontos do corpo.
from mediapipe.framework.formats import landmark_pb2

# Importa o NumPy, a biblioteca de matemática mais famosa do Python.
# Vamos chamá-lo de "Calculadora Científica".
import numpy as np


# --- FUNÇÃO 1: O DESENHISTA ---
# Essa função recebe a imagem original e o resultado da IA, e devolve a imagem rabiscada.
def draw_landmarks_on_image(rgb_image, detection_result):
  # Pega a lista de corpos detectados (geralmente é só 1, você).
  pose_landmarks_list = detection_result.pose_landmarks
  
  # Cria uma cópia da imagem original para não estragar a original. Vamos desenhar na cópia.
  annotated_image = np.copy(rgb_image)

  # Loop: Para cada pessoa detectada na imagem...
  for idx in range(len(pose_landmarks_list)):
    # Pega os pontos (landmarks) dessa pessoa específica.
    pose_landmarks = pose_landmarks_list[idx]

    # --- PARTE TÉCNICA (TRADUÇÃO) ---
    # O desenhista do MediaPipe é chato. Ele exige os dados num formato muito específico chamado "Proto".
    # Aqui estamos convertendo nossa lista simples de pontos para esse formato "Proto" chique.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    
    # Preenche esse formato chique com os nossos dados x (horizontal), y (vertical) e z (profundidade).
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    
    # --- AÇÃO DE DESENHAR ---
    # Chama a função oficial do Google para desenhar.
    solutions.drawing_utils.draw_landmarks(
      annotated_image,         # Onde desenhar (na nossa cópia)
      pose_landmarks_proto,    # O que desenhar (os pontos traduzidos)
      solutions.pose.POSE_CONNECTIONS, # Como ligar os pontos (ex: ligar cotovelo no ombro)
      solutions.drawing_styles.get_default_pose_landmarks_style()) # Qual cor e espessura usar (estilo padrão)
      
  # Devolve a imagem pronta, toda rabiscada com o esqueleto.
  return annotated_image


# --- FUNÇÃO 2: A CALCULADORA DE ÂNGULOS ---
# Essa é uma função genérica. Se você der 3 pontos (A, B, C), ela diz qual o ângulo no ponto B.
def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos (a, b, c).
    b é o ponto central (o vértice do ângulo).
    """
    # Converte os pontos para "arrays do numpy" para podermos fazer contas complexas fácil.
    a = np.array(a) # Primeiro ponto (ex: Ombro)
    b = np.array(b) # Ponto do meio (ex: Quadril) - O ângulo é medido aqui!
    c = np.array(c) # Ponto final (ex: Joelho)

    # --- MATEMÁTICA PURA ---
    # Usa a função arco-tangente (atan2) para descobrir a inclinação das linhas e subtrai uma da outra.
    # O resultado vem em radianos (unidade matemática), então multiplicamos por 180/PI para virar Graus.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi) # np.abs garante que o número seja positivo.

    # O corpo humano não gira 360 graus como uma roda.
    # Se o ângulo passar de 180, subtraímos de 360 para pegar o ângulo interno (o menor).
    # Ex: Em vez de dizer 350 graus, dizemos 10 graus.
    if angle > 180.0:
        angle = 360 - angle

    # Devolve o valor final (ex: 175 graus).
    return angle


# --- FUNÇÃO 3: O ESPECIALISTA EM COLUNA ---
# Essa função usa a calculadora acima especificamente para checar se você está curvado.
def calculate_angle_rosca(landmarks, mp_pose):
    # --- MAPEAMENTO ---
    # Pega os números de identificação (IDs) das partes do corpo que nos interessam.
    # O MediaPipe numera o corpo todo (ex: nariz é 0, ombro esquerdo é 11, etc).
    idx_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value # ID do Ombro Esquerdo
    idx_hip = mp_pose.PoseLandmark.LEFT_HIP.value           # ID do Quadril Esquerdo
    idx_knee = mp_pose.PoseLandmark.LEFT_KNEE.value         # ID do Joelho Esquerdo

    # --- EXTRAÇÃO DE COORDENADAS ---
    # Vai na lista de pontos (landmarks) e pega o X e Y desses IDs específicos.
    shoulder = [landmarks[idx_shoulder].x, landmarks[idx_shoulder].y]
    hip = [landmarks[idx_hip].x, landmarks[idx_hip].y]
    knee = [landmarks[idx_knee].x, landmarks[idx_knee].y]

    # --- CÁLCULO FINAL ---
    # Chama a função matemática (calculate_angle) passando:
    # A=Ombro, B=Quadril, C=Joelho.
    # Isso vai medir se o quadril está reto (180°) ou dobrado.
    angle_back = calculate_angle(shoulder, hip, knee)
    
    # Devolve o ângulo das costas para o programa principal decidir se dá o ALERTA.
    return angle_back
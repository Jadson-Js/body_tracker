# --- IMPORTAÇÃO DE FERRAMENTAS ---
# Importa o OpenCV, a ferramenta que nos permite acessar a câmera e mexer nas imagens.
import cv2 

# Importa o MediaPipe, a "inteligência" do Google que sabe reconhecer corpos e rostos.
import mediapipe as mp 

# Importa funções que criamos no outro arquivo (utils.py) para desenhar o esqueleto e calcular matemática.
from utils import draw_landmarks_on_image, calculate_angle_rosca 

# Define onde está salvo o arquivo da IA (o "cérebro" treinado para reconhecer poses) no seu computador.
model_path = './model/pose_landmarker_heavy.task' 

# Liga a webcam. O número '0' geralmente representa a câmera padrão do notebook/USB.
webcam = cv2.VideoCapture(0) 

# --- CONFIGURAÇÃO DA INTELIGÊNCIA ARTIFICIAL (MEDIAPIPE) ---
# Aqui estamos criando "apelidos" para comandos longos do MediaPipe para facilitar a escrita.
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Aqui configuramos as regras da IA:
options = PoseLandmarkerOptions(
    # Dizemos onde está o arquivo do modelo (o cérebro).
    base_options=BaseOptions(model_asset_path=model_path),
    # Pedimos para a IA gerar máscaras de segmentação (separar o corpo do fundo), se necessário.
    output_segmentation_masks=True,
    # Avisamos a IA que vamos mandar um VÍDEO (sequência de fotos), não uma foto parada. Isso ajuda ela a ser mais suave.
    running_mode=VisionRunningMode.VIDEO
)

# Carrega a solução de pose para podermos usar nomes dos pontos do corpo (ex: nariz, cotovelo).
mp_pose = mp.solutions.pose

# Cria um relógio interno começando do zero. O modo VÍDEO precisa saber o tempo de cada quadro.
timestamp = 0 

# --- VERIFICAÇÃO DE SEGURANÇA ---
# Pergunta: "A câmera ligou de verdade?"
if not webcam.isOpened():
    print("Error ao abrir a câmera") # Se não ligou, avisa o erro.
    exit() # E encerra o programa imediatamente.

# --- INÍCIO DO PROGRAMA PRINCIPAL ---
# Carrega a IA com as opções que definimos acima. O 'with' garante que ela feche sozinha se der erro.
with PoseLandmarker.create_from_options(options) as landmarker:
    
    # Loop Infinito: "Faça isso para sempre (até alguém mandar parar)".
    while True:
        # Avança o relógio interno em 30.000 microssegundos (simulando 33ms por frame, padrão de vídeo).
        timestamp += 30000
        
        # Tira uma foto (frame) da webcam.
        # 'success' diz se funcionou (True/False), 'frame' é a imagem em si.
        success, frame = webcam.read() 
        
        # Se a foto falhou (câmera desconectou), para o loop.
        if not success:
            break

        # Espelha a imagem horizontalmente (como um espelho de banheiro). 
        # Sem isso, levantar a mão esquerda parece a direita na tela, o que confunde o usuário.
        frame = cv2.flip(frame, 1)

        # O OpenCV vê cores como Azul-Verde-Vermelho (BGR). O MediaPipe prefere Vermelho-Verde-Azul (RGB).
        # Aqui fazemos essa conversão de cores.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transforma a imagem em um formato especial que o MediaPipe consegue ler.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        
        # --- A MÁGICA ACONTECE AQUI ---
        # Manda a imagem e o tempo atual para a IA analisar: "Onde estão as partes do corpo aqui?"
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Pega o resultado da IA e desenha os risquinhos coloridos (esqueleto) em cima da foto original.
        annotated_image = draw_landmarks_on_image(frame, result)

        # --- LÓGICA DE ANÁLISE DE POSTURA ---
        # Verifica: "A IA encontrou algum corpo na imagem?"
        if result.pose_landmarks:
            # Pega a lista de pontos do primeiro corpo encontrado (índice 0).
            landmarks = result.pose_landmarks[0] 
            
            # Chama sua função matemática para calcular o ângulo das costas.
            # Ela usa os pontos do ombro, quadril e joelho (provavelmente).
            angle_back = calculate_angle_rosca(landmarks, mp_pose)

            # Verifica: "O cálculo deu certo?" (Às vezes retorna None se o corpo estiver escondido).
            if angle_back is not None:
                
                # 1. VISUALIZAÇÃO (DEBUG)
                # Escreve o número do ângulo na tela em amarelo, para você acompanhar em tempo real.
                cv2.putText(annotated_image, f"Angulo: {int(angle_back)}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                # 2. SISTEMA DE ALERTA
                # Se o ângulo for menor que 170 (curvado p/ frente) OU maior que 195 (curvado p/ trás).
                if angle_back < 170 or angle_back > 195:
                    
                    # Escreve "ALERTA: COLUNA!" em vermelho na tela.
                    cv2.putText(annotated_image, "ALERTA: COLUNA!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                    
                    # Desenha uma borda grossa vermelha ao redor de toda a tela para chamar atenção.
                    cv2.rectangle(annotated_image, (0,0), (640,480), (0,0,255), 10)
                
                # Se não estiver errado (está entre 170 e 195), então está certo.
                else:
                    # Escreve "POSTURA OK" em verde.
                    cv2.putText(annotated_image, "POSTURA OK", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- EXIBIÇÃO ---
        # Cria uma janela no Windows/Mac e mostra a imagem final (com esqueleto e textos).
        cv2.imshow("Webcam - Q para sair", annotated_image)

        # Espera 1 milissegundo por uma tecla. Se a tecla for 'q' (código ascii), sai do loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# --- LIMPEZA ---
# "Solta" a câmera para que outros programas possam usá-la.
webcam.release()
# Fecha todas as janelas que o Python abriu.
cv2.destroyAllWindows()
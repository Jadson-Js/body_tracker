import cv2 # Biblioteca para  lidar com a webcam
import mediapipe as mp # Blib para visão computacional
from utils import draw_landmarks_on_image, calculate_angle_rosca
model_path = './model/pose_landmarker_heavy.task' # modelo ia para analise de frames

webcam = cv2.VideoCapture(0) # Habilita a webcam padrão

# Configurações iniciais do mediapipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.VIDEO
)
mp_pose = mp.solutions.pose
timestamp = 0 # Inicialmente o timestamp será 0

# Condicional para validar webcam
if not webcam.isOpened():
    print("Error ao abrir a câmera")
    exit()

# PCria uma instancia do landmarker carregando as options
with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        timestamp += 30000
        success, frame = webcam.read() # da function read, será extraido o status de sucesso e o frame da webcam
        if not success:
            break

        # Espelha a tela para visão mais intuitiva
        frame = cv2.flip(frame, 1)

        # converte em uma array matriz de rgb
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        
        # A IA analisa o frame e retornar o resultado
        result = landmarker.detect_for_video(mp_image, timestamp)

        # O resultado é enviado para o util que desenha o "esquelo" com bases nos pontos do corpo, e é retornado o desenho do esqueleto
        annotated_image = draw_landmarks_on_image(frame, result)

        if result.pose_landmarks:
            # 2. Pega os pontos da primeira pessoa detectada
            landmarks = result.pose_landmarks[0] 
            # 3. Passa os pontos para a função
            angle_back = calculate_angle_rosca(landmarks, mp_pose)
        if angle_back < 170 or angle_back > 190:
            cv2.putText(annotated_image, "ALERTA: POSTURA!", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            
            # Opcional: Desenhar uma caixa vermelha em volta da tela
            cv2.rectangle(annotated_image, (0,0), (640,480), (0,0,255), 10)
            
        else:
            # Se estiver tudo bem, mostra um texto verde discreto
            cv2.putText(annotated_image, "OK", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # O desenho do esqueleto é renderizado na tela
        cv2.imshow("Webcam - Q para sair", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break





webcam.release()
cv2.destroyAllWindows()

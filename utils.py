from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos (a, b, c).
    b é o ponto central (o vértice do ângulo).
    """
    a = np.array(a) # Primeiro ponto (ex: Ombro)
    b = np.array(b) # Ponto do meio (ex: Quadril)
    c = np.array(c) # Ponto final (ex: Joelho)

    # Cálculo matemático usando arco tangente (atan2)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    # O ângulo pode passar de 180, então normalizamos
    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_angle_rosca(landmarks, mp_pose):
    # Índices
    idx_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    idx_hip = mp_pose.PoseLandmark.LEFT_HIP.value
    idx_knee = mp_pose.PoseLandmark.LEFT_KNEE.value

    shoulder = [landmarks[idx_shoulder].x, landmarks[idx_shoulder].y]
    hip = [landmarks[idx_hip].x, landmarks[idx_hip].y]
    knee = [landmarks[idx_knee].x, landmarks[idx_knee].y]

    angle_back = calculate_angle(shoulder, hip, knee)
    return angle_back
            

       

       

      

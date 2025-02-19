# file: src/visualization/drawer.py
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

"""
The LandmarkDrawer class is responsible for drawing face, hand, gesture and pose landmarks on an image. 
"""


class LandmarkDrawer:
    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)

    @staticmethod
    def draw_landmarks(image, face_result, hand_result, gesture_result, pose_result):
        annotated_image = image.copy()

        if face_result and face_result.face_landmarks:
            LandmarkDrawer._draw_face_landmarks(annotated_image, face_result.face_landmarks)

        if hand_result and hand_result.hand_landmarks:
            LandmarkDrawer._draw_hand_landmarks(
                annotated_image,
                hand_result.hand_landmarks,
                hand_result.handedness
            )
        
        if gesture_result and gesture_result.gestures:
            LandmarkDrawer._draw_gesture_results(annotated_image, gesture_result.gestures)
        
        if pose_result and pose_result.pose_landmarks:
            LandmarkDrawer._draw_pose_landmarks(annotated_image, pose_result.pose_landmarks)
        

        return annotated_image

    @staticmethod
    def _draw_face_landmarks(image, face_landmarks_list):
        for face_landmarks in face_landmarks_list:
            # Convert Tasks API landmarks to Solutions API format
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for landmark in face_landmarks:
                landmark_list.landmark.add(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )

            # Draw face landmarks using converted protobuf object
            solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_list,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_list,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_list,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    @staticmethod
    def _draw_hand_landmarks(image, hand_landmarks_list, handedness_list):
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx] if idx < len(handedness_list) else None

            # Convert hand landmarks to protobuf format
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for landmark in hand_landmarks:
                landmark_list.landmark.add(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )

            # Draw hand landmarks
            solutions.drawing_utils.draw_landmarks(
                image,
                landmark_list,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            if handedness:
                height, width, _ = image.shape
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coords) * width)
                text_y = int(min(y_coords) * height) - LandmarkDrawer.MARGIN
                cv2.putText(image, handedness[0].category_name,
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            LandmarkDrawer.FONT_SIZE, LandmarkDrawer.HANDEDNESS_TEXT_COLOR,
                            LandmarkDrawer.FONT_THICKNESS, cv2.LINE_AA)
                

    @staticmethod
    def _draw_gesture_results(image, gestures):
        height, _ = image.shape[:2]
        text_y = height - 20  # Position text at bottom of frame
        
        for idx, gesture in enumerate(gestures):
            category_name = gesture[0].category_name
            score = gesture[0].score
            # Draw gesture name and confidence score
            text = f"{category_name}: {score:.2f}"
            cv2.putText(
                image,
                text,
                (10, text_y - (30 * idx)),  # Stack multiple gestures vertically
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # Green color
                2
            )


    @staticmethod   
    def _draw_pose_landmarks(image, pose_landmarks_list):
        for pose_landmarks in pose_landmarks_list:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=pose_landmarks_proto,
                connections=solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=None)

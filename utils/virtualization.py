def draw_face_landmarks(image, face_landmarks):
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    # Draw tesselation
    solutions.drawing_utils.draw_landmarks(
        image=image,
        landmark_list=face_landmarks_proto,
        connections=solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style())

    # Draw contours
    solutions.drawing_utils.draw_landmarks(
        image=image,
        landmark_list=face_landmarks_proto,
        connections=solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style())

    # Draw irises
    solutions.drawing_utils.draw_landmarks(
        image=image,
        landmark_list=face_landmarks_proto,
        connections=solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style())


def draw_hand_landmarks(image, hand_landmarks, handedness):
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=image,
        landmark_list=hand_landmarks_proto,
        connections=solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style())

    # Draw handedness text
    if handedness:
        height, width, _ = image.shape
        x_coords = [landmark.x for landmark in hand_landmarks]
        y_coords = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - 10  # MARGIN

        cv2.putText(image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1, (88, 205, 54), 1, cv2.LINE_AA)
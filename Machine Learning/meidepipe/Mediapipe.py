import cv2
import mediapipe as mp

mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#Static image ->
# image_file = []
# with mp_hands.Hands(
#     static_image_mode = True,
#     max_num_hands = 2,
#     min_detection_confidence =0.5
# ) as hands:
#     for idx, file in enumerate(image_file):
#         #Read iamge
#         #Flip around y-axis for correct handness output
#         image = cv2.flip(cv2.imread(file), 1)
#
#         #Convert the BGR to RGB
#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#         #Print handedness and draw hand landmarks on image
#         print('Handedness: ', results.multi_handedness)
#
#         if not results.multi_hand_landmarks:
#             continue
#
#         image_height, image_width, _ =image.shape
#         annotated_iamge = image.copy()
#         for hands_landmarks in results.multi_hand_landmarks:
#             print('hand_landmarks: ', hands_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hands_landmarks.landmark[mp_hands,HandLandmark.INDEX_FINGER_TIP].x *image_width},'
#                 f'{hands_landmarks.landmark[mp_hands,HandLandmark.INDEX_FINGER_TIP].y *image_height},)'
#             )
#
#             mp_drawing.draw_landmarks(
#                 annotated_iamge,
#                 hands_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#             cv2.imwrite(
#                 '/temp/annotated_image' + str(idx)+'.png', cv2.flip(annotated_image,1))
#
#

#WEB CAM input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")

            #IF loading image , use break not continue
            continue

        #Flip image horizontally for later selfie-view
        #display
        #COnvert BGR to RGB.

        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)

        #Impropve performance, optionally mark image as not writable
        #to pass by reference

        image.flags.writeable = False
        results = hands.process(image)

        # DRAW hand annotation ->
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hands_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hands_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.imshow('Mediapipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
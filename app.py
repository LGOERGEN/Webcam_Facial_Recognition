import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load images
angelina_image = face_recognition.load_image_file("Angelina/angelina.jpg")
angelina_face_encoding = face_recognition.face_encodings(angelina_image)[0]

benedict_image = face_recognition.load_image_file("Benedict/benedict.jpg")
benedict_face_encoding = face_recognition.face_encodings(benedict_image)[0]


known_face_encodings = [
    angelina_face_encoding,
    benedict_face_encoding,
]
known_face_names = [
    "Angelina",
    "Benedict"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]


    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        r=15
        d=30
        t=4
        colour=(127,255,255)

        # Draw top left corner
        cv2.line(frame,(left+r,top),(left+r+d,top),colour,t)
        cv2.line(frame,(left,top+r),(left,top+r+d),colour,t)
        cv2.ellipse(frame,(left+r,top+r),(r,r),180,0,90,colour,t)

        # Draw top right corner
        cv2.line(frame,(right-r,top),(right-r-d,top),colour,t)
        cv2.line(frame,(right,top+r),(right,top+r+d),colour,t)
        cv2.ellipse(frame,(right-r,top+r),(r,r),270,0,90,colour,t)

        # Draw bottom left corner
        cv2.line(frame,(left+r,bottom),(left+r+d,bottom),colour,t)
        cv2.line(frame,(left,bottom-r), (left,bottom-r-d),colour,t)
        cv2.ellipse(frame, (left + r, bottom - r), (r, r), 90, 0, 90, colour, t)

        #Draw bottom right corner
        cv2.line(frame, (right - r, bottom), (right - r - d, bottom), colour, t)
        cv2.line(frame, (right, bottom - r), (right, bottom - r - d), colour, t)
        cv2.ellipse(frame, (right - r, bottom - r), (r, r), 0, 0, 90, colour, t)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom+20 ), (right, bottom+60), colour, cv2.FILLED)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, name, (left+15, bottom+45), font, 0.8, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'Esc' to quit
    if cv2.waitKey(1) & 0xFF == ord('\x1b'):
        break

# Webcam
video_capture.release()
cv2.destroyAllWindows()
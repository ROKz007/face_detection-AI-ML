import cv2
import os

# Load Haar cascade reliably
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

data = []

# Open default webcam (0 = default camera)
cam = cv2.VideoCapture(0)

while len(data) < 100:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Detect faces
    face_points = classifier.detectMultiScale(frame, 1.3, 5)

    if len(face_points) > 0:
        for x, y, w, h in face_points:
            face_frame = frame[y:y + h, x:x + w]  # cropped face
            cv2.imshow("Only face", face_frame)

            if len(data) < 100:
                print(len(data) + 1, "/100")
                data.append(face_frame)
                break  # take only one face per frame

    # Display progress on main frame
    cv2.putText(frame, str(len(data)), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.imshow("frame", frame)

    if cv2.waitKey(30) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

# Save collected face samples
if len(data) == 100:
    name = input("Enter Face holder name: ")
    if not os.path.exists("images"):
        os.makedirs("images")  # create folder if not exists
    for i in range(100):
        cv2.imwrite(f"images/{name}_{i}.jpg", data[i])
    print("Done ✅")
else:
    print("Need more data")

import cv2
import cs
import os
import face_recognition


cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
eyePath=os.path.dirname(cv2.__file__)+"/data/haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
video_capture = cv2.VideoCapture(0)
num=1
ctr=0
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    _,orig  =   video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,255),2)
        face = frames[y:y+h, x:x+w]
        status = cv2.imwrite('faces_detected'+str(ctr)+'.jpg',  face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
        max=0
        for (ex,ey,ew,eh) in eyes:
            if max!=2:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                max+=1
    # Display the resulting frame
    cv2.imshow('Video', frames)
    
    ctr+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()

for i in range(ctr):
    for j in range(i+1,ctr):
        if i!=j:
            try:
                print("Comparing ",i," and ",j," images...")
                image1 = face_recognition.load_image_file("faces_detected"+str(i)+".jpg")
                image2 = face_recognition.load_image_file("faces_detected"+str(j)+".jpg")
                face1 = face_recognition.face_locations(image1)
                face2 = face_recognition.face_locations(image2)
                one = face_recognition.face_encodings(image1,known_face_locations=face1)
                if len(one) > 0:
                    one = one[0]
                else:
                    print("No faces found in the image!")
                    os.remove("faces_detected"+str(i)+".jpg")
                    continue
                two = face_recognition.face_encodings(image2,known_face_locations=face2)
                if len(two) > 0:
                    two = two[0]
                else:
                    print("No faces found in the image!")
                    os.remove("faces_detected"+str(j)+".jpg")
                    continue
                result = face_recognition.compare_faces([one], two,tolerance=0.5)
                print(result)
                if result[0]:
                    os.remove("faces_detected"+str(j)+".jpg")
            except FileNotFoundError:
                continue

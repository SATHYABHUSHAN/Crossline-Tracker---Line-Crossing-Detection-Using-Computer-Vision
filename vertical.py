from ultralytics import YOLO
import cv2


model = YOLO("yolov8s.pt") 


line_x = 300 #
line_color = (255,0 , 0)
line_thickness = 2



cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model.predict(source=frame, conf=0.55, save=False, show=False, verbose=False)

    detections = results[0].boxes.data.cpu().numpy() 

   
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), line_color, line_thickness)

    
    crossline_detected = False
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_id = int(cls)
        
       
        if class_id == 0: 
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            
            
            if x2 > line_x >= x1:  
                crossline_detected = True

 
    if crossline_detected:
        cv2.putText(frame, "Crossline detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    cv2.imshow("Crossline Detection", frame)

   
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

# Load pre-trained car detection classifier
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Load video file
video_capture = cv2.VideoCapture('carsone.mp4')

# Initialize car counter
car_count = 0
detected_cars = set()

while True:
    # Read each frame of the video
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected cars and count them
    for (x, y, w, h) in cars:
        car_key = (x, y, w, h)
        if car_key not in detected_cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            detected_cars.add(car_key)
            car_count += 1
    
    # Display car count on the frame
    cv2.putText(frame, f'Car Count: {car_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with car detections
    cv2.imshow('Car Detection', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()
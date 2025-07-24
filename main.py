from ultralytics import YOLO                                                                                    # Bu Algoritmadaki amaç ilk frame de tracklediği nesnelerden alan hesabına göre önce en küçük olanın 
from deep_sort_realtime.deepsort_tracker import DeepSort                                                        # trck_id sini aldıktan sonra o nesne kaybolana kadar onu takipte olacak şekilde merkez kkordinatlarını 
import cv2                                                                                                      # current_x ve current_y değerleri ile eşlemesi. Buradaki asıl amaç anlık olarak alan değişiklikleri 
import torch                                                                                                    # yaşandığı için sistemin kafasının karışmamasıdır.
import numpy as np
import threading
from simplebgc_serial_protocol import *

#--------------------------------------------------------------------------------------------------
current_x = None   
current_y = None    
current_w = None
current_h = None

follow_id = None
target_x = None                             #Görüntü işleme için
target_y = None
target_w = None
target_h = None
flag = None
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for detecting objects
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Cihaz: {'GPU' if device == 'cuda' else 'CPU'} kullanılıyor.")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
model_path = r'C:\\Users\\aktas\\Desktop\\Python\\deepsortV1\\HavaSavunmaBest.pt'
model = YOLO(model_path)
tracker = DeepSort(max_age=40,nms_max_overlap=0.5,max_iou_distance=0.9)
#---------------------------------------------------------------------------------------------------

# Kameranın başarılı bir şekilde açılıp açılmadığını kontrol et
if not cam.isOpened():
    print("Hata: Kamera açılamadı. Lütfen kamera bağlantınızı kontrol edin.")
    exit()

print("Kameradan görüntü okunuyor... Çıkmak için 'q' tuşuna basın.")

class PID:
    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d
        self.last_error_x = 0
        self.last_error_y = 0
        self.integral_x = 0
        self.integral_y = 0

    def compute(self, error_x, error_y):
        # Proportional term
        p_term_x = self.p * error_x
        p_term_y = self.p * error_y

        # Integral term
        self.integral_x += error_x
        self.integral_y += error_y
        i_term_x = self.i * self.integral_x
        i_term_y = self.i * self.integral_y

        # Derivative term
        d_term_x = self.d * (error_x - self.last_error_x) / (1/30)
        d_term_y = self.d * (error_y - self.last_error_y) / (1/30)

        # Update last errors
        self.last_error_x = error_x
        self.last_error_y = error_y

        return p_term_x + i_term_x + d_term_x, p_term_y + i_term_y + d_term_y

    def reset(self):
        self.last_error = 0
        self.integral = 0

def initialize_bgc():
    print(execute_menu(11))
    pass

def track(pid : PID, width, height, x, y, w, h):

    cx, cy = x + w // 2, y + h // 2
    
    # calculate the error
    error_x = (cx - width // 2)
    error_y = (cy - height // 2)

    x, y = pid.compute(error_x, error_y)
    
    cx_diff, cy_diff = abs(width // 2 - cx), abs(height // 2 - cy)
    
    if cx_diff < 60 and cy_diff < 60:
        print("Hedefe ulaşıldı, PID kontrolü sıfırlanıyor.")
        pid.reset()
    
    print(f"Error X: {error_x}, Error Y: {error_y}")
    print(f"Control X: {x}, Control Y: {y}")
    
    motor_x = error_x * 50
    motor_y = error_y * 50
    
    if motor_x > 30000:
        motor_x = 30000
    elif motor_x < -30000:
        motor_x = -30000
        
    if motor_y > 30000:
        motor_y = 30000
    elif motor_y < -30000:
        motor_y = -30000
    
    control_motors(ControlMode.Speed, 0, motor_y, motor_x, 0, 0, 0)
    

track_objects = []  #Görüntü işlemede işimize yarayan bir liste
while True:
    initialize_bgc()

    pid = PID(0.1, 0.01, 0)

    # 2. Kameradan anlık bir kare (frame) oku
    ret, frame = cam.read()
    if not ret:
        print("Hata: Görüntü alınamadı.")
        break
    detections = model(frame)[0]
    results = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
            
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence,class_id])
            
    tracks = tracker.update_tracks(results,frame=frame)
        
    for track in tracks :
        if not track.is_confirmed():
            continue

        track_id = track.track_id        # Get the track ID
        track_class = track.det_class    # Takip deki nesnenin sınıfını belirtir.
        ltrb = track.to_ltrb()           # Get the bounding box coordinates
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        area = int((xmax-xmin)*(ymax-ymin))
        if track_class == 1:
            target_x = int(xmin)
            target_y = int(ymin)
            target_w = int(xmax-xmin)
            target_h = int(ymax-ymin)
            follow_id = track_id
            track_objects.append([area,follow_id,target_x,target_y,target_w,target_h])    
        # Draw the bounding box and the track ID on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0),2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), (0, 255, 0), -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)

    track_objects.sort()
        
    print(track_objects)
        
    try:
        if track_objects[0][1] != flag:
            t = False
            print("for a girecek")
            for i in range(0,len(track_objects)):
                if track_objects[i][1] == flag:
                    current_x = track_objects[i][2]
                    current_y = track_objects[i][3] 
                    current_w = track_objects[i][4]
                    current_h = track_objects[i][5]
                    flag = track_objects[i][1]
                    print(f"for içinde {current_x}, id {track_objects[i][1]}")
                    t = True
                    break
            if t == False:                      #bu kısım ilk başta None olan flag değerini ayarlamak için sonrasında bu kısma girilmiyor.
                flag = track_objects[0][1]    
                t = False
                        
        else:
            current_x = track_objects[0][2]
            current_y = track_objects[0][3]  
            current_w = track_objects[0][4]
            current_h = track_objects[0][5]
            print(f"Else içinde {current_x} id {track_objects[0][1]}")
            flag = track_objects[0][1]  
                
        track_objects.clear()           
    except:
        print("Nesne Yok") 

    main_center = (frame.shape[1]/2, frame.shape[0]/2)
    
    cv2.circle(frame, main_center, 5, (0, 0, 255), -1)

    # Merkezin etrafına 30x30 piksellik bir çerçeve çiz (15 piksel sağa-sola, yukarı-aşağı)
    cv2.rectangle(frame,
                (main_center[0] - 30, main_center[1] - 30),
                (main_center[0] + 30, main_center[1] + 30),
                (255, 0, 0), 2)  # Mavi renkli çerçeve
    
    if current_x != None:
        center_x = int(current_x + (current_w / 2))
        center_y = int(current_y + (current_h / 2))
        center = (center_x,center_y)
    
        t1 = threading.Thread(target=track, args=(pid, frame.shape[1], frame.shape[0], current_x, current_y, current_w, current_h))
        t1.start()
        t1.join()

        # Orta noktaya küçük bir daire çiz
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Orta nokta koordinatlarını ekrana yazdır
        text = f"Orta Nokta: ({center[0]}, {center[1]})"
        cv2.putText(frame, text, (center[0] - 50, center[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'HEDEF', (int(current_x + (current_w / 2)),int(current_y + (current_h / 2))), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

    cv2.imshow("TEST", frame)

    # 'q' tuşuna basıldığında döngüden çık ve programı sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. Döngü bittiğinde kaynakları serbest bırak ve pencereleri kapat
cam.release()
cv2.destroyAllWindows()
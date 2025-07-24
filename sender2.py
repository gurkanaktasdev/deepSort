from ultralytics import YOLO                                                                                    # Bu Algoritmadaki amaç ilk frame de tracklediği nesnelerden alan hesabına göre önce en küçük olanın 
from deep_sort_realtime.deepsort_tracker import DeepSort                                                        # trck_id sini aldıktan sonra o nesne kaybolana kadar onu takipte olacak şekilde merkez kkordinatlarını 
import cv2                                                                                                      # current_x ve current_y değerleri ile eşlemesi. Buradaki asıl amaç anlık olarak alan değişiklikleri 
import torch   
import cv2
import numpy as np
import socket
import time                                         # UI dan process ile erşip çalıştırdığımız dosya
import struct
import zlib


current_x = None    #kordinat noktası sol üstte olduğu duruma göre hesaplandır
current_y = None    # Alanı en Küçük olan balnonun merkez kordinatları
current_w = None
current_h = None

follow_id = None
target_x = None
target_y = None
target_w = None
target_h = None
flag = None
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for detecting objects
        

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Cihaz: {'GPU' if device == 'cuda' else 'CPU'} kullanılıyor.")
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
model_path = r'C:\\Users\\aktas\\Desktop\\Python\\deepsortV1\\HavaSavunmaBest.pt'
model = YOLO(model_path)
tracker = DeepSort(max_age=40,nms_max_overlap=0.5,max_iou_distance=0.9)
  
track_objects = []
# UDP settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5007
FRAGMENT_SIZE = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

header_format = "<HHHHI"  # frame_id, frag_id, total_frags, frag_len, crc32

time.sleep(0.03)

# Camera setup
cam= cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("Kamera acilmadi! Lutfen baglantiyi kontrol edin.")
    exit()

frame_id = 0
target_frame_time = 1/30

# High-quality settings for speed
encode_params = [
    int(cv2.IMWRITE_JPEG_QUALITY), 85,
    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
    int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0
]

USE_JPEG = False

while True:
    start_time = time.time()
    ret,frame = cam.read()
    if not ret:
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

        center_x = int(current_x + (current_w / 2))
        center_y = int(current_y + (current_h / 2))
        center = (center_x,center_y)    
        cv2.circle(frame, center, 5, (0,255,0), -1)
        hedef_x = int(current_x + current_w - 50)
        hedef_y = int(current_y - 10)
        cv2.putText(frame, 'HEDEF', (hedef_x,hedef_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        track_objects.clear()           
    except:
        print("Nesne Yok")
        
    
    if USE_JPEG:
        ret, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            continue
    else:
        ret, encoded = cv2.imencode(".png", frame)
        if not ret:
            continue
        
    data = bytearray(encoded)
    fragments = [data[i:i+FRAGMENT_SIZE] for i in range(0, len(data), FRAGMENT_SIZE)]
    total_fragments = len(fragments)
    
    for frag_id, frag_data in enumerate(fragments):
        crc32 = zlib.crc32(frag_data) & 0xffffffff
        header = struct.pack(header_format, frame_id, frag_id, total_fragments, len(frag_data), crc32)
        
        try:
            sock.sendto(header + frag_data, (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"UDP error: {e}")
            continue
    
    frame_id = (frame_id + 1) % 65536
    
    elapsed = time.time() - start_time
    if elapsed < target_frame_time:
        time.sleep(target_frame_time - elapsed)

cam.release()
sock.close()
cv2.destroyAllWindows()
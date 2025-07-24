
from ultralytics import YOLO                                                                                    # Bu Algoritmadaki amaç ilk frame de tracklediği nesnelerden alan hesabına göre önce en küçük olanın 
from deep_sort_realtime.deepsort_tracker import DeepSort                                                        # trck_id sini aldıktan sonra o nesne kaybolana kadar onu takipte olacak şekilde merkez kkordinatlarını 
import cv2                                                                                                      # current_x ve current_y değerleri ile eşlemesi. Buradaki asıl amaç anlık olarak alan değişiklikleri 
import torch                                                                                                    # yaşandığı için sistemin kafasının karışmamasıdır.


current_x = None    #kordinat noktası sol üstte olduğu duruma göre hesaplandır
current_y = None    # Alanı en Küçük olan balnonun merkez kordinatları

follow_id = None
target_x = None
target_y = None
flag = None


def Tracktor():
    global target_x,target_y,follow_id,current_x,current_y,flag
    CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for detecting objects
    GREEN = (0, 255, 0)  
    WHITE = (255, 255, 255)  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Cihaz: {'GPU' if device == 'cuda' else 'CPU'} kullanılıyor.")

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    model_path = r'C:\\Users\\aktas\\Desktop\\Python\\deepsortV1\\HavaSavunmaBest.pt'
    model = YOLO(model_path)
    tracker = DeepSort(max_age=40,nms_max_overlap=0.5,max_iou_distance=0.9)
  
    track_objects = []
    while True:
   
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
            area, center_x, center_y= AlanHesapla(xmin,ymin,xmax,ymax)
            if track_class == 1:
                target_x = int(center_x)
                target_y = int(center_y)
                follow_id = track_id
                track_objects.append([area,follow_id,target_x,target_y])    
            # Draw the bounding box and the track ID on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN,2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.circle(frame,(center_x,center_y),5,(0,255,0),3)

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
                print(f"Else içinde {current_x} id {track_objects[0][1]}")
                flag = track_objects[0][1]  
    
            track_objects.clear()           
        except:
            print("Nesne Yok")       

        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("q"):
            break  # Exit the loop if 'q' key is pressed



    cam.release()
    cv2.destroyAllWindows()

def AlanHesapla(xmin,ymin,xmax,ymax):
    
    alan = int((xmax-xmin)*(ymax-ymin))
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)

    return alan,center_x,center_y 
    




Tracktor()  
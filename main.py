import cv2
import warnings
warnings.filterwarnings("ignore")

def put_text(pic, x1, y1, x2, y2, color, text):
    h, w, _ = pic.shape
    fontSize = max(0.6, min(1, (y2-y1)/300))
    textSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontSize, 1)
    if y1 < textSize[1] + 10:
        y1_text = y1 + textSize[1] + 5
    else:
        y1_text = y1 - 10
    if ((x1 + textSize[0]) > w):
        x1_text = x1 - ((x1 + textSize[0])-w)
    else: x1_text = x1
    cv2.rectangle(pic, (x1_text , y1_text - textSize[1] - baseLine), (x1_text + textSize[0], y1_text + baseLine), color, cv2.FILLED)
    cv2.putText(pic,text,(x1_text, y1_text),cv2.FONT_HERSHEY_SIMPLEX,fontSize,(0, 0, 0), 1)
    return pic


def ocrDetect(img, x1, y1, x2, y2, reader):
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    speed = reader.readtext(crop, detail=0, allowlist='0123456789')
    if speed:
        if speed[0] != "" :
            if int(speed[0]) != 0:
                return int(speed[0])
    return ''

def picDetect(imgPath=None, pic=None, model=None, reader = None):
    if imgPath is not None:
        pic = cv2.imread(imgPath)
    box_colors = [(0, 255, 255),(128, 0, 128),(255, 0, 0),(0, 165, 255),
                (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 0, 255)]
    resizedpic = cv2.resize(pic, (640, 640))
    h, w, _ = pic.shape
    x_ratio = w / 640
    y_ratio = h / 640
    names = model.names
    results = model.predict(resizedpic, conf = 0.45 , verbose=False)
    result = results[0]
    boxes = result.boxes
    for box in boxes:
        cls = box.cls.item()
        clsName = names[cls]
        conf = f"{int(box.conf.item()*100)}%"
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = [int(x1 * x_ratio), int(y1 * y_ratio), 
                          int(x2 * x_ratio), int(y2 * y_ratio)]
        color = box_colors[int(cls)]
        cv2.rectangle(pic, (x1, y1), (x2, y2), color, 2)
        if clsName == "Speed_Limit":
            speed = ocrDetect(pic, x1, y1, x2, y2, reader)
            text = f'{clsName} {speed}: {conf}'
        else: 
            text = f'{clsName}: {conf}'
        pic = put_text(pic, x1, y1, x2, y2, color, text)
    pic = cv2.resize(pic, (w, h))
    return pic



def vidDetect(vidPath, model=None, reader = None, outputPath=None):
    vid = cv2.VideoCapture(vidPath)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    fps = int(vid.get(cv2.CAP_PROP_FPS)) or 30
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(outputPath,fourcc, fps, (width, height))
    frame_count = 0
    while True:
        isFrame, frame = vid.read() # bool is there a pic, frame
        if not isFrame:
            break
        frame_count += 1
        if frame_count % 15 == 0:
            out.write(frame)
            continue
        out.write(picDetect(pic=frame, model=model, reader=reader))
    vid.release()
    out.release()


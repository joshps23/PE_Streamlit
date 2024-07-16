import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import imutils
from deep_sort_realtime.deepsort_tracker import DeepSort
import av
import cv2
import math
import numpy as np

from sample_utils.turn import get_ice_servers

st.slider("Threshold1",0,1000,100)
st.slider("Threshold2",0,1000,200)

model=YOLO("ballDetectBestV2.pt")
classNames = ["ball"
              ]
names = None
cache_id=[]

object_tracker = DeepSort(max_iou_distance=0.7,
                            max_age=30,
                            n_init=3,
                            nms_max_overlap=1.0,
                            max_cosine_distance=0.2,
                            nn_budget=None,
                            gating_only_position=False,
                            override_track_class=None,
                            embedder="mobilenet",
                            half=True,
                            bgr=True,
                            embedder_gpu=True,
                            embedder_model_name=None,
                            embedder_wts=None,
                            polygon=False,
                            today=None
                            )

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    results=model(img,stream=True)

    if names == None:
        names = results[0].names
    for r in results:
        detections = []
        for data in r.boxes.data.tolist():
            _,_, _,_, conf, id = data
            name = names[id]
            detections.append(data)
            img = draw_Box(data,img,name)

        details = get_details(r, img)
        tracks = object_tracker.update_tracks(details, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            break
        track_id = track.track_id
        bbox = track.to_ltrb()
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (200,0, 0), 6)


    x = bbox[0]
    y = bbox[1]
    id = track_id
    
    # boxes=r.boxes
    # for box in boxes:
    #     x1,y1,x2,y2=box.xyxy[0]
    #     x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
    #     print(x1,y1,x2,y2)
    #     cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
    #     conf=math.ceil((box.conf[0]*100))/100
    #     cls=int(box.cls[0])
    #     class_name=classNames[cls]
    #     label=f'{class_name}{conf}'
    #     t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    #     print(t_size)
    #     c2 = x1 + t_size[0], y1 - t_size[1] - 3
    #     cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        # yield img
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def draw_Box(data,image,name):
    x1,y1,x2,y2,conf,id = data
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))
    cv2.rectangle(image,p1,p2,(0,0,255),1)
    cv2.putText(image,name,p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    return image

def get_details(result,image):

    classes = result.boxes.cls.numpy()
    conf = result.boxes.conf.numpy()
    xywh = result.boxes.xywh.numpy()

    detections = []
    for i,item in enumerate(xywh):
        sample = (item,conf[i] ,classes[i])
        detections.append(sample)

    return detections

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
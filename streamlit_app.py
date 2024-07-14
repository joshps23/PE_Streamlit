import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import cv2

from sample_utils.turn import get_ice_servers

st.slider("Threshold1",0,1000,100)
st.slider("Threshold2",0,1000,200)

model=YOLO("ballDetectBestV2.pt")
classNames = ["ball"
              ]

def callback(frame: av.VideoFrame) -> av.VideoFrame:
  img = frame.to_ndarray(format="bgr24")
  results=model(img,stream=True)
  for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        # yield img
  return av.VideoFrame.from_ndarray(img, format="bgr24")

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
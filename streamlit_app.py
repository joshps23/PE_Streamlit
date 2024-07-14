import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import cv2

from sample_utils.turn import get_ice_servers

st.slider("Threshold1",0,1000,100)
st.slider("Threshold2",0,1000,200)

def callback(frame: av.VideoFrame) -> av.VideoFrame:
  img = frame.to_ndarray(format="bgr24")
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
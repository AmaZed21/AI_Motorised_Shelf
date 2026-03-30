import streamlit as st
import streamlit.components.v1 as components
from simulator import Logger, Compartment, Shelf, Voice, STATE_STOPPED, STATE_MOVING_UP, STATE_MOVING_DOWN, VOICE_AVAILABLE
import pandas as pd
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode #type: ignore
import av
import vosk
import json
import os
import zipfile
import urllib.request

st.set_page_config(page_title="Shelf Control", layout="wide")
st.title("Motorised Shelf Dashboard")

def ensure_model():
    model_path = "models/vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        zip_path = "models/vosk-model.zip"
        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        with st.spinner("Downloading voice model..."):
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall("models/")
            os.remove(zip_path)

if 'shelf' not in st.session_state:
    com_1 = Compartment(1, weight=0.5)
    com_2 = Compartment(2, weight=0.4)
    com_3 = Compartment(3)
    st.session_state.shelf = Shelf([com_1, com_2, com_3])
    st.session_state.logger = Logger('data/logs.csv')
    st.session_state.last_command = ""
    st.session_state.command_queue = []

    if VOICE_AVAILABLE:
        ensure_model()
        model = vosk.Model("models/vosk-model-small-en-us-0.15")
        st.session_state.recognizer = vosk.KaldiRecognizer(model, 16000)

shelf  = st.session_state.shelf
logger = st.session_state.logger

shelf.update_all(0.1)

#Compartment selection
with st.sidebar:
    st.subheader("Select Compartments")
    for c in shelf.total_com:
        label = f"Compartment {c.com_no} — {', '.join(c.contents) if c.contents else 'Empty'}"
        st.checkbox(label, key=f"com_check_{c.com_no}")

selected_coms = [c for c in shelf.total_com if st.session_state.get(f"com_check_{c.com_no}", False)]

#Graphics
col_vis, col_ctrl, col_log = st.columns([5, 2, 5])

with col_vis:
    st.subheader("Cabinet View")

    cab_cols = st.columns(len(shelf.total_com), gap="small")

    for i, com in enumerate(shelf.total_com):
        with cab_cols[i]:
            pct = max(0.0, min(com.position / Compartment.MAX_HEIGHT, 1.0))
            top = (1.0 - pct) * 84

            color = {
                STATE_STOPPED: "#888888",
                STATE_MOVING_UP: "#00cc44",
                STATE_MOVING_DOWN: "#ff0000",
            }.get(com.state, "#888888")

            border = "#ffffff" if st.session_state.get(f"com_check_{com.com_no}", False) else "#444444"

            cabinet_html = f"""
            <html>
            <body style="margin:0; padding:0; background:white; overflow:hidden;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="
                        position:relative;
                        width:80px;
                        height:300px;
                        border:3px solid {border};
                        border-radius:8px;
                        background:#1a1a2e;
                    ">
                        <div style="
                            position:absolute;
                            left:50%;
                            top:0;
                            bottom:0;
                            width:4px;
                            background:#333;
                            transform:translateX(-50%);
                        "></div>

                        <div style="
                            position:absolute;
                            left:8px;
                            right:8px;
                            top:{top}%;
                            height:10%;
                            background:{color};
                            border-radius:6px;
                            box-shadow:0 0 8px {color};
                        "></div>

                        <div style="
                            position:absolute;
                            bottom:0;
                            left:0;
                            right:0;
                            height:6px;
                            background:#555;
                            border-radius:0 0 6px 6px;
                        "></div>
                    </div>
                </div>
            </body>
            </html>
            """
            components.html(cabinet_html, height=320, width=90, scrolling=False)

            st.markdown(
                f"<div style='text-align:center; margin-top:-6px;'>"
                f"<div style='font-weight:600;'>Compartment {com.com_no}: {', '.join(com.contents) if com.contents else 'Empty'}</div>"
                f"<div style='color:{color}; font-size:13px;'>{com.state}</div>"
                f"<div style='font-size:13px;'>{com.position:.1f} cm</div>"
                f"</div>",
                unsafe_allow_html=True
            )

#Controls
with col_ctrl:
    st.subheader("Controls")

    if not selected_coms:
        st.warning("Select at least one compartment.")
    else:
        for com in selected_coms:
            st.markdown(f"**#{com.com_no}** `{com.state}` `{com.position:.0f} cm` `{com.speed:.2f} cm/s`")

    st.divider()

    if st.button("⬆"):
        for com in selected_coms:
            com.move_up()
            logger.log(com, 'COMMAND_UP')

    if st.button("⬇"):
        for com in selected_coms:
            com.move_down()
            logger.log(com, 'COMMAND_DOWN')

    if st.button("⏹"):
        for com in selected_coms:
            com.stop()
            logger.log(com, 'COMMAND_STOP')

    if st.button("Add Obstruction"):
        for com in selected_coms:
            com.sensor_distance = 0.5
            logger.log(com, 'OBSTRUCTION_INJECTED')

    if st.button("Remove Obstruction"):
        for com in selected_coms:
            com.sensor_distance = com.position + 2.0
            logger.log(com, 'OBSTRUCTION_REMOVED')

    if st.button("Inject Overload"):
        for com in selected_coms:
            com.weight = 2.0
            logger.log(com, 'OVERLOAD_INJECTED')

    if st.button("Remove Overload"):
        for com in selected_coms:
            com.weight = 0.5
            logger.log(com, 'OVERLOAD_REMOVED')

    if st.button("Reset System"):
        shelf.reset()
        for c in shelf.total_com:
            logger.log(c, 'RESET')

    if not VOICE_AVAILABLE:
        st.warning("Voice unavailable.")
    elif 'recognizer' not in st.session_state:
        st.warning("Voice model not loaded yet, please wait...")
    else:
        recognizer = st.session_state.recognizer
        voice = Voice(shelf, "models/vosk-model-small-en-us-0.15")

        def audio_frame_callback(frame: av.AudioFrame):
            raw = bytes(frame.to_ndarray().tobytes())
            if recognizer.AcceptWaveform(raw):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    st.session_state.command_queue.append(text) 
            return frame

        while st.session_state.command_queue:
            text = st.session_state.command_queue.pop(0)
            st.session_state.last_command = text
            voice.handle_command(text)

        webrtc_streamer(
            key="voice",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                ]
            },
            media_stream_constraints={"audio": True, "video": False},
            audio_frame_callback=audio_frame_callback,
            async_processing=True
        )

        st.info(f"Last heard: **{st.session_state.get('last_command', '—')}**")
        st.caption("`bring <item>` · `stop <item/all>` · `retract <item/empty>`")

#Logs
with col_log:
    st.subheader("Event Log")
    try:
        df = pd.read_csv('data/logs.csv')
        st.dataframe(df.tail(50).iloc[::-1], use_container_width=True, height=500)
    except FileNotFoundError:
        st.info("No logs yet.")

# Contents Editor
st.divider()
st.subheader("Compartment Contents")

content_cols = st.columns(len(shelf.total_com), gap="small")

for i, com in enumerate(shelf.total_com):
    with content_cols[i]:
        is_moving = com.state in (STATE_MOVING_UP, STATE_MOVING_DOWN)

        current_contents = ", ".join(com.contents) if com.contents else ""

        new_val = st.text_input(
            label=f"Compartment {com.com_no}",
            value=current_contents,
            key=f"contents_input_{com.com_no}",
            disabled=is_moving,
            placeholder="e.g. towel, bottle",
        )

        if not is_moving:
            parsed = [item.strip().lower() for item in new_val.split(",") if item.strip()]
            if parsed != com.contents:
                com.contents = parsed
                logger.log(com, f'CONTENTS_UPDATED: {parsed}')

time.sleep(0.1)
st.rerun()
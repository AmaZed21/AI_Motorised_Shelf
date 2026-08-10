import os
import time
import threading
import streamlit as st
import streamlit.components.v1 as components
from simulator import Logger, SensorDataLogger, Compartment, Shelf, ShelfSafetyMonitor, STATE_STOPPED, STATE_MOVING_UP, STATE_MOVING_DOWN, LABEL_MANUAL_STOP, MODEL_PATH
import pandas as pd

st.set_page_config(page_title="Shelf Control", layout="wide")
st.title("Motorised Shelf Dashboard")

#Logger for ML MODEL
SAMPLE_INTERVAL_SECONDS = 0.1
TEMP_SENSOR_CSV = "data/training_data.csv"

def collect_sensor_data(
    shelf,
    sensor_logger,
    event_logger,
    safety_monitor,
    stop_event,
):
    while not stop_event.is_set():
        start_time = time.perf_counter()

        # Updates movement and current sensor values.
        shelf.update_all(SAMPLE_INTERVAL_SECONDS)

        for compartment in shelf.total_com:
            # Random Forest reads current sensor values.
            detected_fault = safety_monitor.check_compartment(compartment)

            # Keep the existing Logger system.
            if detected_fault is not None:
                event_logger.log(
                    compartment,
                    f"ML_SAFETY_STOP: {detected_fault.upper()}",
                )

            # Keep logging samples for future ML training.
            sensor_logger.log_sample(compartment)

        elapsed = time.perf_counter() - start_time
        stop_event.wait(max(0, SAMPLE_INTERVAL_SECONDS - elapsed))

if "shelf" not in st.session_state:
    os.makedirs("data", exist_ok=True)

    com_1 = Compartment(1, weight=0.5)
    com_2 = Compartment(2, weight=0.4)
    com_3 = Compartment(3)

    st.session_state.shelf = Shelf([com_1, com_2, com_3])

    # Existing dashboard/event CSV: unchanged
    st.session_state.logger = Logger("data/logs.csv")

    st.session_state.safety_monitor = ShelfSafetyMonitor(
        model_path=MODEL_PATH,
        confidence_threshold=0.60,
        required_consecutive_predictions=1,
    )

    st.session_state.sensor_logger = SensorDataLogger(TEMP_SENSOR_CSV)

    st.session_state.sensor_stop_event = threading.Event()

    st.session_state.sensor_thread = threading.Thread(
        target=collect_sensor_data,
        args=(
            st.session_state.shelf,
            st.session_state.logger,
            st.session_state.sensor_logger,
            st.session_state.safety_monitor,
            st.session_state.sensor_stop_event,
        ),
        daemon=True,
    )
    st.session_state.sensor_thread.start()

shelf  = st.session_state.shelf
logger = st.session_state.logger
safety_monitor = st.session_state.safety_monitor

def handle_cmd():
    raw = st.session_state.get("text_cmd", "").strip().lower()
    if not raw:
        return

    executed = False

    if raw == "reset":
        shelf.reset()
        for c in shelf.total_com:
            logger.log(c, 'COMMAND_RESET')
        executed = True

    elif raw.startswith("bring "):
        item = raw[len("bring "):].strip()
        com = shelf.find_item(item)
        if com:
            com.move_down()
            logger.log(com, f'COMMAND_BRING: {item}')
            executed = True

    elif raw.startswith("put back "):
        item = raw[len("put back "):].strip()
        com = shelf.find_item(item)
        if com:
            com.move_up()
            logger.log(com, f'COMMAND_PUT_BACK: {item}')
            executed = True

    elif raw.startswith("stop "):
        item = raw[len("stop "):].strip()
        com = shelf.find_item(item)
        if com:
            com.stop()
            logger.log(com, f'COMMAND_STOP: {item}')
            executed = True

    if executed:
        st.session_state["text_cmd"] = ""   # clears the box

#Selection of cabinet
for c in shelf.total_com:
    if f"com_selected_{c.com_no}" not in st.session_state:
        st.session_state[f"com_selected_{c.com_no}"] = False

selected_coms = [c for c in shelf.total_com if st.session_state.get(f"com_selected_{c.com_no}", False)]

#Graphics
col_vis, col_ctrl, col_log = st.columns([4, 2, 6])

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
                "FAULT": "#e5ff00",
            }.get(com.state, "#888888")

            border = "#ffffff" if st.session_state.get(f"com_selected_{com.com_no}", False) else "#444444"

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

            is_selected = st.session_state.get(f"com_selected_{com.com_no}", False)
            dot_color = "🟢" if is_selected else "🔴"
            if st.button(f"{dot_color} {com.com_no}", key=f"btn_select_{com.com_no}", use_container_width=True):
                st.session_state[f"com_selected_{com.com_no}"] = not is_selected
                st.rerun()

            st.markdown(
                f"<div style='text-align:center; margin-top:-6px;'>"
                f"<div style='font-weight:600;'>Compartment {com.com_no}: {', '.join(com.contents) if com.contents else 'Empty'}</div>"
                f"<div style='color:{color}; font-size:13px;'>{com.state}</div>"
                f"<div style='font-size:13px;'>{com.position:.1f} cm</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    st.divider()
    st.text_input(
        label="",
        placeholder='Bring, Put back, Reset, Stop',
        key="text_cmd",
        label_visibility="collapsed",
        on_change= handle_cmd 
    )


#Controls
with col_ctrl:
    st.subheader("Controls")

    if not selected_coms:
        st.warning("Select at least one compartment.")
    else:
        for com in selected_coms:
            st.markdown(
                f"### Compartment {com.com_no}\n"
                f"{com.state} | {com.position:.0f} cm | "
                f"{com.speed:.2f} cm/s"
            )

            # Add this directly below the current status line
            st.caption(
                f"Current: {com.motor_current:.2f} A | "
                f"Speed: {com.speed:.2f} cm/s | "
                f"Label: {com.label}"
            )

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
            com.stop(label=LABEL_MANUAL_STOP)
            logger.log(com, "COMMAND_STOP")

    st.divider()
    if st.button("Create Obstruction", use_container_width=True):
        for com in selected_coms:
            if com.state not in (STATE_MOVING_UP, STATE_MOVING_DOWN):
                st.warning(
                    f"Compartment {com.com_no} must be moving before "
                    "an obstruction can be created."
                )
                continue
            com.sensor_distance = 0.5

            logger.log(com, "SCENARIO_CREATED: OBSTRUCTION")
        
    if st.button("Create Overload", use_container_width=True):
        for com in selected_coms:
            if com.state not in (STATE_MOVING_UP, STATE_MOVING_DOWN):
                st.warning(
                    f"Compartment {com.com_no} must be moving before "
                    "an overload can be created."
                )
                continue
            com.weight = com.MAX_WEIGHT + 1.0

            logger.log(com, "SCENARIO_CREATED: OVERLOAD")

    if st.button("Clear Fault / Recover", use_container_width=True):
        for com in selected_coms:
            com.clear_fault()
            logger.log(com, "SCENARIO_CLEARED")

        if st.button("Reset System"):
            shelf.reset()
            for c in shelf.total_com:
                logger.log(c, 'RESET')

#Logs
with col_log:
    st.subheader("Event Log")
    try:
        df = pd.read_csv('data/logs.csv', index_col = 'timestamp')
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
            placeholder="items (separated using commas)",
        )

        if not is_moving:
            parsed = [item.strip().lower() for item in new_val.split(",") if item.strip()]
            if parsed != com.contents:
                com.contents = parsed
                logger.log(com, f'CONTENTS_UPDATED: {parsed}')

time.sleep(0.1)
st.rerun()
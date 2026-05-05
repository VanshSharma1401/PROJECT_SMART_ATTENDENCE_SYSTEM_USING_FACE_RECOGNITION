"""Streamlit dashboard for the Smart Attendance System."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.attendance import AttendanceLogger
from src.config import ensure_project_dirs, settings
from src.encode import FaceEncodingStore
from src.utils import (
    bgr_to_rgb,
    draw_face_box,
    list_image_files,
    list_people,
    open_camera,
    overlay_timestamp,
    sanitize_person_name,
)


ensure_project_dirs()


st.set_page_config(
    page_title="Smart Attendance",
    page_icon="SA",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1280px; }
    [data-testid="stSidebar"] { background: #101820; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    h1, h2, h3 { letter-spacing: 0; }
    .metric-row [data-testid="stMetric"] {
        background: #f7f9fb;
        border: 1px solid #dfe7ef;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.2rem 0.58rem;
        font-size: 0.82rem;
        font-weight: 700;
        background: #e8f3ff;
        color: #075985;
        border: 1px solid #bfdbfe;
    }
    .danger-pill {
        background: #fff1f2;
        color: #be123c;
        border-color: #fecdd3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_logs(storage: str, limit: int | None = 200) -> pd.DataFrame:
    logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
    rows = logger.read_logs(limit=limit)
    return pd.DataFrame(rows)


def show_runtime_error(title: str, exc: Exception) -> None:
    st.error(title)
    st.code(str(exc))


def import_cv2():
    import cv2

    return cv2


def import_face_registrar():
    from src.register import FaceRegistrar

    return FaceRegistrar


def import_face_recognizer():
    from src.recognize import FaceRecognizer

    return FaceRecognizer


def dashboard_page() -> None:
    st.title("Smart Attendance System")

    people = list_people()
    image_count = len(list_image_files(settings.dataset_dir))
    names, encodings, _ = FaceEncodingStore().load()
    logs_df = load_logs("csv", limit=None)

    today_count = 0
    if not logs_df.empty and "date" in logs_df.columns:
        today_count = int((logs_df["date"] == time.strftime("%Y-%m-%d")).sum())

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered People", len(people))
    col2.metric("Dataset Images", image_count)
    col3.metric("Cached Encodings", len(encodings))
    col4.metric("Today Marks", today_count)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Recent Attendance")
        if logs_df.empty:
            st.info("No attendance logs yet.")
        else:
            st.dataframe(logs_df.head(10), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Encoding Cache")
        store = FaceEncodingStore()
        cache_current = store.cache_is_current()
        pill_class = "status-pill" if cache_current else "status-pill danger-pill"
        label = "Current" if cache_current else "Needs rebuild"
        st.markdown(f'<span class="{pill_class}">{label}</span>', unsafe_allow_html=True)
        st.write(f"Cache file: `{settings.encodings_file.name}`")
        if st.button("Rebuild Encodings", use_container_width=True):
            try:
                report = store.build(force=True)
            except Exception as exc:  # noqa: BLE001
                show_runtime_error("Encoding rebuild failed.", exc)
            else:
                st.success(
                    f"Encoded {report.encoded_faces} faces from {report.total_images} images."
                )


def register_page() -> None:
    st.title("Register User")

    col1, col2, col3 = st.columns([1.4, 0.8, 0.8])
    person_name = col1.text_input("Name", placeholder="Example: Priya Sharma")
    samples = col2.slider(
        "Samples",
        settings.minimum_samples_per_user,
        settings.maximum_samples_per_user,
        settings.default_samples_per_user,
    )
    camera_index = col3.number_input("Camera", min_value=0, max_value=5, value=settings.camera_index)

    frame_slot = st.empty()
    status_slot = st.empty()
    progress_slot = st.empty()

    if st.button("Capture Samples", type="primary", use_container_width=True):
        try:
            safe_name = sanitize_person_name(person_name)
        except ValueError as exc:
            st.error(str(exc))
            return

        try:
            cv2 = import_cv2()
            FaceRegistrar = import_face_registrar()
            registrar = FaceRegistrar()
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Registration tools could not be loaded.", exc)
            return

        existing_count = registrar.count_samples(safe_name)
        target_count = existing_count + samples
        progress = progress_slot.progress(0.0)

        with st.spinner("Initializing camera..."):
            try:
                camera = open_camera(int(camera_index))
            except RuntimeError as exc:
                st.error(str(exc))
                return

        last_capture = 0.0
        start_time = time.time()
        last_message = "Camera ready"

        current_count = existing_count
        try:
            while current_count < target_count:
                ok, frame = camera.read()
                if not ok:
                    st.error("Camera frame could not be read.")
                    break

                now = time.time()
                result = None
                if now - last_capture >= settings.registration_capture_interval_sec:
                    result = registrar.validate_and_save_frame(frame, safe_name)
                    last_capture = now
                    last_message = result.message
                    if result and result.saved:
                        current_count += 1

                annotated = frame.copy()
                if result and result.face_location:
                    color = (40, 180, 80) if result.saved else (0, 180, 255)
                    draw_face_box(
                        annotated,
                        result.face_location,
                        result.status.replace("_", " ").title(),
                        color,
                    )

                cv2.putText(
                    annotated,
                    f"{safe_name}: {current_count}/{target_count}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    registrar.next_prompt(current_count),
                    (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frame_slot.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
                progress.progress(min(1.0, (current_count - existing_count) / samples))
                status_slot.info(last_message)

                if time.time() - start_time > settings.registration_timeout_sec:
                    st.warning("Registration timed out.")
                    break
                time.sleep(0.01)
        finally:
            camera.release()

        final_count = registrar.count_samples(safe_name)
        st.success(f"{safe_name} now has {final_count} dataset images.")
        st.warning("Rebuild encodings before starting attendance.")


def attendance_page() -> None:
    st.title("Live Attendance")

    col1, col2, col3, col4 = st.columns(4)
    camera_index = col1.number_input("Camera", min_value=0, max_value=5, value=settings.camera_index)
    storage = col2.selectbox("Storage", ["csv", "sqlite"], index=0)
    duration = col3.slider("Session Seconds", 15, 600, 120, step=15)
    threshold = col4.slider("Threshold", 0.45, 0.60, settings.recognition_threshold, step=0.01)

    require_liveness = st.checkbox("Require Blink", value=False)
    mark_probable = st.checkbox("Mark Probable Matches", value=False)

    frame_slot = st.empty()
    result_slot = st.empty()
    metric_slot = st.empty()

    if st.button("Start Attendance Session", type="primary", use_container_width=True):
        try:
            FaceRecognizer = import_face_recognizer()
            attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
            recognizer = FaceRecognizer(
                attendance_logger=attendance_logger,
                recognition_threshold=threshold,
                require_liveness=require_liveness,
                mark_probable=mark_probable,
            )
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Recognition tools could not be loaded.", exc)
            return

        if len(recognizer.label_to_name) == 0:
            st.error("No known face encodings found. Register users and rebuild encodings first.")
            return

        try:
            cv2 = import_cv2()
            camera = open_camera(int(camera_index))
        except RuntimeError as exc:
            st.error(str(exc))
            return

        start = time.time()
        frame_count = 0
        processed_count = 0
        last_results = []

        try:
            while time.time() - start < duration:
                ok, frame = camera.read()
                if not ok:
                    st.error("Camera frame could not be read.")
                    break

                frame_count += 1
                if frame_count % settings.process_every_n_frames == 0:
                    annotated, last_results = recognizer.recognize_frame(
                        frame,
                        mark_attendance=True,
                        annotate=True,
                    )
                    processed_count += 1
                else:
                    annotated = frame.copy()
                    overlay_timestamp(annotated)

                frame_slot.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

                elapsed = max(time.time() - start, 1e-6)
                fps = frame_count / elapsed
                metric_slot.metric("Live FPS", f"{fps:.1f}", f"processed {processed_count}")

                if last_results:
                    result_slot.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "name": item.name,
                                    "confidence": item.confidence,
                                    "class": item.classification,
                                    "distance": item.distance,
                                    "attendance": item.attendance_event.reason
                                    if item.attendance_event
                                    else "",
                                }
                                for item in last_results
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                time.sleep(0.01)
        finally:
            camera.release()

        st.success("Attendance session ended.")


def logs_page() -> None:
    st.title("Attendance Logs")

    col1, col2 = st.columns([0.35, 0.65])
    storage = col1.selectbox("Storage", ["csv", "sqlite"], index=0, key="logs_storage")
    limit = col1.number_input("Rows", min_value=10, max_value=5000, value=200, step=10)

    attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
    rows = attendance_logger.read_logs(limit=int(limit))
    logs_df = pd.DataFrame(rows)

    if logs_df.empty:
        col2.info("No logs found.")
    else:
        col2.dataframe(logs_df, use_container_width=True, hide_index=True)

    export_path = attendance_logger.export_csv(settings.logs_dir / "attendance_export.csv")
    st.download_button(
        "Export CSV",
        data=export_path.read_bytes(),
        file_name="attendance_export.csv",
        mime="text/csv",
        use_container_width=True,
    )


def dataset_page() -> None:
    st.title("Dataset Health")

    store = FaceEncodingStore()
    deep = st.checkbox("Deep Validation", value=False)

    if st.button("Run Validation", use_container_width=True):
        try:
            health = store.validate_dataset(deep=deep)
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Dataset validation failed.", exc)
        else:
            st.json(health)

    if st.button("Rebuild Encodings Now", type="primary", use_container_width=True):
        try:
            report = store.build(force=True)
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Encoding rebuild failed.", exc)
        else:
            st.success(
                f"Encoded {report.encoded_faces}/{report.total_images} images for {report.people} people."
            )
            if report.errors:
                st.warning(f"{len(report.errors)} images were skipped.")
                st.write(report.errors[:25])

    people = list_people()
    table = [
        {
            "person": person,
            "images": len(list_image_files(settings.dataset_dir / person)),
        }
        for person in people
    ]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


def setup_page() -> None:
    st.title("Setup")
    st.code("python -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt", language="bash")
    st.code("streamlit run app.py", language="bash")
    st.code("python -m src.register \"Student Name\"\npython -m src.encode --force\npython -m src.recognize", language="bash")


def main() -> None:
    st.sidebar.title("Smart Attendance")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Register User", "Live Attendance", "Logs", "Dataset Health", "Setup"],
    )

    if page == "Dashboard":
        dashboard_page()
    elif page == "Register User":
        register_page()
    elif page == "Live Attendance":
        attendance_page()
    elif page == "Logs":
        logs_page()
    elif page == "Dataset Health":
        dataset_page()
    else:
        setup_page()


if __name__ == "__main__":
    main()

from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import json
import threading
import time
import socket
import sys
from flask import Flask, Response, render_template_string
from hailo_apps.hailo_app_python.core.gstreamer import gstreamer_app

# Access Whisplay driver
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPLAY_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "Whisplay"))
DRIVER_DIR = os.path.join(WHISPLAY_DIR, "Driver")
if DRIVER_DIR not in sys.path:
    sys.path.append(DRIVER_DIR)
try:
    from WhisPlay import WhisPlayBoard
except Exception:
    WhisPlayBoard = None

# Whisplay constants
WHISPLAY_WIDTH = 240
WHISPLAY_HEIGHT = 280

# Global frame buffer for web streaming
latest_frame = None
frame_lock = threading.Lock()
WEB_PORT = 8081  # Use a different port than detection if running both

def image_to_rgb565(frame: np.ndarray) -> list:
    """Convert OpenCV BGR frame to RGB565 byte list for Whisplay display."""
    resized = cv2.resize(frame, (WHISPLAY_WIDTH, WHISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    high_byte = (rgb565 >> 8).astype(np.uint8)
    low_byte = (rgb565 & 0xFF).astype(np.uint8)
    pixel_data = np.stack((high_byte, low_byte), axis=2).flatten().tolist()
    return pixel_data

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hailo RPi5 Pose Estimation Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #1a1a1a; color: #ffffff; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #2196F3; }
        .video-container { text-align: center; background-color: #000; padding: 10px; border-radius: 10px; margin: 20px 0; }
        img { max-width: 100%; height: auto; border: 2px solid #2196F3; border-radius: 5px; }
        .info { background-color: #2a2a2a; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 3px; margin: 5px; background-color: #2196F3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧍 Hailo RPi5 Pose Estimation Preview</h1>
        <div class="info">
            <strong>Status:</strong> <span class="status">LIVE</span>
        </div>
        <div class="video-container">
            <img src="/video_feed" alt="Live Pose Estimation Stream">
        </div>
    </div>
</body>
</html>
"""

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

def start_web_server():
    app = Flask(__name__)
    @app.route('/')
    def index(): return render_template_string(HTML_TEMPLATE)
    @app.route('/video_feed')
    def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    local_ip = get_local_ip()
    print(f"\n🌐 Pose Web server starting at http://{local_ip}:{WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, debug=False)

def patched_picamera_thread(pipeline, video_width, video_height, video_format, picamera_config=None):
    from picamera2 import Picamera2
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    with Picamera2() as picam2:
        if picamera_config is None:
            main = {'size': (video_width, video_height), 'format': 'RGB888'}
            lores = {'size': (video_width, video_height), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
        else:
            config = picamera_config
        picam2.configure(config)
        lores_stream = config['lores']
        format_str = 'RGB' if lores_stream['format'] == 'RGB888' else video_format
        width, height = lores_stream['size']
        appsrc.set_property("caps", Gst.Caps.from_string(f"video/x-raw, format={format_str}, width={width}, height={height}, framerate=30/1, pixel-aspect-ratio=1/1"))
        picam2.start()
        frame_count = 0
        while True:
            frame_data = picam2.capture_array('lores')
            if frame_data is None: break
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buffer.pts = frame_count * buffer_duration
            buffer.duration = buffer_duration
            ret = appsrc.emit('push-buffer', buffer)
            if ret != Gst.FlowReturn.OK: break
            frame_count += 1

# Apply the patch
gstreamer_app.picamera_thread = patched_picamera_thread

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

class GStreamerPoseEstimationAppNoDisplay(GStreamerPoseEstimationApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            from hailo_apps.hailo_app_python.core.common.core import get_default_parser
            parser = get_default_parser()
        
        temp_args, _ = parser.parse_known_args()
        cli_input = temp_args.input
        input_source = cli_input if cli_input else (user_data.file_path if user_data.file_path != "NOT_SPECIFIED" else "rpi")
        if input_source is None or str(input_source).lower() == "null": input_source = "rpi"
        
        parser.set_defaults(input=input_source)
        self.detected_width, self.detected_height = 1280, 720
        
        if input_source == "rpi":
            if user_data.input_resolution_width and user_data.input_resolution_height:
                self.detected_width, self.detected_height = user_data.input_resolution_width, user_data.input_resolution_height
            else:
                self.detected_width, self.detected_height = 2028, 1520
        elif os.path.exists(str(input_source)):
            cap = cv2.VideoCapture(input_source)
            if cap.isOpened():
                self.detected_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.detected_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        
        super().__init__(app_callback, user_data, parser)

    def get_pipeline_string(self):
        from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
            SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, 
            TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, QUEUE
        )
        self.video_width, self.video_height = self.detected_width, self.detected_height
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source, video_width=self.video_width, video_height=self.video_height, frame_rate=self.frame_rate, sync=self.sync)
        inference_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size
        )
        inference_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(inference_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)
        filter_pipeline = f"{QUEUE(name='filter_q')} ! identity name=identity_filter"
        overlay_pipeline = f"{QUEUE(name='hailo_overlay_q')} ! hailooverlay"
        
        needs_frame = self.user_data.gui_preview or self.user_data.web_preview or self.user_data.display_preview or self.user_data.show_boxes_only
        if needs_frame:
            preview_width = (self.detected_width // 2) * 2
            preview_height = (self.detected_height // 2) * 2
            
            hardware_post_proc = (
                f"! {QUEUE(name='preview_scale_q')} ! videoscale ! "
                f"video/x-raw, width={preview_width}, height={preview_height} ! "
                f"{QUEUE(name='preview_convert_q')} ! videoconvert ! "
                f"video/x-raw, format=RGB"
            )
        else:
            hardware_post_proc = ""
        
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        sink_pipeline = f"fakesink name=hailo_display sync={self.sync}"

        return f'{source_pipeline} ! {inference_pipeline_wrapper} ! {tracker_pipeline} ! {filter_pipeline} ! {overlay_pipeline} {hardware_post_proc} ! {user_callback_pipeline} ! {sink_pipeline}'

def extract_boxes_grid(frame, detections, width, height, max_boxes=12, box_size=320):
    """
    Extract bounding boxes from frame and arrange them in a grid layout.
    Shows only the cropped content of each detection box.
    """
    if not detections:
        placeholder = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No detections", (50, box_size // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    detections = detections[:max_boxes]
    num_boxes = len(detections)
    cols = int(np.ceil(np.sqrt(num_boxes)))
    rows = int(np.ceil(num_boxes / cols))
    grid_h, grid_w = rows * box_size, cols * box_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for idx, detection in enumerate(detections):
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        track_id = None
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1: track_id = track[0].get_id()
        
        x_min, y_min = int(bbox.xmin() * width), int(bbox.ymin() * height)
        w, h = int(bbox.width() * width), int(bbox.height() * height)
        x_max, y_max = max(0, min(x_min + w, width)), max(0, min(y_min + h, height))
        x_min, y_min = max(0, min(x_min, width)), max(0, min(y_min, height))
        
        if x_max <= x_min or y_max <= y_min: continue
        box_region = frame[y_min:y_max, x_min:x_max].copy()
        box_h, box_w = box_region.shape[:2]
        if box_h == 0 or box_w == 0: continue
        
        scale = min(box_size / box_w, box_size / box_h)
        new_w, new_h = int(box_w * scale), int(box_h * scale)
        if new_w > 0 and new_h > 0:
            resized_box = cv2.resize(box_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            col, row = idx % cols, idx // cols
            y_offset, x_offset = (box_size - new_h) // 2, (box_size - new_w) // 2
            grid_y, grid_x = row * box_size + y_offset, col * box_size + x_offset
            grid[grid_y:grid_y + new_h, grid_x:grid_x + new_w] = resized_box
            color = colors[idx % len(colors)]
            cv2.rectangle(grid, (col * box_size, row * box_size), ((col + 1) * box_size - 1, (row + 1) * box_size - 1), color, 2)
            label_text = f"ID:{track_id} {label}:{confidence:.2f}" if track_id else f"{label}:{confidence:.2f}"
            cv2.putText(grid, label_text, (col * box_size + 5, row * box_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return grid

def get_keypoints():
    return {'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.show_boxes_only = False
        self.use_frame = True
        self.allowed_classes_set = None
        self.confidence_threshold = 0.5
        self.frame_skip = 1
        self.file_path = "NOT_SPECIFIED"
        self.input_resolution_width = None
        self.input_resolution_height = None
        self.web_preview, self.gui_preview, self.display_preview = False, True, False
        self.board = None
        self.load_config()
        self.setup_previews()

    def load_config(self):
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.show_boxes_only = config.get("show_boxes_only", False)
                    self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
                    self.frame_skip = int(config.get("frame_skip", 1))
                    if "file_path" in config: self.file_path = config["file_path"]
                    input_res = config.get("input_resolution", {})
                    if input_res:
                        self.input_resolution_width = int(input_res.get("width", 0)) if input_res.get("width") else None
                        self.input_resolution_height = int(input_res.get("height", 0)) if input_res.get("height") else None
                    self.web_preview = config.get("web_preview", self.web_preview)
                    self.gui_preview = config.get("gui_preview", self.gui_preview)
                    self.display_preview = config.get("display_preview", self.display_preview)
                    detection_classes = config.get("detection_classes", "all")
                    if isinstance(detection_classes, str) and detection_classes.strip().lower() != "all":
                        self.allowed_classes_set = {c.strip().lower() for c in detection_classes.split(",") if c.strip()}
                    print(f"✓ Pose Config loaded: confidence={self.confidence_threshold}, res={self.input_resolution_width}x{self.input_resolution_height}")
            except Exception as e: print(f"⚠️  Config error: {e}")

    def setup_previews(self):
        if self.web_preview: threading.Thread(target=start_web_server, daemon=True).start()
        if self.display_preview and WhisPlayBoard:
            try:
                self.board = WhisPlayBoard()
                self.board.set_backlight(80)
            except: self.display_preview = False

def filter_callback(pad, info, user_data):
    """Filter detections in hardware ROI before they reach the overlay."""
    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK
    
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    for det in detections:
        label = det.get_label().lower()
        confidence = det.get_confidence()
        
        is_allowed = True
        # Check class filter
        if user_data.allowed_classes_set is not None and label not in user_data.allowed_classes_set:
            is_allowed = False
        # Check confidence filter
        if confidence < user_data.confidence_threshold:
            is_allowed = False
            
        if not is_allowed:
            roi.remove_object(det)
            
    return Gst.PadProbeReturn.OK

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK
    user_data.increment()
    if user_data.get_count() % user_data.frame_skip != 0: return Gst.PadProbeReturn.OK
    
    format, width, height = get_caps_from_pad(pad)
    needs_frame = user_data.gui_preview or user_data.web_preview or user_data.display_preview or user_data.show_boxes_only
    frame = get_numpy_from_buffer(buffer, format, width, height) if needs_frame else None

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints_map = get_keypoints()
    string_to_print = f"Frame: {user_data.get_count()}\n"

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        
        # Get tracking ID
        track_id = None
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            
        if label == "person":
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if track_id is not None:
                string_to_print += f"Pose: ID {track_id} Conf: {confidence:.2f}\n"
            else:
                string_to_print += f"Pose: Conf: {confidence:.2f}\n"
                
            if frame is not None and user_data.show_boxes_only and len(landmarks) != 0:
                # In grid mode, we draw keypoints manually because hardware overlay 
                # can't see the cropped frame
                points = landmarks[0].get_points()
                bbox = detection.get_bbox()
                for point in points:
                    # Normalized to pixel coordinates (relative to the full frame)
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    # We draw on the full frame BEFORE it gets passed to extract_boxes_grid
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Handle "boxes only" preview mode (Grid Mode)
        if user_data.show_boxes_only:
            frame = extract_boxes_grid(frame, detections, width, height)
            
        if user_data.web_preview:
            global latest_frame
            with frame_lock: latest_frame = frame.copy()
        if user_data.display_preview and user_data.board:
            user_data.board.draw_image(0, 0, WHISPLAY_WIDTH, WHISPLAY_HEIGHT, image_to_rgb565(frame))
        if user_data.gui_preview: user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationAppNoDisplay(app_callback, user_data)
    app.options_menu.use_frame = True
    user_data.use_frame = True
    
    # Patch display
    original_display = gstreamer_app.display_user_data_frame
    def patched_display(ud):
        if ud.gui_preview: original_display(ud)
        else:
            while ud.running: time.sleep(1)
    gstreamer_app.display_user_data_frame = patched_display

    # Connect filter
    filter_identity = app.pipeline.get_by_name("identity_filter")
    if filter_identity:
        filter_identity.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, filter_callback, user_data)
    
    app.run()


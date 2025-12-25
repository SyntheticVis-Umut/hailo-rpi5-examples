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
WEB_PORT = 8083  # Unique port for depth

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
    <title>Hailo RPi5 Depth Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #1a1a1a; color: #ffffff; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #9C27B0; }
        .video-container { text-align: center; background-color: #000; padding: 10px; border-radius: 10px; margin: 20px 0; }
        img { max-width: 100%; height: auto; border: 2px solid #9C27B0; border-radius: 5px; }
        .info { background-color: #2a2a2a; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 3px; margin: 5px; background-color: #9C27B0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⛰️ Hailo RPi5 Depth Preview</h1>
        <div class="info">
            <strong>Status:</strong> <span class="status">LIVE (Depth Map)</span>
        </div>
        <div class="video-container">
            <img src="/video_feed" alt="Live Depth Stream">
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
    print(f"\n🌐 Depth web server starting at http://{local_ip}:{WEB_PORT}")
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

gstreamer_app.picamera_thread = patched_picamera_thread

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.depth.depth_pipeline import GStreamerDepthApp

class GStreamerDepthAppNoDisplay(GStreamerDepthApp):
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
            USER_CALLBACK_PIPELINE, QUEUE
        )
        self.video_width, self.video_height = self.detected_width, self.detected_height
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source, video_width=self.video_width, video_height=self.video_height, frame_rate=self.frame_rate, sync=self.sync)
        
        # Depth inference pipeline
        inference_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size
        )
        inference_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(inference_pipeline)
        
        needs_frame = self.user_data.gui_preview or self.user_data.web_preview or self.user_data.display_preview
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

        return f'{source_pipeline} ! {inference_pipeline_wrapper} ! identity name=depth_identity {hardware_post_proc} ! {user_callback_pipeline} ! {sink_pipeline}'

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
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
                    self.frame_skip = int(config.get("frame_skip", 1))
                    if "file_path" in config: self.file_path = config["file_path"]
                    input_res = config.get("input_resolution", {})
                    if input_res:
                        self.input_resolution_width = int(input_res.get("width", 0)) if input_res.get("width") else None
                        self.input_resolution_height = int(input_res.get("height", 0)) if input_res.get("height") else None
                    self.web_preview = config.get("web_preview", self.web_preview)
                    self.gui_preview = config.get("gui_preview", self.gui_preview)
                    self.display_preview = config.get("display_preview", self.display_preview)
                    print(f"✓ Depth Config loaded: res={self.input_resolution_width}x{self.input_resolution_height}, previews=[web={self.web_preview}, gui={self.gui_preview}, display={self.display_preview}]")
            except Exception as e: print(f"⚠️  Config error: {e}")

    def setup_previews(self):
        if self.web_preview: threading.Thread(target=start_web_server, daemon=True).start()
        if self.display_preview and WhisPlayBoard:
            try:
                self.board = WhisPlayBoard()
                self.board.set_backlight(80)
            except: self.display_preview = False

    def calculate_average_depth(self, depth_mat):
        depth_values = np.array(depth_mat).flatten()
        try:
            m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]
        except: m_depth_values = np.array([])
        return np.mean(m_depth_values) if len(m_depth_values) > 0 else 0

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK
    user_data.increment()
    if user_data.get_count() % user_data.frame_skip != 0: return Gst.PadProbeReturn.OK
    
    format, width, height = get_caps_from_pad(pad)
    
    # Get depth metadata
    roi = hailo.get_roi_from_buffer(buffer)
    depth_masks = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    
    avg_depth = 0
    depth_frame = None
    if len(depth_masks) > 0:
        depth_mask = depth_masks[0]
        depth_data = np.array(depth_mask.get_data())
        avg_depth = user_data.calculate_average_depth(depth_data)
        
        # Colorize depth for preview if needed
        needs_preview = user_data.gui_preview or user_data.web_preview or user_data.display_preview
        if needs_preview:
            # Reshape based on model output (typically small)
            h, w = depth_mask.get_height(), depth_mask.get_width()
            depth_data = depth_data.reshape((h, w))
            # Normalize to 0-255
            depth_min, depth_max = depth_data.min(), depth_data.max()
            if depth_max > depth_min:
                depth_norm = (255 * (depth_data - depth_min) / (depth_max - depth_min)).astype(np.uint8)
            else:
                depth_norm = np.zeros((h, w), dtype=np.uint8)
            # Color map (Magma or Jet is good for depth)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            # Resize to preview size
            depth_frame = cv2.resize(depth_color, (width, height), interpolation=cv2.INTER_LINEAR)

    if depth_frame is not None:
        # Depth frame is already BGR from ColorMap
        if user_data.web_preview:
            global latest_frame
            with frame_lock: latest_frame = depth_frame.copy()
        if user_data.display_preview and user_data.board:
            user_data.board.draw_image(0, 0, WHISPLAY_WIDTH, WHISPLAY_HEIGHT, image_to_rgb565(depth_frame))
        if user_data.gui_preview: user_data.set_frame(depth_frame)

    print(f"Frame: {user_data.get_count()} | Average Depth: {avg_depth:.2f}")
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    user_data = user_app_callback_class()
    app = GStreamerDepthAppNoDisplay(app_callback, user_data)
    app.options_menu.use_frame = True
    user_data.use_frame = True
    
    # Patch display
    original_display = gstreamer_app.display_user_data_frame
    def patched_display(ud):
        if ud.gui_preview: original_display(ud)
        else:
            while ud.running: time.sleep(1)
    gstreamer_app.display_user_data_frame = patched_display
    
    app.run()
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
WEB_PORT = 8082  # Use a different port than detection/pose if running both

def image_to_rgb565(frame: np.ndarray) -> list:
    """Convert OpenCV BGR frame to RGB565 byte list for Whisplay display."""
    resized = cv2.resize(frame, (WHISPLAY_WIDTH, WHISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # OPTIMIZATION: Use vectorized operations more efficiently
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    # OPTIMIZATION: Use more efficient array operations
    high_byte = (rgb565 >> 8).astype(np.uint8)
    low_byte = (rgb565 & 0xFF).astype(np.uint8)
    # OPTIMIZATION: Use ravel instead of flatten for better performance
    pixel_data = np.stack((high_byte, low_byte), axis=2).ravel().tolist()
    return pixel_data

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hailo RPi5 Instance Segmentation Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #1a1a1a; color: #ffffff; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #FF6B6B; }
        .video-container { text-align: center; background-color: #000; padding: 10px; border-radius: 10px; margin: 20px 0; }
        img { max-width: 100%; height: auto; border: 2px solid #FF6B6B; border-radius: 5px; }
        .info { background-color: #2a2a2a; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 3px; margin: 5px; background-color: #FF6B6B; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Hailo RPi5 Instance Segmentation Preview</h1>
        <div class="info">
            <strong>Status:</strong> <span class="status">LIVE</span>
        </div>
        <div class="video-container">
            <img src="/video_feed" alt="Live Segmentation Stream">
        </div>
    </div>
</body>
</html>
"""

def get_local_ip():
    """Get the local IP address of the device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def generate_frames():
    """Generate MJPEG frames for video streaming."""
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

def start_web_server():
    """Start Flask web server."""
    app = Flask(__name__)
    @app.route('/')
    def index(): return render_template_string(HTML_TEMPLATE)
    @app.route('/video_feed')
    def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    local_ip = get_local_ip()
    print(f"\n🌐 Segmentation web server starting at http://{local_ip}:{WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, debug=False)

# -----------------------------------------------------------------------------------------------
# Monkey-patch picamera_thread to fix high resolution issues
# -----------------------------------------------------------------------------------------------
def patched_picamera_thread(pipeline, video_width, video_height, video_format, picamera_config=None):
    from picamera2 import Picamera2
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    
    with Picamera2() as picam2:
        if picamera_config is None:
            # FIX: Ensure main stream is at least as large as lores stream
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
        print(f"Patched Picamera2 configuration: width={width}, height={height}, format={format_str}")
        appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, format={format_str}, width={width}, height={height}, "
                f"framerate=30/1, pixel-aspect-ratio=1/1"
            )
        )
        picam2.start()
        frame_count = 0
        while True:
            frame_data = picam2.capture_array('lores')
            if frame_data is None:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buffer.pts = frame_count * buffer_duration
            buffer.duration = buffer_duration
            ret = appsrc.emit('push-buffer', buffer)
            if ret == Gst.FlowReturn.FLUSHING:
                break
            if ret != Gst.FlowReturn.OK:
                break
            frame_count += 1

# Apply the patch
gstreamer_app.picamera_thread = patched_picamera_thread

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.instance_segmentation.instance_segmentation_pipeline import GStreamerInstanceSegmentationApp

# Predefined colors (BGR format) for masks
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0)   # Olive
]

# -----------------------------------------------------------------------------------------------
# User-defined GStreamer App to suppress the main window
# -----------------------------------------------------------------------------------------------
class GStreamerInstanceSegmentationAppNoDisplay(GStreamerInstanceSegmentationApp):
    def __init__(self, app_callback, user_data, parser=None):
        # We need to detect resolution before create_pipeline is called
        if parser is None:
            from hailo_apps.hailo_app_python.core.common.core import get_default_parser
            parser = get_default_parser()
        
        # Check CLI arguments first
        temp_args, _ = parser.parse_known_args()
        cli_input = temp_args.input
        
        input_source = cli_input
        if cli_input is None:
            if user_data.file_path != "NOT_SPECIFIED":
                if user_data.file_path is None or str(user_data.file_path).lower() == "null":
                    input_source = "rpi"
                else:
                    input_source = user_data.file_path
                print(f"✓ Using input source from config: {input_source}")
            else:
                input_source = "rpi"
        else:
            print(f"✓ Using input source from CLI: {input_source}")
            
        parser.set_defaults(input=input_source)
        
        self.detected_width = 1280
        self.detected_height = 720
        
        if input_source:
            if input_source.startswith("rpi"):
                if user_data.input_resolution_width and user_data.input_resolution_height:
                    self.detected_width = user_data.input_resolution_width
                    self.detected_height = user_data.input_resolution_height
                    print(f"✓ Using RPi camera with input resolution override: {self.detected_width}x{self.detected_height}")
                else:
                    self.detected_width = 2028
                    self.detected_height = 1520
                    print(f"✓ Using RPi camera native resolution: {self.detected_width}x{self.detected_height}")
            elif not input_source.startswith("libcamera") and os.path.exists(input_source):
                cap = cv2.VideoCapture(input_source)
                if cap.isOpened():
                    self.detected_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.detected_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    print(f"✓ Detected dynamic resolution from input file: {self.detected_width}x{self.detected_height}")
        
        super().__init__(app_callback, user_data, parser)

    def get_pipeline_string(self):
        from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
            SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, 
            TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, QUEUE
        )
        
        # Override base class variables to ensure camera thread uses correct resolution
        self.video_width = self.detected_width
        self.video_height = self.detected_height
        
        # 1. Source (Hardware Scaling to input_resolution happens here)
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,
                                          video_width=self.video_width, video_height=self.video_height,
                                          frame_rate=self.frame_rate, sync=self.sync)
        
        # 2. Inference
        inference_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.config_file
        )
        inference_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(inference_pipeline)
        
        # 3. Tracker
        tracker_pipeline = TRACKER_PIPELINE(class_id=1)
        
        # 4. Hardware Filter (Python Callback to remove unwanted detections)
        filter_pipeline = f"{QUEUE(name='filter_q')} ! identity name=identity_filter"
        
        # 5. Hardware Overlay (Box Drawing) - Faster than Python
        overlay_pipeline = f"{QUEUE(name='hailo_overlay_q')} ! hailooverlay"
        
        # 6. Hardware Scaling & Conversion (Only include if pixels are needed)
        needs_frame = self.user_data.gui_preview or self.user_data.web_preview or self.user_data.display_preview or self.user_data.show_boxes_only
        
        if needs_frame:
            preview_width = self.detected_width
            preview_height = self.detected_height
            
            # Ensure width/height are even (required by some hardware encoders/scalers)
            preview_width = (preview_width // 2) * 2
            preview_height = (preview_height // 2) * 2
            
            hardware_post_proc = (
                f"! {QUEUE(name='preview_scale_q')} ! videoscale name=preview_videoscale ! "
                f"video/x-raw, width={preview_width}, height={preview_height} ! "
                f"{QUEUE(name='preview_convert_q')} ! videoconvert name=preview_videoconvert ! "
                f"video/x-raw, format=RGB"
            )
        else:
            hardware_post_proc = ""
        
        # 7. User Callback (Python logic for display/grid/masks)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # 8. Hardware Sink (fakesink to suppress second window)
        sink_pipeline = f"fakesink name=hailo_display sync={self.sync}"

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{inference_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{filter_pipeline} ! '
            f'{overlay_pipeline} '
            f'{hardware_post_proc} ! '
            f'{user_callback_pipeline} ! '
            f'{sink_pipeline}'
        )
        return pipeline_string

# -----------------------------------------------------------------------------------------------
# User-defined helper functions
# -----------------------------------------------------------------------------------------------

def extract_boxes_grid(frame, detections, width, height, max_boxes=12, box_size=320):
    """
    Extract bounding boxes from frame and arrange them in a grid layout.
    Shows only the cropped content of each detection box with masks if available.
    """
    if not detections:
        placeholder = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No detections", (50, box_size // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    # Limit number of boxes
    detections = detections[:max_boxes]
    
    # Calculate grid dimensions
    num_boxes = len(detections)
    cols = int(np.ceil(np.sqrt(num_boxes)))
    rows = int(np.ceil(num_boxes / cols))
    
    # Create grid canvas
    grid_h = rows * box_size
    grid_w = cols * box_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, detection in enumerate(detections):
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Get tracking ID
        track_id = None
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        
        # Get pixel coordinates from normalized coordinates [0, 1]
        x_min = int(bbox.xmin() * width)
        y_min = int(bbox.ymin() * height)
        w = int(bbox.width() * width)
        h = int(bbox.height() * height)
        x_max = x_min + w
        y_max = y_min + h
        
        # Clamp to frame boundaries
        x_min = max(0, min(x_min, width))
        y_min = max(0, min(y_min, height))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))
        
        # Ensure valid box dimensions
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Extract box region from frame
        box_region = frame[y_min:y_max, x_min:x_max].copy()
        box_h, box_w = box_region.shape[:2]
        
        if box_h == 0 or box_w == 0:
            continue
        
        # Get mask if available
        masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
        mask_overlay = None
        if len(masks) != 0:
            mask = masks[0]
            mask_height = mask.get_height()
            mask_width = mask.get_width()
            data = np.array(mask.get_data())
            data = data.reshape((mask_height, mask_width))
            
            # Resize mask to box region size
            resized_mask = cv2.resize(data, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            resized_mask = (resized_mask > 0.5).astype(np.uint8)
            
            # Create colored mask overlay
            color = COLORS[(track_id % len(COLORS)) if track_id is not None else idx % len(COLORS)]
            mask_overlay = np.zeros_like(box_region)
            mask_overlay[:, :] = color
            mask_overlay = mask_overlay * resized_mask[:, :, np.newaxis]
        
        # Resize box to fit grid cell (maintain aspect ratio)
        scale = min(box_size / box_w, box_size / box_h)
        new_w = int(box_w * scale)
        new_h = int(box_h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_box = cv2.resize(box_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Resize mask overlay if available
            if mask_overlay is not None:
                resized_mask_overlay = cv2.resize(mask_overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Calculate position in grid
            col = idx % cols
            row = idx // cols
            
            # Center the resized box in the grid cell
            y_offset = (box_size - new_h) // 2
            x_offset = (box_size - new_w) // 2
            
            grid_y = row * box_size + y_offset
            grid_x = col * box_size + x_offset
            
            # Place box in grid
            grid[grid_y:grid_y + new_h, grid_x:grid_x + new_w] = resized_box
            
            # Apply mask overlay if available
            if mask_overlay is not None:
                grid[grid_y:grid_y + new_h, grid_x:grid_x + new_w] = cv2.addWeighted(
                    grid[grid_y:grid_y + new_h, grid_x:grid_x + new_w], 1.0,
                    resized_mask_overlay, 0.5, 0
                )
            
            # Draw border and label
            color = COLORS[(track_id % len(COLORS)) if track_id is not None else idx % len(COLORS)]
            cv2.rectangle(grid, 
                         (col * box_size, row * box_size),
                         ((col + 1) * box_size - 1, (row + 1) * box_size - 1),
                         color, 2)
            
            # Draw label with tracking ID if available
            if track_id is not None:
                label_text = f"ID:{track_id} {label}: {confidence:.2f}"
            else:
                label_text = f"{label}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = row * box_size + 20
            
            # Label background
            cv2.rectangle(grid,
                         (col * box_size, label_y - label_size[1] - 5),
                         (col * box_size + label_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Label text
            cv2.putText(grid, label_text,
                       (col * box_size + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return grid

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
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
        
        # Preview toggles
        self.web_preview = False
        self.gui_preview = True
        self.display_preview = False
        self.board = None
        
        # Pixelation output resolution control
        self.keep_pixelated_resolution = False  # If True, keep output at pixelated resolution; if False, upscale back to original
        self.pixelated_width = 64  # Default pixelation width
        self.pixelated_height = 48  # Default pixelation height
        
        self.load_config()
        self.setup_previews()

    def load_config(self):
        """Load configuration from config.json if present."""
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.show_boxes_only = config.get("show_boxes_only", False)
                    self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
                    self.frame_skip = int(config.get("frame_skip", 1))
                    
                    if "file_path" in config:
                        self.file_path = config["file_path"]
                    
                    # Handle input resolution override
                    input_res = config.get("input_resolution", {})
                    if input_res:
                        self.input_resolution_width = int(input_res.get("width", 0)) if input_res.get("width") else None
                        self.input_resolution_height = int(input_res.get("height", 0)) if input_res.get("height") else None
                    
                    # Preview toggles
                    self.web_preview = config.get("web_preview", self.web_preview)
                    self.gui_preview = config.get("gui_preview", self.gui_preview)
                    self.display_preview = config.get("display_preview", self.display_preview)
                    
                    # Pixelation output resolution control
                    self.keep_pixelated_resolution = config.get("keep_pixelated_resolution", False)
                    
                    # Handle pixelated resolution from config
                    pixelated_res = config.get("pixelated_resolution", {})
                    if pixelated_res:
                        self.pixelated_width = int(pixelated_res.get("width", 64))
                        self.pixelated_height = int(pixelated_res.get("height", 48))
                    else:
                        # Use defaults if not specified
                        self.pixelated_width = 64
                        self.pixelated_height = 48
                    
                    # Handle detection classes filtering
                    detection_classes = config.get("detection_classes", "all")
                    if isinstance(detection_classes, str) and detection_classes.strip().lower() != "all":
                        self.allowed_classes_set = {c.strip().lower() for c in detection_classes.split(",") if c.strip()}
                    else:
                        self.allowed_classes_set = None
                        
                    print(f"✓ Segmentation Config loaded: show_boxes_only={self.show_boxes_only}, "
                          f"confidence={self.confidence_threshold}, frame_skip={self.frame_skip}, "
                          f"classes={detection_classes}, "
                          f"web={self.web_preview}, gui={self.gui_preview}, display={self.display_preview}, "
                          f"keep_pixelated_resolution={self.keep_pixelated_resolution}, "
                          f"pixelated_resolution={self.pixelated_width}x{self.pixelated_height}")
                    if self.input_resolution_width and self.input_resolution_height:
                        print(f"  Input resolution override: {self.input_resolution_width}x{self.input_resolution_height}")
            except Exception as e:
                print(f"⚠️  Failed to read config.json: {e}")

    def setup_previews(self):
        """Initialize web server and Whisplay if enabled."""
        if self.web_preview:
            threading.Thread(target=start_web_server, daemon=True).start()
            
        if self.display_preview and WhisPlayBoard is not None:
            try:
                self.board = WhisPlayBoard()
                self.board.set_backlight(80)
                print("✓ Whisplay display initialized!")
            except Exception as e:
                print(f"✗ Failed to initialize Whisplay: {e}")
                self.display_preview = False

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

def filter_callback(pad, info, user_data):
    """Filter detections in hardware ROI before they reach the overlay."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get the ROI from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Remove objects that are not in the allowed classes or below confidence threshold
    for det in detections:
        label = det.get_label().lower()
        confidence = det.get_confidence()
        
        # Check both class and confidence
        is_allowed = True
        if user_data.allowed_classes_set is not None and label not in user_data.allowed_classes_set:
            is_allowed = False
        if confidence < user_data.confidence_threshold:
            is_allowed = False
            
        if not is_allowed:
            roi.remove_object(det)
            
    return Gst.PadProbeReturn.OK

def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    
    # Apply frame skip
    if user_data.get_count() % user_data.frame_skip != 0:
        return Gst.PadProbeReturn.OK
        
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Determine if we actually need the frame pixels
    needs_frame = user_data.show_boxes_only or user_data.gui_preview or user_data.web_preview or user_data.display_preview
    
    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if needs_frame and user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
    elif user_data.use_frame and not needs_frame:
        # Optimization: skip copy if no preview is active
        pass
    else:
        # If frame is not available but expected, log once
        if user_data.use_frame and not hasattr(user_data, '_frame_error_logged'):
            print("⚠️  Frame capture failed: format, width or height is None")
            user_data._frame_error_logged = True

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections (they are already filtered by filter_callback)
    # Pre-extract tracking IDs, masks, and detection info to avoid repeated calls
    detection_info = []
    string_parts = []  # OPTIMIZATION: Use list for string building (faster than concatenation)
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Get tracking ID once
        track_id = None
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        
        # OPTIMIZATION: Pre-extract mask to avoid calling get_objects_typed in loop
        masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
        mask_obj = masks[0] if len(masks) > 0 else None
        
        detection_info.append((detection, label, bbox, confidence, track_id, mask_obj))
        
        # Build string parts for printing
        if track_id is not None:
            string_parts.append(f"Segmentation: ID: {track_id} Label: {label} Confidence: {confidence:.2f}")
        else:
            string_parts.append(f"Segmentation: Label: {label} Confidence: {confidence:.2f}")
    
    # OPTIMIZATION: Join string parts once (faster than concatenation)
    if string_parts:
        string_to_print += "\n".join(string_parts) + "\n"
        
    if user_data.use_frame and frame is not None:
        # Convert to BGR for OpenCV/Preview (GStreamer is now sending RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # OPTIMIZATION: Only process pixels if at least one preview is active
        any_preview = user_data.gui_preview or user_data.web_preview or user_data.display_preview
        
        if any_preview:
            # Handle "boxes only" preview mode (Grid Mode)
            if user_data.show_boxes_only:
                if user_data.get_count() % 30 == 0:
                    print(f"DEBUG: Grid mode active. Detections: {len(detections)}")
                frame = extract_boxes_grid(frame, detections, width, height)
            else:
                # NORMAL MODE: Show only colored segmentation masks on black background
                # Start with a black background
                masked_frame = np.zeros_like(frame)
                
                # OPTIMIZATION: Early exit if no detections
                if len(detection_info) == 0:
                    frame = masked_frame
                else:
                    # OPTIMIZATION: Pre-compute colors for all detections (avoid repeated modulo)
                    colors_list = [COLORS[(track_id % len(COLORS)) if track_id is not None else idx % len(COLORS)]
                                  for idx, (_, _, _, _, track_id, _) in enumerate(detection_info)]
                    
                    # Draw colored masks for each detection (using pre-extracted info)
                    # Masks will be drawn smoothly, then full frame pixelation will be applied
                    for idx, (detection, label, bbox, confidence, track_id, mask_obj) in enumerate(detection_info):
                        # OPTIMIZATION: Use pre-extracted mask (no need to call get_objects_typed again)
                        if mask_obj is None:
                            continue
                        
                        mask = mask_obj
                        mask_height = mask.get_height()
                        mask_width = mask.get_width()
                        
                        # Calculate ROI coordinates first (before any processing)
                        roi_width = int(bbox.width() * width)
                        roi_height = int(bbox.height() * height)
                        x_min = int(bbox.xmin() * width)
                        y_min = int(bbox.ymin() * height)
                        x_max = x_min + roi_width
                        y_max = y_min + roi_height
                        
                        # Clamp to frame boundaries
                        y_min = max(y_min, 0)
                        x_min = max(x_min, 0)
                        y_max = min(y_max, frame.shape[0])
                        x_max = min(x_max, frame.shape[1])
                        
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        
                        # Get mask data and resize directly to ROI size (smooth, no pixelation)
                        data = np.array(mask.get_data())
                        data = data.reshape((mask_height, mask_width))
                        
                        # Resize mask directly to ROI size (smooth interpolation)
                        roi_h, roi_w = y_max - y_min, x_max - x_min
                        resized_mask = cv2.resize(data, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Threshold and convert to binary
                        binary_mask = (resized_mask > 0.5).astype(np.uint8)
                        
                        # OPTIMIZATION: Use pre-computed color (no repeated calculation)
                        color = colors_list[idx]
                        
                        # OPTIMIZATION: Use direct multiplication instead of np.where (faster)
                        # Create colored mask using broadcasting and multiplication
                        binary_mask_3d = binary_mask[:, :, np.newaxis]
                        colored_mask = (binary_mask_3d * color).astype(np.uint8)
                        
                        # Draw colored mask on black background
                        # Use maximum to handle overlapping masks
                        masked_frame[y_min:y_max, x_min:x_max] = np.maximum(
                            masked_frame[y_min:y_max, x_min:x_max], colored_mask
                        )
                    
                    # Replace frame with colored masks only (after all detections processed)
                    frame = masked_frame
            
            # Pixelate the entire output frame (applies to both grid and normal modes)
            # Get original frame dimensions
            orig_h, orig_w = frame.shape[:2]
            orig_aspect = orig_w / orig_h
            
            # Calculate pixelated dimensions maintaining aspect ratio
            # Use the configured pixelated resolution as a base, but adjust to match input aspect ratio
            config_pixel_w = user_data.pixelated_width
            config_pixel_h = user_data.pixelated_height
            config_aspect = config_pixel_w / config_pixel_h
            
            # Check if aspect ratios match (within small tolerance)
            aspect_match = abs(orig_aspect - config_aspect) < 0.01
            
            if aspect_match:
                # Aspect ratios match, use configured resolution directly
                OUTPUT_PIXELATED_WIDTH = config_pixel_w
                OUTPUT_PIXELATED_HEIGHT = config_pixel_h
            else:
                # Aspect ratios don't match - maintain input aspect ratio
                # Scale based on the smaller dimension to fit within configured bounds
                if orig_aspect > config_aspect:
                    # Input is wider - scale based on height
                    OUTPUT_PIXELATED_HEIGHT = config_pixel_h
                    OUTPUT_PIXELATED_WIDTH = int(config_pixel_h * orig_aspect)
                else:
                    # Input is taller - scale based on width
                    OUTPUT_PIXELATED_WIDTH = config_pixel_w
                    OUTPUT_PIXELATED_HEIGHT = int(config_pixel_w / orig_aspect)
                
                # Warn user about aspect ratio mismatch
                if not hasattr(user_data, '_aspect_warned'):
                    print(f"⚠️  Aspect ratio mismatch detected!")
                    print(f"   Input: {orig_w}x{orig_h} (aspect: {orig_aspect:.3f})")
                    print(f"   Config pixelated: {config_pixel_w}x{config_pixel_h} (aspect: {config_aspect:.3f})")
                    print(f"   Using: {OUTPUT_PIXELATED_WIDTH}x{OUTPUT_PIXELATED_HEIGHT} to maintain input aspect ratio")
                    user_data._aspect_warned = True
            
            # OPTIMIZATION: Skip pixelation if resolution already matches (no resize needed)
            if orig_w == OUTPUT_PIXELATED_WIDTH and orig_h == OUTPUT_PIXELATED_HEIGHT:
                # Already at target resolution, no pixelation needed
                pass
            else:
                # Downscale entire frame to low resolution
                pixelated_frame = cv2.resize(frame, (OUTPUT_PIXELATED_WIDTH, OUTPUT_PIXELATED_HEIGHT), 
                                           interpolation=cv2.INTER_NEAREST)
                
                # Conditionally upscale back to original size based on config
                if user_data.keep_pixelated_resolution:
                    # Keep output at pixelated resolution (no upscale)
                    frame = pixelated_frame
                else:
                    # Upscale back to original size using nearest neighbor to maintain pixelated look
                    frame = cv2.resize(pixelated_frame, (orig_w, orig_h), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Update latest_frame for web streaming
            # OPTIMIZATION: Only copy if web preview is active
            if user_data.web_preview:
                global latest_frame
                with frame_lock:
                    # Use view instead of copy if possible, but for safety use copy
                    # since frame may be modified later
                    latest_frame = frame.copy()
            
            # Send to Whisplay
            if user_data.display_preview and user_data.board:
                try:
                    pixel_data = image_to_rgb565(frame)
                    user_data.board.draw_image(0, 0, WHISPLAY_WIDTH, WHISPLAY_HEIGHT, pixel_data)
                except Exception as e:
                    print(f"✗ Whisplay display error: {e}")
                    user_data.display_preview = False
            
            # Set frame for local GUI if enabled
            if user_data.gui_preview:
                user_data.set_frame(frame)

    #print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerInstanceSegmentationAppNoDisplay(app_callback, user_data)
    
    # Force use_frame to True so the preview window starts by default
    app.options_menu.use_frame = True
    user_data.use_frame = True
    
    # Patch display_user_data_frame to respect gui_preview
    original_display = gstreamer_app.display_user_data_frame
    def patched_display(ud):
        if ud.gui_preview:
            original_display(ud)
        else:
            while ud.running:
                time.sleep(1)
    
    gstreamer_app.display_user_data_frame = patched_display
    
    # Connect the hardware filter probe BEFORE running the pipeline
    filter_identity = app.pipeline.get_by_name("identity_filter")
    if filter_identity:
        filter_pad = filter_identity.get_static_pad("src")
        filter_pad.add_probe(Gst.PadProbeType.BUFFER, filter_callback, user_data)
        print("✓ Hardware filter connected (filtering before box overlay)")
    
    app.run()


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
WEB_PORT = 8080

def image_to_rgb565(frame: np.ndarray) -> list:
    """Convert OpenCV BGR frame to RGB565 byte list for Whisplay display."""
    # Resize to fit Whisplay
    resized = cv2.resize(frame, (WHISPLAY_WIDTH, WHISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Efficient conversion to RGB565 using numpy
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    
    # Convert to big-endian bytes
    high_byte = (rgb565 >> 8).astype(np.uint8)
    low_byte = (rgb565 & 0xFF).astype(np.uint8)
    
    # Stack and flatten
    pixel_data = np.stack((high_byte, low_byte), axis=2).flatten().tolist()
    return pixel_data

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hailo RPi5 Detection Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #1a1a1a; color: #ffffff; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #4CAF50; }
        .video-container { text-align: center; background-color: #000; padding: 10px; border-radius: 10px; margin: 20px 0; }
        img { max-width: 100%; height: auto; border: 2px solid #4CAF50; border-radius: 5px; }
        .info { background-color: #2a2a2a; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 3px; margin: 5px; background-color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Hailo RPi5 Detection Preview</h1>
        <div class="info">
            <strong>Status:</strong> <span class="status">LIVE</span>
        </div>
        <div class="video-container">
            <img src="/video_feed" alt="Live Detection Stream">
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
                # latest_frame is already BGR (ready for encode)
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
    print(f"\n🌐 Web server starting at http://{local_ip}:{WEB_PORT}")
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
            # We use the requested resolution for both to avoid "lores exceeds main" error
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
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined GStreamer App to suppress the main window
# -----------------------------------------------------------------------------------------------
class GStreamerDetectionAppNoDisplay(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, parser=None):
        # We need to detect resolution before create_pipeline is called
        if parser is None:
            from hailo_apps.hailo_app_python.core.common.core import get_default_parser
            parser = get_default_parser()
        
        # Check CLI arguments first
        temp_args, _ = parser.parse_known_args()
        cli_input = temp_args.input
        
        # Priority: 
        # 1. CLI Argument (if provided and not None)
        # 2. config.json file_path (if specified and not "NOT_SPECIFIED")
        # 3. Default (rpi or example video)
        
        input_source = cli_input
        
        if cli_input is None:
            # Use config.json if CLI is not provided
            if user_data.file_path != "NOT_SPECIFIED":
                if user_data.file_path is None or str(user_data.file_path).lower() == "null":
                    input_source = "rpi"
                else:
                    input_source = user_data.file_path
                print(f"✓ Using input source from config: {input_source}")
            else:
                # Default fallback
                input_source = "rpi"
        else:
            print(f"✓ Using input source from CLI: {input_source}")
            
        # Update the parser's default value so the parent class uses it
        parser.set_defaults(input=input_source)
        
        self.detected_width = 1280
        self.detected_height = 720
        
        if input_source:
            if input_source.startswith("rpi"):
                # Raspberry Pi camera dynamic resolution
                # You can customize these defaults for RPi here
                self.detected_width = 2028
                self.detected_height = 1520
                print(f"✓ Using RPi camera resolution: {self.detected_width}x{self.detected_height}")
            elif not input_source.startswith("libcamera") and os.path.exists(input_source):
                # If it's a file, try to get resolution
                cap = cv2.VideoCapture(input_source)
                if cap.isOpened():
                    self.detected_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.detected_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    print(f"✓ Detected dynamic resolution from input file: {self.detected_width}x{self.detected_height}")
        
        # Initialize the parent class
        super().__init__(app_callback, user_data, parser)

    def get_pipeline_string(self):
        # Override dimensions with detected values
        self.video_width = self.detected_width
        self.video_height = self.detected_height
        # Set video_sink to fakesink to suppress the main GStreamer window
        self.video_sink = "fakesink"
        return super().get_pipeline_string()

# -----------------------------------------------------------------------------------------------
# User-defined helper functions
# -----------------------------------------------------------------------------------------------

def extract_boxes_grid(frame, detections, width, height, max_boxes=12, box_size=320):
    """
    Extract bounding boxes from frame and arrange them in a grid layout.
    Shows only the cropped content of each detection box.
    """
    if not detections:
        # Return a placeholder if no detections
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
    
    # Color palette for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
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
        
        # Resize box to fit grid cell (maintain aspect ratio)
        scale = min(box_size / box_w, box_size / box_h)
        new_w = int(box_w * scale)
        new_h = int(box_h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_box = cv2.resize(box_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
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
            
            # Draw border and label
            color = colors[idx % len(colors)]
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
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.show_boxes_only = False
        self.use_frame = True  # Ensure frame is captured for preview
        self.allowed_classes_set = None
        self.confidence_threshold = 0.5
        self.frame_skip = 1
        self.file_path = "NOT_SPECIFIED"
        self.display_scale = 1.0
        
        # Preview toggles
        self.web_preview = False
        self.gui_preview = True
        self.display_preview = False
        self.board = None
        
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
                    self.display_scale = float(config.get("display_scale", 1.0))
                    
                    if "file_path" in config:
                        self.file_path = config["file_path"]
                    
                    # New preview toggles
                    self.web_preview = config.get("web_preview", self.web_preview)
                    self.gui_preview = config.get("gui_preview", self.gui_preview)
                    self.display_preview = config.get("display_preview", self.display_preview)
                    
                    # Handle detection classes filtering
                    detection_classes = config.get("detection_classes", "all")
                    if isinstance(detection_classes, str) and detection_classes.strip().lower() != "all":
                        self.allowed_classes_set = {c.strip().lower() for c in detection_classes.split(",") if c.strip()}
                    else:
                        self.allowed_classes_set = None
                        
                    print(f"✓ Config loaded: show_boxes_only={self.show_boxes_only}, "
                          f"confidence={self.confidence_threshold}, frame_skip={self.frame_skip}, "
                          f"scale={self.display_scale}, classes={detection_classes}, "
                          f"web={self.web_preview}, gui={self.gui_preview}, display={self.display_preview}")
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

    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
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

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
    else:
        # If frame is not available but expected, log once
        if user_data.use_frame and not hasattr(user_data, '_frame_error_logged'):
            print("⚠️  Frame capture failed: format, width or height is None")
            user_data._frame_error_logged = True

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Filter detections by confidence and allowed classes
    filtered_detections = []
    for det in detections:
        label = det.get_label().lower()
        confidence = det.get_confidence()
        
        # Check confidence threshold
        if confidence < user_data.confidence_threshold:
            continue
            
        # Check class filter
        if user_data.allowed_classes_set is not None:
            if label not in user_data.allowed_classes_set:
                continue
        
        filtered_detections.append(det)
    
    detections = filtered_detections

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Get tracking ID for all detections
        track_id = None
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        
        # Print detection info with tracking ID
        if track_id is not None:
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
        else:
            string_to_print += (f"Detection: Label: {label} Confidence: {confidence:.2f}\n")
        detection_count += 1
        
    if user_data.use_frame and frame is not None:
        # OPTIMIZATION: Only process frame if at least one preview is enabled
        any_preview = user_data.gui_preview or user_data.web_preview or user_data.display_preview
        
        if any_preview:
            # Handle "boxes only" preview mode or normal mode
            if user_data.show_boxes_only:
                # Create grid of cropped boxes
                if user_data.get_count() % 30 == 0:
                    print(f"DEBUG: Grid mode active. Detections: {len(detections)}")
                frame = extract_boxes_grid(frame, detections, width, height)
            else:
                if user_data.get_count() % 30 == 0:
                    print(f"DEBUG: Normal mode active. Detections: {len(detections)}")
                # Normal mode: Draw bounding boxes on full frame
                for detection in detections:
                    label = detection.get_label()
                    bbox = detection.get_bbox()
                    confidence = detection.get_confidence()
                    
                    # Get tracking ID
                    track_id = None
                    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                    if len(track) == 1:
                        track_id = track[0].get_id()
                    
                    x_min = int(bbox.xmin() * width)
                    y_min = int(bbox.ymin() * height)
                    w = int(bbox.width() * width)
                    h = int(bbox.height() * height)
                    x_max = x_min + w
                    y_max = y_min + h
                    
                    # Clamp coordinates
                    x_min = max(0, min(x_min, width))
                    y_min = max(0, min(y_min, height))
                    x_max = max(0, min(x_max, width))
                    y_max = max(0, min(y_max, height))
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Build label with tracking ID if available
                    if track_id is not None:
                        label_text = f"ID:{track_id} {label}: {confidence:.2f}"
                    else:
                        label_text = f"{label}: {confidence:.2f}"
                    
                    cv2.putText(frame, label_text, (x_min, y_min - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print the detection count and other info to the frame
            cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert the frame to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Apply display scale if configured
            if user_data.display_scale != 1.0:
                h, w = frame.shape[:2]
                new_w = int(w * user_data.display_scale)
                new_h = int(h * user_data.display_scale)
                if new_w > 0 and new_h > 0:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Update latest_frame for web streaming
            if user_data.web_preview:
                global latest_frame
                with frame_lock:
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

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    # app = GStreamerDetectionApp(app_callback, user_data)
    app = GStreamerDetectionAppNoDisplay(app_callback, user_data)
    
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
    
    app.run()


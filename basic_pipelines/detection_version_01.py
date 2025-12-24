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
from hailo_apps.hailo_app_python.core.gstreamer import gstreamer_app

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
        
        # Override the input if provided in config.json
        if user_data.file_path != "NOT_SPECIFIED":
            # If JSON null (Python None) or string "null", default to "rpi"
            if user_data.file_path is None or str(user_data.file_path).lower() == "null":
                input_source = "rpi"
            else:
                input_source = user_data.file_path
            
            # Update the parser's default value so the parent class uses it
            parser.set_defaults(input=input_source)
            print(f"✓ Using input source from config: {input_source}")
        else:
            temp_args, _ = parser.parse_known_args()
            input_source = temp_args.input
        
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
            
            # Draw label
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
        self.load_config()

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
                    
                    # Handle detection classes filtering
                    detection_classes = config.get("detection_classes", "all")
                    if isinstance(detection_classes, str) and detection_classes.strip().lower() != "all":
                        self.allowed_classes_set = {c.strip().lower() for c in detection_classes.split(",") if c.strip()}
                    else:
                        self.allowed_classes_set = None
                        
                    print(f"✓ Config loaded: show_boxes_only={self.show_boxes_only}, "
                          f"confidence={self.confidence_threshold}, frame_skip={self.frame_skip}, "
                          f"classes={detection_classes}, file_path={self.file_path}")
            except Exception as e:
                print(f"⚠️  Failed to read config.json: {e}")

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
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
        detection_count += 1
        
    if user_data.use_frame and frame is not None:
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
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print the detection count and other info to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert the frame to BGR for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
    
    app.run()


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
    # Limit queue size to prevent buffer accumulation and increasing delay
    # max-bytes: maximum bytes in queue (roughly 2-3 frames at 1280x720 RGB)
    frame_bytes = video_width * video_height * 3  # RGB = 3 bytes per pixel
    appsrc.set_property("max-bytes", frame_bytes * 3)  # Allow max 3 frames in queue
    appsrc.set_property("max-buffers", 3)  # Max 3 buffers in queue
    # Note: leaky-type=downstream is set in the pipeline string to drop old buffers when queue is full
    
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
        import time
        last_frame_time = time.time()
        target_frame_time = 1.0 / 30.0  # 30 FPS target
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while True:
            # Throttle frame rate to prevent buffer accumulation
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            last_frame_time = time.time()
            
            frame_data = picam2.capture_array('lores')
            if frame_data is None:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # Flip frame horizontally to fix upside-down camera
            frame = cv2.flip(frame, -1)  # -1 = flip both horizontally and vertically (rotate 180)
            frame = np.asarray(frame)
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buffer.pts = frame_count * buffer_duration
            buffer.duration = buffer_duration
            ret = appsrc.emit('push-buffer', buffer)
            
            if ret == Gst.FlowReturn.FLUSHING:
                break
            elif ret != Gst.FlowReturn.OK:
                # Pipeline is full or error occurred, wait a bit before trying again
                # This handles backpressure when the pipeline can't keep up
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print(f"⚠️  Pipeline backpressure (ret={ret}), dropping frame {frame_count}")
                    consecutive_errors = 0
                time.sleep(0.01)  # Small delay to let pipeline catch up
                continue
            
            # Reset error counter on success
            consecutive_errors = 0
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
        print("🔍 [LOG] Step 1: Creating source pipeline...")
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,
                                          video_width=self.video_width, video_height=self.video_height,
                                          frame_rate=self.frame_rate, sync=self.sync)
        
        # Note: Camera flip is handled in patched_picamera_thread using cv2.flip
        # This ensures the flip happens at the source for appsrc-based pipelines
        
        print(f"   ✓ Source pipeline type: {type(source_pipeline)}, length: {len(source_pipeline) if isinstance(source_pipeline, str) else 'N/A'}")
        print(f"   ✓ Source pipeline preview: {source_pipeline[:150] if isinstance(source_pipeline, str) else str(source_pipeline)[:150]}")
        
        # 2. Segmentation Inference
        print("🔍 [LOG] Step 2: Creating segmentation inference pipeline...")
        inference_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.config_file,
            name='segmentation_inference'
        )
        print(f"   ✓ Inference pipeline type: {type(inference_pipeline)}, value: {str(inference_pipeline)[:200]}")
        inference_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(inference_pipeline, name='segmentation_inference_wrapper')
        print(f"   ✓ Inference wrapper type: {type(inference_pipeline_wrapper)}, length: {len(inference_pipeline_wrapper) if isinstance(inference_pipeline_wrapper, str) else 'N/A'}")
        print(f"   ✓ Inference wrapper preview: {inference_pipeline_wrapper[:200] if isinstance(inference_pipeline_wrapper, str) else str(inference_pipeline_wrapper)[:200]}")
        
        # 2b. Depth Inference (if depth HEF path is available) - Sequential after segmentation
        depth_inference_wrapper = None
        depth_inference_pipeline_direct = None
        if self.user_data.depth_enabled and self.user_data.depth_hef_path:
            print("🔍 [LOG] Step 2b: Creating depth inference pipeline...")
            try:
                # Check if using fast_depth.hef - it doesn't need post-processing
                # libdepth_postprocess.so is only for scdepthv3
                depth_hef_name = Path(self.user_data.depth_hef_path).name.lower()
                is_fast_depth = "fast_depth" in depth_hef_name
                
                if is_fast_depth:
                    print(f"   ✓ Detected fast_depth model - skipping post-processing")
                    depth_post_so_str = None
                    depth_post_function = None
                else:
                    # For scdepthv3, use post-processing
                    project_root = Path(__file__).resolve().parent.parent
                    depth_post_so = project_root / "resources" / "so" / "libdepth_postprocess.so"
                    depth_post_so_str = str(depth_post_so) if depth_post_so.exists() else None
                    depth_post_function = "filter" if depth_post_so_str else None
                    print(f"   ✓ Depth post-process SO: {depth_post_so_str}")
                
                # Use depth inference pipeline with wrapper (like depth_version_01.py)
                # The wrapper is needed for proper metadata attachment
                # Even though hailocropper is in the wrapper, depth metadata should still be at top-level ROI
                depth_inference_pipeline = INFERENCE_PIPELINE(
                    hef_path=self.user_data.depth_hef_path,
                    post_process_so=depth_post_so_str,
                    post_function_name=depth_post_function,
                    batch_size=self.batch_size,
                    name='depth_inference'
                )
                depth_inference_wrapper = INFERENCE_PIPELINE_WRAPPER(depth_inference_pipeline, name='depth_inference_wrapper')
                depth_inference_pipeline_direct = None  # Use wrapper instead
                print(f"   ✓ Depth inference pipeline type: {type(depth_inference_pipeline)}, value: {str(depth_inference_pipeline)[:200]}")
                print(f"   ✓ Depth wrapper type: {type(depth_inference_wrapper)}, length: {len(depth_inference_wrapper) if isinstance(depth_inference_wrapper, str) else 'N/A'}")
                print(f"✓ Depth inference pipeline configured with HEF: {self.user_data.depth_hef_path}")
            except Exception as e:
                print(f"⚠️  Failed to configure depth inference pipeline: {e}")
                import traceback
                traceback.print_exc()
                depth_inference_wrapper = None
                depth_inference_pipeline_direct = None
                self.user_data.depth_enabled = False
        
        # 3. Tracker
        print("🔍 [LOG] Step 3: Creating tracker pipeline...")
        tracker_pipeline = TRACKER_PIPELINE(class_id=1)
        print(f"   ✓ Tracker pipeline: {tracker_pipeline[:150] if isinstance(tracker_pipeline, str) else str(tracker_pipeline)[:150]}")
        
        # 4. Hardware Filter (Python Callback to remove unwanted detections)
        print("🔍 [LOG] Step 4: Creating filter pipeline...")
        filter_pipeline = f"{QUEUE(name='filter_q')} ! identity name=identity_filter"
        print(f"   ✓ Filter pipeline: {filter_pipeline}")
        
        # 5. Hardware Overlay (Box Drawing) - Faster than Python
        print("🔍 [LOG] Step 5: Creating overlay pipeline...")
        overlay_pipeline = f"{QUEUE(name='hailo_overlay_q')} ! hailooverlay"
        print(f"   ✓ Overlay pipeline: {overlay_pipeline}")
        
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

        # Build pipeline string - run depth BEFORE segmentation so depth processes raw video
        # Check if depth metadata persists through segmentation and can be accessed in main callback
        print("🔍 [LOG] Step 6: Building final pipeline string...")
        if depth_inference_wrapper is not None:
            # Fallback: use wrapper if direct pipeline not available
            print("   → Building pipeline WITH depth (depth → depth_extractor → queue → segmentation)")
            depth_extractor = f"identity name=depth_extractor"
            intermediate_queue = QUEUE(name='depth_to_segmentation_q')
            pipeline_string = (
                f'{source_pipeline} ! '
                f'{depth_inference_wrapper} ! '
                f'{depth_extractor} ! '
                f'{intermediate_queue} ! '
                f'{inference_pipeline_wrapper} ! '
                f'{tracker_pipeline} ! '
                f'{filter_pipeline} ! '
                f'{overlay_pipeline} '
                f'{hardware_post_proc} ! '
                f'{user_callback_pipeline} ! '
                f'{sink_pipeline}'
            )
            print(f"✓ Pipeline configured with sequential depth inference (before segmentation)")
            print(f"   Note: Depth metadata should persist in buffer and be accessible in main callback")
        else:
            # Standard pipeline without depth
            print("   → Building pipeline WITHOUT depth")
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
        
        print(f"🔍 [LOG] Final pipeline string length: {len(pipeline_string)}")
        print(f"🔍 [LOG] Final pipeline preview (first 500 chars):\n{pipeline_string[:500]}")
        print(f"🔍 [LOG] Final pipeline preview (last 500 chars):\n{pipeline_string[-500:]}")
        print("🔍 [LOG] Pipeline string construction complete. Returning to GStreamer...")
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
        
        # Depth integration
        self.depth_hef_path = None  # Path to depth model HEF file
        self.depth_enabled = False  # Whether depth processing is enabled
        self.depth_data_cache = None  # Cache for depth data extracted before segmentation
        self.depth_min_cache = 0
        self.depth_max_cache = 1
        
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
                    
                    # Handle depth configuration
                    depth_config = config.get("depth", {})
                    if depth_config:
                        depth_path = depth_config.get("hef_path")
                        if depth_path:
                            # Resolve relative path
                            if not os.path.isabs(depth_path):
                                project_root = Path(__file__).resolve().parent.parent
                                depth_path = project_root / depth_path
                            else:
                                depth_path = Path(depth_path)
                            if depth_path.exists():
                                self.depth_hef_path = str(depth_path)
                                self.depth_enabled = True
                                print(f"✓ Depth HEF path from config: {self.depth_hef_path}")
                            else:
                                print(f"⚠️  Depth HEF file not found: {depth_path}")
                        else:
                            # Try default fast_depth.hef location
                            project_root = Path(__file__).resolve().parent.parent
                            default_depth_hef = project_root / "resources" / "models" / "hailo8l" / "fast_depth.hef"
                            if default_depth_hef.exists():
                                self.depth_hef_path = str(default_depth_hef)
                                self.depth_enabled = True
                                print(f"✓ Using default depth model: {self.depth_hef_path}")
                    
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
                          f"pixelated_resolution={self.pixelated_width}x{self.pixelated_height}, "
                          f"depth_enabled={self.depth_enabled}")
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

def depth_extraction_callback(pad, info, user_data):
    """Extract depth data right after depth inference, before segmentation processes it."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Only extract if depth is enabled
    if not user_data.depth_enabled:
        return Gst.PadProbeReturn.OK
    
    # Get the ROI from the buffer (this should have depth data from depth inference)
    roi = hailo.get_roi_from_buffer(buffer)
    depth_masks = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    
    # Detailed debug logging (first few frames)
    if not hasattr(user_data, '_depth_extraction_frame_count'):
        user_data._depth_extraction_frame_count = 0
    
    user_data._depth_extraction_frame_count += 1
    
    if user_data._depth_extraction_frame_count <= 5:
        print(f"🔍 [DEPTH EXTRACT] Frame {user_data._depth_extraction_frame_count}:")
        print(f"   ROI type: {type(roi)}")
        print(f"   Depth masks found: {len(depth_masks)}")
        
        # Check all objects in ROI
        try:
            all_objects = roi.get_objects()
            print(f"   Total objects in ROI: {len(all_objects)}")
            for idx, obj in enumerate(all_objects[:3]):
                obj_type = type(obj).__name__
                obj_depth = obj.get_objects_typed(hailo.HAILO_DEPTH_MASK)
                print(f"     Object {idx}: {obj_type}, depth masks: {len(obj_depth)}")
        except Exception as e:
            print(f"   Error getting objects: {e}")
    
    if len(depth_masks) > 0:
        depth_mask = depth_masks[0]
        depth_data = np.array(depth_mask.get_data())
        depth_h = depth_mask.get_height()
        depth_w = depth_mask.get_width()
        depth_data = depth_data.reshape((depth_h, depth_w))
        
        # Cache the raw depth data (we'll resize it later in the main callback)
        user_data.depth_data_cache = depth_data
        user_data.depth_min_cache = depth_data.min()
        user_data.depth_max_cache = depth_data.max()
        if user_data.depth_max_cache <= user_data.depth_min_cache:
            user_data.depth_max_cache = user_data.depth_min_cache + 1
        
        # Debug logging (first frame only)
        if not hasattr(user_data, '_depth_extraction_logged'):
            print(f"✓ Depth extraction callback: extracted {depth_h}x{depth_w} depth data")
            print(f"   Depth range: min={user_data.depth_min_cache:.3f}, max={user_data.depth_max_cache:.3f}")
            user_data._depth_extraction_logged = True
    else:
        # Debug logging (first frame only)
        if not hasattr(user_data, '_depth_extraction_logged'):
            print(f"⚠️  Depth extraction callback: no depth masks found in ROI")
            user_data._depth_extraction_logged = True
    
    return Gst.PadProbeReturn.OK

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
    
    # Get depth data if depth is enabled
    # First try cached data from depth extraction callback, then try to get from ROI directly
    depth_data_full = None
    depth_min = 0
    depth_max = 1
    
    # Try to get depth from ROI directly (in case it persists through the pipeline)
    if user_data.depth_enabled:
        roi_depth_masks = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        if len(roi_depth_masks) > 0 and user_data.depth_data_cache is None:
            # Found depth in ROI but not in cache - extract it now
            depth_mask = roi_depth_masks[0]
            depth_data = np.array(depth_mask.get_data())
            depth_h = depth_mask.get_height()
            depth_w = depth_mask.get_width()
            depth_data = depth_data.reshape((depth_h, depth_w))
            user_data.depth_data_cache = depth_data
            user_data.depth_min_cache = depth_data.min()
            user_data.depth_max_cache = depth_data.max()
            if user_data.depth_max_cache <= user_data.depth_min_cache:
                user_data.depth_max_cache = user_data.depth_min_cache + 1
            if not hasattr(user_data, '_depth_found_in_main_callback'):
                print(f"✓ [MAIN CALLBACK] Found depth in ROI: {depth_h}x{depth_w}")
                user_data._depth_found_in_main_callback = True
    
    if user_data.depth_enabled and user_data.depth_data_cache is not None:
        depth_data = user_data.depth_data_cache
        depth_h, depth_w = depth_data.shape
        
        # Handle letterboxing for fast_depth (outputs square 224x224, but input may not be square)
        if format is not None and width is not None and height is not None:
            input_aspect = width / height if height > 0 else 1.0
            model_aspect = 1.0  # fast_depth is always square (224x224)
            
            # Calculate the actual content area in the model output (excluding letterboxing)
            if abs(input_aspect - model_aspect) > 0.01:  # Aspect ratios don't match
                # Input was letterboxed to fit square model input
                if input_aspect > model_aspect:
                    # Input is wider (e.g., 16:9) - letterboxing on top/bottom
                    content_h = int(depth_w / input_aspect)  # Height of content in square output
                    content_w = depth_w  # Full width
                    y_offset = (depth_h - content_h) // 2  # Top padding
                    x_offset = 0
                else:
                    # Input is taller (e.g., 9:16) - letterboxing on left/right
                    content_w = int(depth_h * input_aspect)  # Width of content in square output
                    content_h = depth_h  # Full height
                    x_offset = (depth_w - content_w) // 2  # Left padding
                    y_offset = 0
                
                # Extract only the content area (excluding letterboxing)
                depth_content = depth_data[y_offset:y_offset+content_h, x_offset:x_offset+content_w]
                
                # Resize content area to original input dimensions
                depth_data_full = cv2.resize(depth_content, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                # Aspect ratios match (both square) - no letterboxing, resize directly
                depth_data_full = cv2.resize(depth_data, (width, height), interpolation=cv2.INTER_LINEAR)
            
            depth_min = user_data.depth_min_cache
            depth_max = user_data.depth_max_cache
            
            # Debug logging (first frame only)
            if not hasattr(user_data, '_depth_usage_logged'):
                print(f"✓ Using cached depth data: model output={depth_h}x{depth_w}, resized to {width}x{height}")
                print(f"   Depth range: min={depth_min:.3f}, max={depth_max:.3f}")
                if abs(input_aspect - model_aspect) > 0.01:
                    print(f"   Letterboxing handled: extracted {content_h}x{content_w} from {depth_h}x{depth_w}")
                user_data._depth_usage_logged = True
    elif user_data.depth_enabled and user_data.depth_data_cache is None:
        # Debug logging (first frame only)
        if not hasattr(user_data, '_depth_usage_logged'):
            print(f"⚠️  Depth enabled but depth_data_cache is None - depth extraction callback may not be working")
            user_data._depth_usage_logged = True

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
                pixelated_frame = frame.copy()
            else:
                # Downscale entire frame to low resolution
                pixelated_frame = cv2.resize(frame, (OUTPUT_PIXELATED_WIDTH, OUTPUT_PIXELATED_HEIGHT), 
                                           interpolation=cv2.INTER_NEAREST)
            
            # Blend depth with pixelated masks if depth is enabled
            if user_data.depth_enabled and depth_data_full is not None:
                # Pixelate depth to match pixelated frame resolution
                pixelated_depth = cv2.resize(depth_data_full, (OUTPUT_PIXELATED_WIDTH, OUTPUT_PIXELATED_HEIGHT), 
                                            interpolation=cv2.INTER_NEAREST)
                
                # Log pixelated depth info
                if not hasattr(user_data, '_pixelated_depth_logged'):
                    print(f"✓ [BLEND] Pixelated depth ready: shape={pixelated_depth.shape}, range=[{pixelated_depth.min():.3f}, {pixelated_depth.max():.3f}]")
                    print(f"   Pixelated frame shape: {pixelated_frame.shape}")
                    mask_pixels_count = np.sum((pixelated_frame > 0).any(axis=2))
                    print(f"   Mask pixels to blend: {mask_pixels_count}")
                    user_data._pixelated_depth_logged = True
                
                # Normalize depth to 0-1 range (closer = higher value, farther = lower value)
                if depth_max > depth_min:
                    # Invert so closer objects have higher values (for brightness)
                    normalized_depth = 1.0 - ((pixelated_depth - depth_min) / (depth_max - depth_min))
                    normalized_depth = np.clip(normalized_depth, 0.0, 1.0)
                else:
                    normalized_depth = np.ones((OUTPUT_PIXELATED_HEIGHT, OUTPUT_PIXELATED_WIDTH), dtype=np.float32)
                
                # Apply depth modulation to colored masks
                # Closer objects (high depth value) = brighter, farther (low depth value) = darker
                # Only modulate pixels that are part of masks (non-black pixels)
                mask_pixels = (pixelated_frame > 0).any(axis=2)  # Find non-black pixels
                
                # Make the depth effect more pronounced: 0.2 (minimum brightness) + 0.8 * depth
                # This creates a stronger contrast between close and far objects
                depth_factor = 0.2 + 0.8 * normalized_depth
                depth_factor_3d = depth_factor[:, :, np.newaxis]
                
                # Log depth factor info
                if not hasattr(user_data, '_depth_factor_logged'):
                    if np.sum(mask_pixels) > 0:
                        depth_factor_in_masks = depth_factor[mask_pixels]
                        print(f"✓ [BLEND] Depth factor range in masks: [{depth_factor_in_masks.min():.3f}, {depth_factor_in_masks.max():.3f}]")
                    user_data._depth_factor_logged = True
                
                # Apply depth modulation only to mask pixels
                pixelated_frame = np.where(
                    mask_pixels[:, :, np.newaxis],
                    (pixelated_frame * depth_factor_3d).astype(np.uint8),
                    pixelated_frame
                )
                
                # Debug logging (first frame only)
                if not hasattr(user_data, '_depth_blend_logged'):
                    mask_count = np.sum(mask_pixels)
                    depth_min_val = normalized_depth[mask_pixels].min() if mask_count > 0 else 0
                    depth_max_val = normalized_depth[mask_pixels].max() if mask_count > 0 else 0
                    print(f"✓ [BLEND] Depth blending applied: {mask_count} mask pixels")
                    print(f"   Normalized depth range in masks: {depth_min_val:.3f} to {depth_max_val:.3f}")
                    print(f"   Brightness range: {0.2 + 0.8 * depth_min_val:.3f} to {0.2 + 0.8 * depth_max_val:.3f}")
                    user_data._depth_blend_logged = True
            elif user_data.depth_enabled and depth_data_full is None:
                # Debug logging (first frame only)
                if not hasattr(user_data, '_depth_blend_logged'):
                    print(f"⚠️  [BLEND] Depth enabled but depth_data_full is None - blending skipped")
                    print(f"   depth_data_cache is None: {user_data.depth_data_cache is None}")
                    user_data._depth_blend_logged = True
            
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
    
    # Connect the depth extraction probe if depth is enabled
    # The depth wrapper's hailocropper expects detections, so it creates empty ROI when depth runs first
    # Try to probe the depth inference element's output directly, before the wrapper's cropper
    if user_data.depth_enabled:
        # Try the depth inference element's output pad (before wrapper's cropper)
        depth_inference = app.pipeline.get_by_name("depth_inference")
        if depth_inference:
            depth_pad = depth_inference.get_static_pad("src")
            if depth_pad:
                depth_pad.add_probe(Gst.PadProbeType.BUFFER, depth_extraction_callback, user_data)
                print("✓ Depth extraction callback connected on depth_inference element output (before wrapper cropper)")
            else:
                print("⚠️  depth_inference found but has no src pad")
        else:
            # Fallback: try the identity element we added
            depth_extractor = app.pipeline.get_by_name("depth_extractor")
            if depth_extractor:
                depth_pad = depth_extractor.get_static_pad("src")
                if depth_pad:
                    depth_pad.add_probe(Gst.PadProbeType.BUFFER, depth_extraction_callback, user_data)
                    print("✓ Depth extraction callback connected on depth_extractor (extracting depth before segmentation)")
                else:
                    print("⚠️  depth_extractor found but has no src pad")
            else:
                # List available elements for debugging
                print(f"⚠️  Depth enabled but depth_inference element not found")
                try:
                    all_elements = [e.get_name() for e in app.pipeline.iterate_elements()]
                    depth_related = [e for e in all_elements if 'depth' in e.lower()]
                    print(f"   Depth-related elements found: {depth_related[:10]}")
                except:
                    pass
    
    app.run()


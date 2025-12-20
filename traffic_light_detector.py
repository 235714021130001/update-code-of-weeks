#!/usr/bin/env python3
"""
🚦 Traffic Light Detection using RKNN Model
===========================================

Sử dụng model YOLOv11-seg RKNN để phát hiện đèn giao thông:
- Class 1: green (đèn xanh)
- Class 6: red (đèn đỏ)
- Class 11: yellow (đèn vàng)

Model: best_fix_1712_11nano_fp16.rknn
Input: 288x640 (H×W)
"""

import cv2
import numpy as np
import time
import os
import glob
from datetime import datetime
from collections import deque

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    print("⚠️ RKNNLite not available, will use dummy mode")


class TrafficLightDetector:
    """
    Detector đèn giao thông sử dụng RKNN NPU
    
    Features:
    - Phát hiện 3 loại đèn: GREEN, RED, YELLOW
    - Vẽ bounding box + confidence
    - Hiển thị FPS realtime
    """
    
    def __init__(self, model_path, target_size=(288, 640)):
        """
        Args:
            model_path: Path to .rknn model
            target_size: (H, W) model input size
        """
        self.model_path = model_path
        self.target_h, self.target_w = target_size
        
        # Traffic light class IDs
        self.TRAFFIC_LIGHT_CLASSES = {
            1: 'GREEN',
            6: 'RED',
            11: 'YELLOW'
        }
        
        # Colors for visualization (BGR)
        self.CLASS_COLORS = {
            1: (0, 255, 0),      # GREEN
            6: (0, 0, 255),      # RED
            11: (0, 255, 255)    # YELLOW
        }
        
        # Processing area colors
        self.PURPLE_COLOR = (255, 0, 255)   # Màu tím - trong vùng xử lý
        self.ORANGE_COLOR = (0, 165, 255)   # Màu cam - ngoài vùng xử lý
        
        # Processing area thresholds
        self.AREA_MIN = 22000  # 22k px²
        self.AREA_MAX = 45000  # 45k px²
        
        # Purple box offset settings (dễ tùy chỉnh)
        self.PURPLE_OFFSET_SMALL = 70   # Offset cho area < 40k
        self.PURPLE_OFFSET_LARGE = 100  # Offset tối đa cho area >= 40k (hoặc max space)
        
        # RKNN model
        self.rknn = None
        
        # YOLOv11-seg output structure
        self.num_classes = 12
        self.mask_dim = 32
        
        # Detection confidence - TĂNG LÊN để giảm false positives
        self.conf_threshold = 0.40  # Tăng từ 0.40 lên 0.60
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        
        print(f"🚦 Traffic Light Detector initialized")
        print(f"   Model: {model_path}")
        print(f"   Input size: {self.target_h}×{self.target_w}")
        print(f"   Classes: {list(self.TRAFFIC_LIGHT_CLASSES.values())}")
    
    def load_model(self):
        """Load RKNN model to NPU"""
        if not RKNN_AVAILABLE:
            print("❌ RKNNLite not available!")
            return False
        
        print("\n📦 Loading RKNN model...")
        
        try:
            self.rknn = RKNNLite()
            
            # Load model
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                print(f"❌ Load model failed! ret={ret}")
                return False
            
            print("   ✓ Model loaded")
            
            # Init runtime (use all 3 NPU cores)
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                print(f"❌ Init runtime failed! ret={ret}")
                return False
            
            print("   ✓ Runtime initialized (NPU 3 cores)")
            print("✅ Model ready!\n")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def release(self):
        """Release RKNN model"""
        if self.rknn:
            self.rknn.release()
            print("🔓 RKNN model released")
    
    def preprocess(self, frame):
        """
        Preprocess frame cho RKNN model
        
        Args:
            frame: BGR image from camera (480×640)
            
        Returns:
            Preprocessed image (1×288×640×3) - with batch dimension
        """
        # Crop center (480×640 → 288×640)
        h, w = frame.shape[:2]
        crop_h = self.target_h
        start_y = (h - crop_h) // 2
        end_y = start_y + crop_h
        
        img = frame[start_y:end_y, :]  # (288, 640, 3)
        
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension: (288, 640, 3) → (1, 288, 640, 3)
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    def postprocess(self, outputs):
        """
        Postprocess RKNN outputs để lấy traffic light detections và masks
        
        Returns:
            List of detections: [{'class_id', 'confidence', 'bbox', 'mask_coeff'}, ...]
        """
        detections = []
        mask_proto = None  # Khởi tạo ngay từ đầu
        
        if outputs is None or len(outputs) == 0:
            print("⚠️ No outputs from model")
            return detections, None
        
        try:
            output = outputs[0]  # (1, 48, 8400)
            
            # Kiểm tra xem có mask proto không
            mask_proto = None
            if len(outputs) > 1 and outputs[1] is not None:
                mask_proto = outputs[1]  # (1, 32, 144, 320)
                print(f"✅ Mask proto shape: {mask_proto.shape}")
            else:
                print("⚠️ Model không có mask proto output (chỉ có bbox detection)")
            
            if output is None:
                print("⚠️ Output is None")
                return detections, mask_proto
            
            # Reshape: (1, 48, 8400) → (48, 8400)
            if len(output.shape) == 3:
                output = output[0]
            
            # Split: [4 boxes | 12 classes | 32 masks]
            boxes = output[:4, :]           # (4, 8400)
            class_scores = output[4:4+self.num_classes, :]  # (12, 8400)
            mask_coeffs = output[4+self.num_classes:, :]  # (32, 8400)
            
            # YOLOv8/v11 format: boxes không cần sigmoid, class_scores cần sigmoid
            class_scores = self.sigmoid(class_scores)
            
            # Lấy class tốt nhất cho mỗi box
            max_scores = np.max(class_scores, axis=0)  # (8400,)
            max_classes = np.argmax(class_scores, axis=0)  # (8400,)
            
            # DEBUG: Kiểm tra các detections có confidence cao (>0.3)
            debug_dets = []
            for i in range(len(max_scores)):
                if max_scores[i] > 0.3:
                    debug_dets.append((max_classes[i], max_scores[i]))
            
            if len(debug_dets) > 0:
                print(f"\n🔍 DEBUG: Found {len(debug_dets)} detections with conf > 0.3:")
                # Group by class
                class_counts = {}
                for cls, conf in debug_dets:
                    if cls not in class_counts:
                        class_counts[cls] = []
                    class_counts[cls].append(conf)
                
                for cls in sorted(class_counts.keys()):
                    confs = class_counts[cls]
                    print(f"   Class {cls}: {len(confs)} detections (max conf: {max(confs):.3f})")
                
                # Kiểm tra xem có traffic light classes không
                tl_found = any(cls in self.TRAFFIC_LIGHT_CLASSES for cls, _ in debug_dets)
                if not tl_found:
                    print(f"   ⚠️ NO TRAFFIC LIGHT CLASSES (1, 6, 11) FOUND!")
                    print(f"   Traffic light classes: {list(self.TRAFFIC_LIGHT_CLASSES.keys())}")
            
            # Filter by confidence và chỉ lấy traffic light classes
            before_filter_count = 0
            after_size_filter_count = 0
            
            for i in range(len(max_scores)):
                conf = max_scores[i]
                cls = max_classes[i]
                
                # Chỉ lấy traffic light classes và confidence > threshold
                if cls not in self.TRAFFIC_LIGHT_CLASSES or conf < self.conf_threshold:
                    continue
                
                before_filter_count += 1
                
                # Get box (cx, cy, w, h) - ĐÃ Ở PIXEL COORDINATES (không phải normalized)
                cx, cy, w, h = boxes[:, i]
                
                # Boxes đã là pixel coordinates, không cần nhân thêm
                x1 = cx - w / 2
                y1 = cy - h / 2
                w_px = w
                h_px = h
                
                # DEBUG: Print first few boxes
                if before_filter_count <= 3:
                    print(f"  Box {before_filter_count}: cls={cls}, conf={conf:.3f}, size={w_px:.1f}x{h_px:.1f}, aspect={w_px/max(h_px,1):.2f}")
                
                # Filter small boxes (phải đủ lớn để là đèn giao thông)
                if w_px < 20 or h_px < 20:
                    continue
                
                # Filter quá lớn (không phải đèn đơn lẻ)
                if w_px > 300 or h_px > 400:
                    continue
                
                # Filter aspect ratio bất thường
                aspect_ratio = w_px / max(h_px, 1)
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    continue
                
                after_size_filter_count += 1
                
                # Clip to image bounds
                x1 = max(0, min(x1, self.target_w))
                y1 = max(0, min(y1, self.target_h))
                x2 = min(x1 + w_px, self.target_w)
                y2 = min(y1 + h_px, self.target_h)
                
                detections.append({
                    'class_id': cls,
                    'class_name': self.TRAFFIC_LIGHT_CLASSES[cls],
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'mask_coeff': mask_coeffs[:, i]  # Mask coefficients (32,)
                })
            
            if before_filter_count > 0:
                print(f"  📊 Traffic light detections: {before_filter_count} → after size/aspect filter: {after_size_filter_count}")
        
        except Exception as e:
            print(f"⚠️ Postprocess error: {e}")
            import traceback
            traceback.print_exc()
            return detections, None
        
        # Chỉ giữ lại 1 detection duy nhất:
        # 1. Tìm class có confidence cao nhất
        # 2. Giữ detection có confidence cao nhất trong class đó
        if len(detections) > 0:
            # Group by class và tìm max confidence per class
            class_best = {}
            for det in detections:
                cls = det['class_id']
                if cls not in class_best or det['confidence'] > class_best[cls]['confidence']:
                    class_best[cls] = det
            
            # Tìm class có confidence cao nhất
            best_det = max(class_best.values(), key=lambda x: x['confidence'])
            detections = [best_det]
            
            print(f"  ✅ Selected: {best_det['class_name']} (conf={best_det['confidence']:.3f})")
        
        return detections, mask_proto
    
    def nms(self, detections, iou_threshold=0.5):
        """
        Non-Maximum Suppression để loại bỏ overlapping boxes
        """
        if len(detections) == 0:
            return []
        
        # Group by class
        classes = {}
        for det in detections:
            cls = det['class_id']
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(det)
        
        # Apply NMS per class
        result = []
        for cls, dets in classes.items():
            # Sort by confidence
            dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while len(dets) > 0:
                # Keep highest confidence
                best = dets[0]
                keep.append(best)
                dets = dets[1:]
                
                # Remove overlapping boxes
                filtered = []
                for det in dets:
                    iou = self.compute_iou(best['bbox'], det['bbox'])
                    if iou < iou_threshold:
                        filtered.append(det)
                dets = filtered
            
            result.extend(keep)
        
        return result
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes (x, y, w, h)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to x1y1x2y2
        b1_x1, b1_y1 = x1, y1
        b1_x2, b1_y2 = x1 + w1, y1 + h1
        b2_x1, b2_y1 = x2, y2
        b2_x2, b2_y2 = x2 + w2, y2 + h2
        
        # Intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union
        b1_area = w1 * h1
        b2_area = w2 * h2
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / max(union_area, 1e-6)
    
    def draw_masks(self, frame, detections, mask_proto, y_offset):
        """
        Vẽ segmentation masks lên frame với transparency
        
        Args:
            frame: Full frame (480×640×3)
            detections: List of detections với mask_coeff
            mask_proto: Mask prototypes (1, 32, 144, 320) hoặc (32, 144, 320)
            y_offset: Y offset do crop (96)
        """
        if mask_proto is None or len(detections) == 0:
            return frame
        
        try:
            # Reshape mask proto nếu cần
            if len(mask_proto.shape) == 4:
                mask_proto = mask_proto[0]  # (32, 144, 320)
            
            proto_h, proto_w = mask_proto.shape[1], mask_proto.shape[2]
            
            # Tạo overlay cho masks
            mask_overlay = np.zeros_like(frame)
            
            for det in detections:
                if 'mask_coeff' not in det:
                    continue
                
                # Lấy bbox trong cropped space (288×640)
                x, y, w, h = det['bbox']
                
                # Lấy mask coefficients (32,)
                mask_coeff = det['mask_coeff']
                
                # Tính mask = sigmoid(coeffs @ proto)
                # mask_coeff: (32,), mask_proto: (32, 144, 320)
                mask = np.sum(mask_coeff[:, None, None] * mask_proto, axis=0)  # (144, 320)
                mask = self.sigmoid(mask)
                
                # Resize mask về kích thước target (288×640)
                mask_resized = cv2.resize(mask, (self.target_w, self.target_h))
                
                # Threshold để tạo binary mask
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # CHỈ GIỮ MASK TRONG BBOX - crop mask theo bbox
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(self.target_w, x + w), min(self.target_h, y + h)
                
                # Tạo mask chỉ trong bbox
                bbox_mask = np.zeros_like(mask_binary)
                bbox_mask[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
                
                # Tạo mask cho full frame (480×640) - chỉ vùng bbox
                full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                full_mask[y_offset:y_offset+self.target_h, :] = bbox_mask
                
                # Lấy màu theo class
                color = self.CLASS_COLORS[det['class_id']]
                
                # Vẽ mask lên overlay
                mask_overlay[full_mask == 1] = color
            
            # Blend overlay với frame gốc (alpha = 0.4)
            frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.4, 0)
            
        except Exception as e:
            print(f"⚠️ Draw masks error: {e}")
            import traceback
            traceback.print_exc()
        
        return frame
    
    def draw_detections(self, frame, detections, fps=0):
        """
        Vẽ bounding box và label lên frame
        
        Args:
            frame: BGR image (480×640)
            detections: List of detection dicts
            fps: Current FPS
        """
        # --- Vẽ 4 đường crop ---
        h, w = frame.shape[:2]
        crop_height = 288
        # Center crop
        start_y = (h - crop_height) // 2
        end_y = start_y + crop_height
        center_color = (0, 255, 255)  # vàng
        cv2.line(frame, (0, start_y), (w, start_y), center_color, 2)
        cv2.line(frame, (0, end_y), (w, end_y), center_color, 2)
        # Offset crop
        offset = 20
        crop_center = (start_y + end_y) // 2
        sec_center = crop_center - offset
        sec_start = sec_center - crop_height // 2
        sec_end = sec_center + crop_height // 2
        sec_color = (255, 0, 255)  # tím
        cv2.line(frame, (0, sec_start), (w, sec_start), sec_color, 2)
        cv2.line(frame, (0, sec_end), (w, sec_end), sec_color, 2)
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        h_frame = frame.shape[0]  # 480
        
        # Draw detections
        for det in detections:
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            x, y, w, h = det['bbox']
            
            # Tính diện tích
            area = w * h
            
            # Kiểm tra diện tích có trong vùng xử lý không
            in_processing_area = self.AREA_MIN <= area <= self.AREA_MAX
            
            if in_processing_area:
                # TRONG VÙNG XỬ LÝ (22k-45k) - Vẽ 2 khung chồng lên nhau
                
                # Tính offset chiều cao
                if area < 40000:
                    # Diện tích < 40k: offset cố định
                    h_offset = self.PURPLE_OFFSET_SMALL
                else:
                    # Diện tích >= 40k: offset max (đến sát biên trên)
                    # Tính khoảng cách từ top của box đến biên trên frame
                    space_to_top = y
                    h_offset = min(space_to_top, self.PURPLE_OFFSET_LARGE)
                
                # Tạo box màu tím với offset chiều cao
                # Box gốc: y -> y+h
                # Box offset: (y - h_offset) -> y+h (chiều cao tăng lên h_offset)
                y_purple = max(0, y - h_offset)  # Dịch lên h_offset
                h_purple = (y + h) - y_purple     # Chiều cao từ y_purple đến y+h
                
                # Đảm bảo không vượt biên dưới
                if y_purple + h_purple > h_frame:
                    h_purple = h_frame - y_purple
                
                print(f"  🟣 PROCESSING: {class_name} | Area: {area:,} px²")
                print(f"     Class box: ({x},{y})→({x+w},{y+h}) | Purple box: ({x},{y_purple})→({x+w},{y_purple+h_purple}) | Offset: +{h_offset}px")
                
                # 1. Vẽ KHUNG CLASS MÀU GỐC trước (nền)
                class_color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), class_color, 3)
                
                # 2. Vẽ KHUNG MÀU TÍM sau (chồng lên)
                cv2.rectangle(frame, (x, y_purple), (x+w, y_purple+h_purple), self.PURPLE_COLOR, 3)
                
                # Label cho khung tím
                label = f"{class_name} {confidence:.2f} [{area/1000:.1f}k] +{h_offset}px"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y_purple - label_h - 10), (x + label_w + 10, y_purple), self.PURPLE_COLOR, -1)
                cv2.putText(frame, label, (x + 5, y_purple - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            else:
                # NGOÀI VÙNG XỬ LÝ - Box màu cam, kích thước raw
                box_color = self.ORANGE_COLOR
                
                print(f"  🟠 OUT OF RANGE: {class_name} | Area: {area:,} px² | Box: ({x},{y})→({x+w},{y+h})")
                
                # Vẽ box màu cam (kích thước raw)
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
                
                # Label
                label = f"{class_name} {confidence:.2f} [{area/1000:.1f}k] OUT"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), box_color, -1)
                cv2.putText(frame, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection count
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def detect(self, frame):
        """
        Phát hiện đèn giao thông trong frame
        
        Args:
            frame: BGR image from camera (480×640×3)
            
        Returns:
            (processed_frame, detections, fps)
            processed_frame: Frame gốc 480×640 với detections được vẽ
        """
        t_start = time.time()
        
        # Preprocess
        img = self.preprocess(frame)
        
        # Inference
        outputs = self.rknn.inference(inputs=[img])
        
        # Postprocess
        detections, mask_proto = self.postprocess(outputs)
        
        # Calculate FPS
        elapsed = time.time() - t_start
        fps = 1.0 / max(elapsed, 1e-6)
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Vẽ detections lên frame gốc (480×640)
        # Điều chỉnh bbox từ cropped space (288×640) về full frame (480×640)
        h, w = frame.shape[:2]
        crop_h = self.target_h
        start_y = (h - crop_h) // 2  # offset Y do crop
        
        display_frame = frame.copy()
        
        # Vẽ segmentation masks trước (overlay)
        if mask_proto is not None and len(detections) > 0:
            display_frame = self.draw_masks(display_frame, detections, mask_proto, start_y)
        
        # Adjust detections to full frame coordinates
        adjusted_detections = []
        for det in detections:
            x, y, w_box, h_box = det['bbox']
            # Dịch Y coordinate về full frame
            y_full = y + start_y
            adjusted_detections.append({
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': (x, y_full, w_box, h_box)
            })
        
        display_frame = self.draw_detections(display_frame, adjusted_detections, avg_fps)
        
        return display_frame, detections, avg_fps


def create_video_writer(output_dir, width, height, fps, max_videos=5):
    """
    Create video writer with auto-incrementing filename.
    Removes old videos to keep only max_videos.
    
    Args:
        output_dir: Directory to save video
        width: Video width
        height: Video height
        fps: Video FPS
        max_videos: Maximum number of videos to keep
        
    Returns:
        tuple: (cv2.VideoWriter, video_path) or (None, None) if failed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")
    
    # Find next video number
    pattern = os.path.join(output_dir, "traffic_light_model_*.mp4")
    existing = glob.glob(pattern)
    
    max_num = 0
    for path in existing:
        basename = os.path.basename(path)
        # traffic_light_model_001.mp4 → 001
        try:
            # Lấy phần số từ tên file
            num_str = basename.replace("traffic_light_model_", "").replace(".mp4", "")
            # Nếu có thêm ký tự khác (underscore), chỉ lấy phần số đầu tiên
            if "_" in num_str:
                num_str = num_str.split("_")[0]
            num = int(num_str)
            max_num = max(max_num, num)
        except:
            pass
    
    next_num = max_num + 1
    # Chỉ dùng số thứ tự, không dùng timestamp - dùng .mp4
    video_path = os.path.join(output_dir, f"traffic_light_model_{next_num:03d}.mp4")
    
    print(f"  Next video: {os.path.basename(video_path)}")
    
    # Remove old videos (keep only max_videos - 1)
    existing_sorted = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    keep = max(0, max_videos - 1)
    for old_path in existing_sorted[keep:]:
        try:
            os.remove(old_path)
            print(f"  Removed old video: {os.path.basename(old_path)}")
        except OSError as e:
            print(f"  WARNING: Failed to remove {old_path}: {e}")
    
    # Create video writer - thử nhiều codec
    fourcc_list = [
        ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H264 - tốt nhất cho ARM
        ('X264', cv2.VideoWriter_fourcc(*'X264')),  # Variant của H264
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Xvid
    ]
    
    writer = None
    for codec_name, fourcc in fourcc_list:
        print(f"  Trying codec: {codec_name}...")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if writer.isOpened():
            print(f"  ✅ Codec {codec_name} works!")
            print(f"  Format: {width}x{height} @ {fps} FPS")
            return writer, video_path
        else:
            writer.release()
            print(f"  ❌ Codec {codec_name} failed")
    
    print(f"ERROR: All codecs failed for video writer: {video_path}")
    return None, None


def draw_recording_indicator(frame, frames_recorded):
    """Draw recording indicator on frame."""
    h, w = frame.shape[:2]
    
    # Red circle (REC indicator)
    cv2.circle(frame, (w - 40, 30), 10, (0, 0, 255), -1)
    
    # REC text
    cv2.putText(frame, "REC", (w - 100, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Frame count
    cv2.putText(frame, f"Frames: {frames_recorded}", (w - 150, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main():
    """Main test program"""
    print("=" * 70)
    print("🚦 TRAFFIC LIGHT DETECTION TEST - RKNN NPU")
    print("=" * 70)
    
    # Configuration
    MODEL_PATH = "/home/orangepi/Desktop/Projects/CDS_UTE_2025/Danh/models/model_seg_v2/best_fix_1712_11nano_fp16.rknn"
    CAMERA_ID = 0
    # Đường dẫn tuyệt đối để tránh lỗi
    VIDEO_OUTPUT_DIR = "/home/orangepi/Desktop/Projects/CDS_UTE_2025/Danh/analysis_traffic_leds/traffic_light/video_records"
    VIDEO_FPS = 10.0  # FPS cho video output
    MAX_VIDEOS = 5    # Giữ tối đa 5 video
    
    # Initialize detector
    detector = TrafficLightDetector(MODEL_PATH, target_size=(288, 640))
    
    if not detector.load_model():
        print("❌ Failed to load model, exiting...")
        return 1
    
    # Open camera
    print("📷 Opening camera...")
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        detector.release()
        return 1
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✅ Camera opened")
    
    # Initialize video recording
    print("\n" + "=" * 70)
    print("📹 Initializing video recording...")
    print("=" * 70)
    video_writer, video_path = create_video_writer(
        VIDEO_OUTPUT_DIR, 640, 480, VIDEO_FPS, MAX_VIDEOS
    )
    
    if video_writer is None:
        print("ERROR: Failed to initialize video recording")
        cap.release()
        detector.release()
        return 1
    
    is_recording = True
    frames_recorded = 0
    print("✅ Video recording started automatically!")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Controls:")
    print("  'q' - Quit (video will be saved automatically)")
    print("  's' - Save current frame as image")
    print("=" * 70 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Cannot read frame")
                continue
            
            frame_count += 1
            
            # Detect traffic lights
            display_frame, detections, fps = detector.detect(frame)
            
            # Verify frame is valid before writing
            if display_frame is None or display_frame.size == 0:
                print(f"⚠️ Frame {frame_count}: Invalid display_frame")
                continue
            
            # Draw recording indicator
            if is_recording:
                draw_recording_indicator(display_frame, frames_recorded)
            
            # Write to video if recording
            if is_recording and video_writer is not None:
                # Debug: check frame properties
                if frames_recorded == 0:
                    print(f"  First frame: shape={display_frame.shape}, dtype={display_frame.dtype}")
                
                video_writer.write(display_frame)
                frames_recorded += 1
                
                # Progress indicator every 50 frames
                if frames_recorded % 50 == 0:
                    print(f"  📹 Recorded {frames_recorded} frames...")
            
            # Print detections
            if detections and frame_count % 30 == 0:
                print(f"\n📊 Frame {frame_count} | Recorded: {frames_recorded}")
                for det in detections:
                    print(f"   {det['class_name']}: {det['confidence']:.2%} at {det['bbox']}")
            
            # Show frame
            cv2.imshow("Traffic Light Detection (press 'q' to quit)", display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            
            elif key == ord('s'):
                filename = f"traffic_light_detection_{frame_count:06d}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"💾 Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user (Ctrl+C)")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        
        # Stop video recording
        if is_recording and video_writer is not None:
            print(f"  Stopping video recording (recorded {frames_recorded} frames)...")
            
            # Force flush and release
            video_writer.release()
            
            # Check if file exists and has content
            if video_path and os.path.exists(video_path):
                size_bytes = os.path.getsize(video_path)
                size_mb = size_bytes / (1024 * 1024)
                
                print(f"  ✅ Video saved: {os.path.basename(video_path)}")
                print(f"     Path: {video_path}")
                print(f"     Frames: {frames_recorded}")
                print(f"     Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
                
                if size_bytes < 10000:  # Less than 10KB
                    print(f"  ⚠️ WARNING: Video file is very small ({size_bytes} bytes)")
                    print(f"     This might indicate recording failed")
            else:
                print(f"  ❌ Video file not found: {video_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        print("\n" + "=" * 70)
        print(f"📊 Final Stats:")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Frames recorded to video: {frames_recorded}")
        if frames_recorded > 0:
            print(f"   Recording rate: {frames_recorded/frame_count*100:.1f}%")
        print("=" * 70)
        print("✅ Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())

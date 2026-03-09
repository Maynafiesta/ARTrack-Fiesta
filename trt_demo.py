import os
import sys
import argparse
import time
from collections import deque
import cv2 as cv
import numpy as np
import torch

from trt_wrapper import TRTWrapper
from lib.test.parameter.artrack_seq import parameters
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.train.data.processing_utils import sample_target, transform_image_to_crop

class TRTTracker:
    def __init__(self, model_name):
        # Strip both possible suffixes
        base_name = model_name.replace("_fp32", "").replace("_fp16", "")
        self.params = parameters(base_name)
        self.cfg = self.params.cfg
        self.bins = self.cfg.MODEL.BINS
        self.preprocessor = Preprocessor()
        self.save_all = 7
        self.store_result = None
        self.template_tensor = None
        
        # Load TensorRT Engine (from Weights folder)
        engine_path = f"Weights/TensorRT/{model_name}.engine"
        if not os.path.exists(engine_path):
            print(f"Error: {engine_path} not found! Please build it first.")
            sys.exit(1)
            
        print(f"Initializing TensorRT Engine from {engine_path}...")
        self.engine = TRTWrapper(engine_path)

    def initialize(self, image, init_bbox):
        self.state = init_bbox
        self.store_result = [init_bbox.copy() for _ in range(self.save_all)]
        
        # Extract template
        z_patch_arr, _, z_amask_arr = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.template_tensor = template.tensors.cpu().numpy() # Shape (1, 3, Template_Size, Template_Size)
        
    def track(self, image):
        H, W, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
                                                                
        seqs_out = None
        for i in range(len(self.store_result)):
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)
                
        seqs_out = seqs_out.unsqueeze(0).long().cpu().numpy() # Shape (1, 28)
        
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search_tensor = search.tensors.cpu().numpy() # Shape (1, 3, H, W)

        feed_dict = {
            'template': self.template_tensor,
            'search': search_tensor,
            'seq_input': seqs_out
        }
        trt_out = self.engine.infer(feed_dict)
        pred_boxes_raw = trt_out['output'] 
        
        pred_boxes_tensor = torch.from_numpy(pred_boxes_raw)
        pred_boxes_tensor = pred_boxes_tensor[:, 0:4] / (self.bins - 1) - 0.5
        pred_boxes_tensor = pred_boxes_tensor.view(-1, 4).mean(dim=0)
        
        pred_new = pred_boxes_tensor.clone()
        pred_new[2] = pred_boxes_tensor[2] - pred_boxes_tensor[0]
        pred_new[3] = pred_boxes_tensor[3] - pred_boxes_tensor[1]
        pred_new[0] = pred_boxes_tensor[0] + pred_new[2] / 2
        pred_new[1] = pred_boxes_tensor[1] + pred_new[3] / 2
        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)
        
        if len(self.store_result) < self.save_all:
            self.store_result.append(self.state.copy())
        else:
            for i in range(self.save_all):
                if i != self.save_all - 1:
                    self.store_result[i] = self.store_result[i + 1]
                else:
                    self.store_result[i] = self.state.copy()
                    
        return self.state

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def main():
    parser = argparse.ArgumentParser(description='Run TensorRT Tracker')
    parser.add_argument('videofile', type=str, help='Path to a video file.')
    args = parser.parse_args()
    
    models = [
        'artrack_seq_large_384_full_fp32', 
        'artrack_seq_large_384_full_fp16',
        'artrack_seq_256_full_fp32',
        'artrack_seq_256_full_fp16'
    ]
    print("\n" + "="*50)
    print("TensorRT Engines (Available Models):")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}.engine")
    print("="*50)
    
    choice = int(input(f"Lütfen bir model numarası seçin (1-{len(models)}): "))
    model_name = models[choice-1]
    
    tracker = TRTTracker(model_name)
    display_name = f'TensorRT Tracking: {model_name}'
    
    cap = cv.VideoCapture(args.videofile)
    cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(display_name, 960, 720)
    
    success, frame = cap.read()
    if not success:
        print("Video açılamadı.")
        sys.exit(1)
        
    # İlk kareyi göster ve kullanıcıdan ROI seçmesini bekle
    while True:
        frame_disp = frame.copy()
        cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1.5, (0, 0, 0), 1)
        # Select ROI
        x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
        init_state = [x, y, w, h]
        tracker.initialize(frame, init_state)
        break
        
    # Warmup
    print("Warming up TRT Engine...")
    for _ in range(5):
        tracker.track(frame)
        
    print("Tracking started!")
    
    fps_history = deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_disp = frame.copy()
        
        t1 = time.time()
        state = tracker.track(frame)
        t2 = time.time()
        
        current_fps = 1.0 / (t2 - t1)
        fps_history.append(current_fps)
        avg_fps = sum(fps_history) / len(fps_history)
        
        state = [int(s) for s in state]
        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                     (0, 255, 0), 5)
                     
        font_color = (0, 255, 255) # Yellow for TRT visibility
        font_scale = 1.2
        thickness = 2
        cv.putText(frame_disp, 'TENSORRT TRACKING', (20, 40), cv.FONT_HERSHEY_COMPLEX_SMALL, font_scale,
                   font_color, thickness)
        cv.putText(frame_disp, f'Model: {model_name}.engine', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, font_scale,
                   font_color, thickness)
        cv.putText(frame_disp, f'FPS: {avg_fps:.1f}', (20, 120), cv.FONT_HERSHEY_COMPLEX_SMALL, font_scale,
                   font_color, thickness)
        cv.putText(frame_disp, 'r: reset | q: quit', (20, 160), cv.FONT_HERSHEY_COMPLEX_SMALL, font_scale,
                   font_color, thickness)
                   
        cv.imshow(display_name, frame_disp)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            ret, frame = cap.read()
            if not ret:
                break
            frame_disp = frame.copy()
            cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                       (0, 0, 0), 1)
            cv.imshow(display_name, frame_disp)
            x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            init_state = [x, y, w, h]
            tracker.initialize(frame, init_state)
            fps_history.clear()

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

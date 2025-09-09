#!/usr/bin/env python3
import cv2
import numpy as np
from collections import OrderedDict
import traceback

class ImageProcessor:
    def __init__(self):
        print("[LOG] Initializing ImageProcessor")
        self.init_function_definitions()
        
    def init_function_definitions(self):
        self.basic_adjustments = OrderedDict()
        self.filter_functions = OrderedDict()
        self.threshold_functions = OrderedDict()
        self.morphology_functions = OrderedDict()
        self.edge_detection_functions = OrderedDict()
        self.enhancement_functions = OrderedDict()
        self.color_functions = OrderedDict()
        self.artistic_functions = OrderedDict()
        
        self.basic_adjustments['brightness'] = {
            'brightness': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['contrast'] = {
            'contrast': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['exposure'] = {
            'exposure': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0}
        }
        self.basic_adjustments['highlights'] = {
            'highlights': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['shadows'] = {
            'shadows': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['whites'] = {
            'whites': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['blacks'] = {
            'blacks': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['vibrance'] = {
            'vibrance': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['saturation'] = {
            'saturation': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['temperature'] = {
            'temperature': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['tint'] = {
            'tint': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['clarity'] = {
            'clarity': {'type': 'slider', 'min': 0, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['dehaze'] = {
            'amount': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.basic_adjustments['vignette'] = {
            'amount': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'midpoint': {'type': 'slider', 'min': 0, 'max': 100, 'default': 50}
        }
        self.basic_adjustments['grain'] = {
            'amount': {'type': 'slider', 'min': 0, 'max': 100, 'default': 0},
            'size': {'type': 'slider', 'min': 1, 'max': 100, 'default': 25}
        }
        
        self.filter_functions['blur'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 99, 'default': 5}
        }
        self.filter_functions['gaussian_blur'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 99, 'default': 5},
            'sigma_x': {'type': 'slider', 'min': 0, 'max': 50, 'default': 0}
        }
        self.filter_functions['median_blur'] = {
            'kernel_size': {'type': 'slider', 'min': 3, 'max': 99, 'default': 5}
        }
        self.filter_functions['bilateral_filter'] = {
            'd': {'type': 'slider', 'min': 1, 'max': 30, 'default': 9},
            'sigma_color': {'type': 'slider', 'min': 1, 'max': 200, 'default': 75},
            'sigma_space': {'type': 'slider', 'min': 1, 'max': 200, 'default': 75}
        }
        self.filter_functions['box_filter'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 99, 'default': 5},
            'normalize': {'type': 'checkbox', 'default': True}
        }
        self.filter_functions['motion_blur'] = {
            'size': {'type': 'slider', 'min': 1, 'max': 100, 'default': 15},
            'angle': {'type': 'slider', 'min': 0, 'max': 360, 'default': 0}
        }
        self.filter_functions['sharpen'] = {
            'amount': {'type': 'slider', 'min': 0, 'max': 300, 'default': 100}
        }
        self.filter_functions['unsharp_mask'] = {
            'radius': {'type': 'slider', 'min': 0, 'max': 100, 'default': 1},
            'amount': {'type': 'slider', 'min': 0, 'max': 300, 'default': 100},
            'threshold': {'type': 'slider', 'min': 0, 'max': 255, 'default': 0}
        }
        
        self.threshold_functions['threshold_binary'] = {
            'threshold': {'type': 'slider', 'min': 0, 'max': 255, 'default': 127},
            'max_value': {'type': 'slider', 'min': 0, 'max': 255, 'default': 255}
        }
        self.threshold_functions['threshold_binary_inv'] = {
            'threshold': {'type': 'slider', 'min': 0, 'max': 255, 'default': 127},
            'max_value': {'type': 'slider', 'min': 0, 'max': 255, 'default': 255}
        }
        self.threshold_functions['threshold_trunc'] = {
            'threshold': {'type': 'slider', 'min': 0, 'max': 255, 'default': 127}
        }
        self.threshold_functions['threshold_tozero'] = {
            'threshold': {'type': 'slider', 'min': 0, 'max': 255, 'default': 127}
        }
        self.threshold_functions['threshold_tozero_inv'] = {
            'threshold': {'type': 'slider', 'min': 0, 'max': 255, 'default': 127}
        }
        self.threshold_functions['threshold_otsu'] = {
            'blur_size': {'type': 'slider', 'min': 0, 'max': 21, 'default': 5}
        }
        self.threshold_functions['adaptive_threshold_mean'] = {
            'max_value': {'type': 'slider', 'min': 0, 'max': 255, 'default': 255},
            'block_size': {'type': 'slider', 'min': 3, 'max': 99, 'default': 11},
            'c': {'type': 'slider', 'min': -50, 'max': 50, 'default': 2}
        }
        self.threshold_functions['adaptive_threshold_gaussian'] = {
            'max_value': {'type': 'slider', 'min': 0, 'max': 255, 'default': 255},
            'block_size': {'type': 'slider', 'min': 3, 'max': 99, 'default': 11},
            'c': {'type': 'slider', 'min': -50, 'max': 50, 'default': 2}
        }
        
        self.morphology_functions['erode'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 3},
            'iterations': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        self.morphology_functions['dilate'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 3},
            'iterations': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        self.morphology_functions['morphology_open'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 3},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        self.morphology_functions['morphology_close'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 3},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        self.morphology_functions['morphology_gradient'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 3},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        self.morphology_functions['morphology_tophat'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 9},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        self.morphology_functions['morphology_blackhat'] = {
            'kernel_size': {'type': 'slider', 'min': 1, 'max': 21, 'default': 9},
            'kernel_shape': {'type': 'combo', 'values': ['RECT', 'CROSS', 'ELLIPSE'], 'default': 'RECT'}
        }
        
        self.edge_detection_functions['canny'] = {
            'threshold1': {'type': 'slider', 'min': 0, 'max': 500, 'default': 100},
            'threshold2': {'type': 'slider', 'min': 0, 'max': 500, 'default': 200},
            'aperture_size': {'type': 'combo', 'values': [3, 5, 7], 'default': 3}
        }
        self.edge_detection_functions['sobel_x'] = {
            'kernel_size': {'type': 'combo', 'values': [1, 3, 5, 7], 'default': 3},
            'scale': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'delta': {'type': 'slider', 'min': 0, 'max': 255, 'default': 0}
        }
        self.edge_detection_functions['sobel_y'] = {
            'kernel_size': {'type': 'combo', 'values': [1, 3, 5, 7], 'default': 3},
            'scale': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'delta': {'type': 'slider', 'min': 0, 'max': 255, 'default': 0}
        }
        self.edge_detection_functions['laplacian'] = {
            'kernel_size': {'type': 'combo', 'values': [1, 3, 5, 7], 'default': 3},
            'scale': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'delta': {'type': 'slider', 'min': 0, 'max': 255, 'default': 0}
        }
        self.edge_detection_functions['scharr_x'] = {
            'scale': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'delta': {'type': 'slider', 'min': 0, 'max': 255, 'default': 0}
        }
        self.edge_detection_functions['scharr_y'] = {
            'scale': {'type': 'slider', 'min': 1, 'max': 10, 'default': 1},
            'delta': {'type': 'slider', 'min': 0, 'max': 255, 'default': 0}
        }
        
        self.enhancement_functions['histogram_equalization'] = {
            'clip_limit': {'type': 'slider', 'min': 1, 'max': 100, 'default': 40},
            'grid_size': {'type': 'slider', 'min': 1, 'max': 16, 'default': 8}
        }
        self.enhancement_functions['gamma_correction'] = {
            'gamma': {'type': 'slider', 'min': 10, 'max': 300, 'default': 100}
        }
        self.enhancement_functions['denoise'] = {
            'h': {'type': 'slider', 'min': 1, 'max': 30, 'default': 10},
            'template_window_size': {'type': 'slider', 'min': 3, 'max': 21, 'default': 7},
            'search_window_size': {'type': 'slider', 'min': 5, 'max': 45, 'default': 21}
        }
        self.enhancement_functions['detail_enhance'] = {
            'sigma_s': {'type': 'slider', 'min': 1, 'max': 200, 'default': 10},
            'sigma_r': {'type': 'slider', 'min': 0, 'max': 100, 'default': 15}
        }
        self.enhancement_functions['edge_preserve_filter'] = {
            'flags': {'type': 'combo', 'values': ['RECURS_FILTER', 'NORMCONV_FILTER'], 'default': 'RECURS_FILTER'},
            'sigma_s': {'type': 'slider', 'min': 1, 'max': 200, 'default': 60},
            'sigma_r': {'type': 'slider', 'min': 0, 'max': 100, 'default': 40}
        }
        
        self.color_functions['hsv_adjustment'] = {
            'hue_shift': {'type': 'slider', 'min': -180, 'max': 180, 'default': 0},
            'saturation_scale': {'type': 'slider', 'min': 0, 'max': 200, 'default': 100},
            'value_scale': {'type': 'slider', 'min': 0, 'max': 200, 'default': 100}
        }
        self.color_functions['lab_adjustment'] = {
            'l_scale': {'type': 'slider', 'min': 0, 'max': 200, 'default': 100},
            'a_shift': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'b_shift': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.color_functions['channel_mixer'] = {
            'red_red': {'type': 'slider', 'min': -200, 'max': 200, 'default': 100},
            'red_green': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0},
            'red_blue': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0},
            'green_red': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0},
            'green_green': {'type': 'slider', 'min': -200, 'max': 200, 'default': 100},
            'green_blue': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0},
            'blue_red': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0},
            'blue_green': {'type': 'slider', 'min': -200, 'max': 200, 'default': 0},
            'blue_blue': {'type': 'slider', 'min': -200, 'max': 200, 'default': 100}
        }
        self.color_functions['color_balance'] = {
            'shadows_cyan_red': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'shadows_magenta_green': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'shadows_yellow_blue': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'midtones_cyan_red': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'midtones_magenta_green': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'midtones_yellow_blue': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'highlights_cyan_red': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'highlights_magenta_green': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'highlights_yellow_blue': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        self.color_functions['selective_color'] = {
            'color': {'type': 'combo', 'values': ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas'], 'default': 'reds'},
            'cyan': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'magenta': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'yellow': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0},
            'black': {'type': 'slider', 'min': -100, 'max': 100, 'default': 0}
        }
        
        self.artistic_functions['pencil_sketch'] = {
            'sigma_s': {'type': 'slider', 'min': 1, 'max': 200, 'default': 60},
            'sigma_r': {'type': 'slider', 'min': 0, 'max': 100, 'default': 7},
            'shade_factor': {'type': 'slider', 'min': 0, 'max': 100, 'default': 2}
        }
        self.artistic_functions['stylization'] = {
            'sigma_s': {'type': 'slider', 'min': 1, 'max': 200, 'default': 60},
            'sigma_r': {'type': 'slider', 'min': 0, 'max': 100, 'default': 45}
        }
        self.artistic_functions['oil_painting'] = {
            'size': {'type': 'slider', 'min': 1, 'max': 10, 'default': 4},
            'dyn_ratio': {'type': 'slider', 'min': 1, 'max': 20, 'default': 1}
        }
        self.artistic_functions['cartoon'] = {
            'num_down': {'type': 'slider', 'min': 1, 'max': 4, 'default': 2},
            'num_bilateral': {'type': 'slider', 'min': 1, 'max': 10, 'default': 7}
        }
        
    def get_basic_adjustments(self):
        return self.basic_adjustments
    
    def get_filter_functions(self):
        return self.filter_functions
    
    def get_threshold_functions(self):
        return self.threshold_functions
    
    def get_morphology_functions(self):
        return self.morphology_functions
    
    def get_edge_detection_functions(self):
        return self.edge_detection_functions
    
    def get_enhancement_functions(self):
        return self.enhancement_functions
    
    def get_color_functions(self):
        return self.color_functions
    
    def get_artistic_functions(self):
        return self.artistic_functions
    
    def apply_edit_line(self, image, edit_line):
        print(f"[LOG] Applying edit line with {len(edit_line)} operations")
        result = image.copy()
        
        for i, edit in enumerate(edit_line):
            func_name = edit['function']
            params = edit['params']
            print(f"[LOG] Applying {i+1}/{len(edit_line)}: {func_name}")
            
            try:
                if func_name in self.basic_adjustments:
                    result = self.apply_basic_adjustment(result, func_name, params)
                elif func_name in self.filter_functions:
                    result = self.apply_filter(result, func_name, params)
                elif func_name in self.threshold_functions:
                    result = self.apply_threshold(result, func_name, params)
                elif func_name in self.morphology_functions:
                    result = self.apply_morphology(result, func_name, params)
                elif func_name in self.edge_detection_functions:
                    result = self.apply_edge_detection(result, func_name, params)
                elif func_name in self.enhancement_functions:
                    result = self.apply_enhancement(result, func_name, params)
                elif func_name in self.color_functions:
                    result = self.apply_color_function(result, func_name, params)
                elif func_name in self.artistic_functions:
                    result = self.apply_artistic_function(result, func_name, params)
            except Exception as e:
                print(f"[LOG] ERROR applying {func_name}: {str(e)}")
                traceback.print_exc()
                
        return result
    
    def apply_basic_adjustment(self, image, func_name, params):
        print(f"[LOG] Applying basic adjustment: {func_name}")
        
        if func_name == 'brightness':
            value = params.get('brightness', 0)
            return cv2.convertScaleAbs(image, alpha=1, beta=value)
            
        elif func_name == 'contrast':
            value = params.get('contrast', 0)
            alpha = (value + 100) / 100
            return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            
        elif func_name == 'exposure':
            value = params.get('exposure', 0)
            exposure_factor = np.power(2, value / 100)
            return cv2.convertScaleAbs(image, alpha=exposure_factor, beta=0)
            
        elif func_name == 'highlights':
            value = params.get('highlights', 0)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mask = cv2.inRange(l, 170, 255)
            adjustment = int(value * 0.5)
            l = cv2.add(l, adjustment, mask=mask)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'shadows':
            value = params.get('shadows', 0)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mask = cv2.inRange(l, 0, 85)
            adjustment = int(value * 0.5)
            l = cv2.add(l, adjustment, mask=mask)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'whites':
            value = params.get('whites', 0)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mask = cv2.inRange(l, 200, 255)
            adjustment = int(value * 0.3)
            l = cv2.add(l, adjustment, mask=mask)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'blacks':
            value = params.get('blacks', 0)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mask = cv2.inRange(l, 0, 55)
            adjustment = int(value * 0.3)
            l = cv2.add(l, adjustment, mask=mask)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'vibrance':
            value = params.get('vibrance', 0)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            low_sat_mask = s < 50
            adjustment = 1 + (value / 100)
            s[low_sat_mask] = s[low_sat_mask] * adjustment
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif func_name == 'saturation':
            value = params.get('saturation', 0)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            adjustment = 1 + (value / 100)
            s = s * adjustment
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif func_name == 'temperature':
            value = params.get('temperature', 0)
            result = image.copy().astype(np.float32)
            if value > 0:
                result[:,:,2] = result[:,:,2] * (1 + value/200)
            else:
                result[:,:,0] = result[:,:,0] * (1 - value/200)
            return np.clip(result, 0, 255).astype(np.uint8)
            
        elif func_name == 'tint':
            value = params.get('tint', 0)
            result = image.copy().astype(np.float32)
            if value > 0:
                result[:,:,1] = result[:,:,1] * (1 + value/200)
            else:
                result[:,:,1] = result[:,:,1] * (1 + value/200)
            return np.clip(result, 0, 255).astype(np.uint8)
            
        elif func_name == 'clarity':
            value = params.get('clarity', 0)
            if value == 0:
                return image
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_float = l.astype(np.float32)
            blurred = cv2.GaussianBlur(l_float, (31, 31), 10)
            detail = l_float - blurred
            l_enhanced = l_float + detail * (value / 50)
            l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
            lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'dehaze':
            amount = params.get('amount', 0)
            if amount == 0:
                return image
            dark_channel = self.get_dark_channel(image)
            atmospheric_light = self.get_atmospheric_light(image, dark_channel)
            transmission = self.get_transmission(image, atmospheric_light, amount/100)
            return self.recover_scene(image, transmission, atmospheric_light)
            
        elif func_name == 'vignette':
            amount = params.get('amount', 0)
            midpoint = params.get('midpoint', 50)
            if amount == 0:
                return image
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            mask = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    mask[y, x] = 1 - (dist / max_dist) * (1 - midpoint/100)
            mask = np.clip(mask, 0, 1)
            if amount < 0:
                mask = 1 - mask * (-amount/100)
            else:
                mask = mask ** (1 + amount/100)
            result = image.astype(np.float32)
            for i in range(3):
                result[:,:,i] = result[:,:,i] * mask
            return np.clip(result, 0, 255).astype(np.uint8)
            
        elif func_name == 'grain':
            amount = params.get('amount', 0)
            size = params.get('size', 25)
            if amount == 0:
                return image
            h, w = image.shape[:2]
            noise = np.random.normal(0, amount/10, (h, w, 3))
            if size > 1:
                kernel_size = int(size/10) * 2 + 1
                noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)
            result = image.astype(np.float32) + noise
            return np.clip(result, 0, 255).astype(np.uint8)
            
        return image
    
    def apply_filter(self, image, func_name, params):
        print(f"[LOG] Applying filter: {func_name}")
        
        if func_name == 'blur':
            kernel_size = params.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.blur(image, (kernel_size, kernel_size))
            
        elif func_name == 'gaussian_blur':
            kernel_size = params.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma_x = params.get('sigma_x', 0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)
            
        elif func_name == 'median_blur':
            kernel_size = params.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.medianBlur(image, kernel_size)
            
        elif func_name == 'bilateral_filter':
            d = params.get('d', 9)
            sigma_color = params.get('sigma_color', 75)
            sigma_space = params.get('sigma_space', 75)
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
        elif func_name == 'box_filter':
            kernel_size = params.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            normalize = params.get('normalize', True)
            return cv2.boxFilter(image, -1, (kernel_size, kernel_size), normalize=normalize)
            
        elif func_name == 'motion_blur':
            size = params.get('size', 15)
            angle = params.get('angle', 0)
            kernel = self.create_motion_blur_kernel(size, angle)
            return cv2.filter2D(image, -1, kernel)
            
        elif func_name == 'sharpen':
            amount = params.get('amount', 100)
            kernel = np.array([[-1, -1, -1], 
                               [-1, 9, -1], 
                               [-1, -1, -1]]) * (amount / 100)
            kernel[1, 1] = 8 + amount / 100
            return cv2.filter2D(image, -1, kernel)
            
        elif func_name == 'unsharp_mask':
            radius = params.get('radius', 1)
            amount = params.get('amount', 100)
            threshold = params.get('threshold', 0)
            kernel_size = int(radius * 2) * 2 + 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)
            sharpened = cv2.addWeighted(image, 1 + amount/100, blurred, -amount/100, 0)
            if threshold > 0:
                diff = cv2.absdiff(image, blurred)
                mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                sharpened = np.where(mask > 0, sharpened, image)
            return sharpened
            
        return image
    
    def apply_threshold(self, image, func_name, params):
        print(f"[LOG] Applying threshold: {func_name}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if func_name == 'threshold_binary':
            threshold = params.get('threshold', 127)
            max_value = params.get('max_value', 255)
            _, result = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
            
        elif func_name == 'threshold_binary_inv':
            threshold = params.get('threshold', 127)
            max_value = params.get('max_value', 255)
            _, result = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY_INV)
            
        elif func_name == 'threshold_trunc':
            threshold = params.get('threshold', 127)
            _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_TRUNC)
            
        elif func_name == 'threshold_tozero':
            threshold = params.get('threshold', 127)
            _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_TOZERO)
            
        elif func_name == 'threshold_tozero_inv':
            threshold = params.get('threshold', 127)
            _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_TOZERO_INV)
            
        elif func_name == 'threshold_otsu':
            blur_size = params.get('blur_size', 5)
            if blur_size > 0:
                if blur_size % 2 == 0:
                    blur_size += 1
                gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif func_name == 'adaptive_threshold_mean':
            max_value = params.get('max_value', 255)
            block_size = params.get('block_size', 11)
            if block_size % 2 == 0:
                block_size += 1
            c = params.get('c', 2)
            result = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
            
        elif func_name == 'adaptive_threshold_gaussian':
            max_value = params.get('max_value', 255)
            block_size = params.get('block_size', 11)
            if block_size % 2 == 0:
                block_size += 1
            c = params.get('c', 2)
            result = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
        else:
            result = gray
            
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    def apply_morphology(self, image, func_name, params):
        print(f"[LOG] Applying morphology: {func_name}")
        kernel_size = params.get('kernel_size', 3)
        kernel_shape = params.get('kernel_shape', 'RECT').upper()
        
        if kernel_shape == 'RECT':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == 'CROSS':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape == 'ELLIPSE':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if func_name == 'erode':
            iterations = params.get('iterations', 1)
            return cv2.erode(image, kernel, iterations=iterations)
            
        elif func_name == 'dilate':
            iterations = params.get('iterations', 1)
            return cv2.dilate(image, kernel, iterations=iterations)
            
        elif func_name == 'morphology_open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
        elif func_name == 'morphology_close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
        elif func_name == 'morphology_gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            
        elif func_name == 'morphology_tophat':
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            
        elif func_name == 'morphology_blackhat':
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            
        return image
    
    def apply_edge_detection(self, image, func_name, params):
        print(f"[LOG] Applying edge detection: {func_name}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if func_name == 'canny':
            threshold1 = params.get('threshold1', 100)
            threshold2 = params.get('threshold2', 200)
            aperture_size = int(params.get('aperture_size', 3))
            edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        elif func_name == 'sobel_x':
            kernel_size = int(params.get('kernel_size', 3))
            scale = params.get('scale', 1)
            delta = params.get('delta', 0)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
            sobel = np.absolute(sobel)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            
        elif func_name == 'sobel_y':
            kernel_size = int(params.get('kernel_size', 3))
            scale = params.get('scale', 1)
            delta = params.get('delta', 0)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
            sobel = np.absolute(sobel)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            
        elif func_name == 'laplacian':
            kernel_size = int(params.get('kernel_size', 3))
            scale = params.get('scale', 1)
            delta = params.get('delta', 0)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
            laplacian = np.absolute(laplacian)
            laplacian = np.uint8(np.clip(laplacian, 0, 255))
            return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
            
        elif func_name == 'scharr_x':
            scale = params.get('scale', 1)
            delta = params.get('delta', 0)
            scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
            scharr = np.absolute(scharr)
            scharr = np.uint8(np.clip(scharr, 0, 255))
            return cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
            
        elif func_name == 'scharr_y':
            scale = params.get('scale', 1)
            delta = params.get('delta', 0)
            scharr = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
            scharr = np.absolute(scharr)
            scharr = np.uint8(np.clip(scharr, 0, 255))
            return cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
            
        return image
    
    def apply_enhancement(self, image, func_name, params):
        print(f"[LOG] Applying enhancement: {func_name}")
        
        if func_name == 'histogram_equalization':
            clip_limit = params.get('clip_limit', 40) / 10
            grid_size = params.get('grid_size', 8)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'gamma_correction':
            gamma = params.get('gamma', 100) / 100
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
            
        elif func_name == 'denoise':
            h = params.get('h', 10)
            template_window_size = params.get('template_window_size', 7)
            if template_window_size % 2 == 0:
                template_window_size += 1
            search_window_size = params.get('search_window_size', 21)
            if search_window_size % 2 == 0:
                search_window_size += 1
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                                  template_window_size, search_window_size)
            
        elif func_name == 'detail_enhance':
            sigma_s = params.get('sigma_s', 10)
            sigma_r = params.get('sigma_r', 15) / 100
            return cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
            
        elif func_name == 'edge_preserve_filter':
            flags = params.get('flags', 'RECURS_FILTER')
            sigma_s = params.get('sigma_s', 60)
            sigma_r = params.get('sigma_r', 40) / 100
            if flags == 'RECURS_FILTER':
                return cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER, 
                                              sigma_s=sigma_s, sigma_r=sigma_r)
            else:
                return cv2.edgePreservingFilter(image, flags=cv2.NORMCONV_FILTER, 
                                              sigma_s=sigma_s, sigma_r=sigma_r)
        
        return image
    
    def apply_color_function(self, image, func_name, params):
        print(f"[LOG] Applying color function: {func_name}")
        
        if func_name == 'hsv_adjustment':
            hue_shift = params.get('hue_shift', 0)
            saturation_scale = params.get('saturation_scale', 100) / 100
            value_scale = params.get('value_scale', 100) / 100
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            
            h = (h + hue_shift) % 180
            s = s * saturation_scale
            v = v * value_scale
            
            s = np.clip(s, 0, 255)
            v = np.clip(v, 0, 255)
            
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif func_name == 'lab_adjustment':
            l_scale = params.get('l_scale', 100) / 100
            a_shift = params.get('a_shift', 0)
            b_shift = params.get('b_shift', 0)
            
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            
            l = l * l_scale
            a = a + a_shift
            b = b + b_shift
            
            l = np.clip(l, 0, 255)
            a = np.clip(a, 0, 255)
            b = np.clip(b, 0, 255)
            
            lab = cv2.merge([l, a, b]).astype(np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif func_name == 'channel_mixer':
            result = np.zeros_like(image, dtype=np.float32)
            
            result[:,:,2] = (image[:,:,2] * params.get('red_red', 100) / 100 +
                           image[:,:,1] * params.get('red_green', 0) / 100 +
                           image[:,:,0] * params.get('red_blue', 0) / 100)
            
            result[:,:,1] = (image[:,:,2] * params.get('green_red', 0) / 100 +
                           image[:,:,1] * params.get('green_green', 100) / 100 +
                           image[:,:,0] * params.get('green_blue', 0) / 100)
            
            result[:,:,0] = (image[:,:,2] * params.get('blue_red', 0) / 100 +
                           image[:,:,1] * params.get('blue_green', 0) / 100 +
                           image[:,:,0] * params.get('blue_blue', 100) / 100)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        elif func_name == 'color_balance':
            result = image.copy().astype(np.float32)
            
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            shadows_mask = l < 85
            midtones_mask = (l >= 85) & (l < 170)
            highlights_mask = l >= 170
            
            a[shadows_mask] += params.get('shadows_magenta_green', 0) * 0.5
            b[shadows_mask] += params.get('shadows_yellow_blue', 0) * 0.5
            
            a[midtones_mask] += params.get('midtones_magenta_green', 0) * 0.5
            b[midtones_mask] += params.get('midtones_yellow_blue', 0) * 0.5
            
            a[highlights_mask] += params.get('highlights_magenta_green', 0) * 0.5
            b[highlights_mask] += params.get('highlights_yellow_blue', 0) * 0.5
            
            lab = cv2.merge([l, np.clip(a, 0, 255), np.clip(b, 0, 255)])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            result[shadows_mask, 2] += params.get('shadows_cyan_red', 0) * 0.5
            result[midtones_mask, 2] += params.get('midtones_cyan_red', 0) * 0.5
            result[highlights_mask, 2] += params.get('highlights_cyan_red', 0) * 0.5
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        elif func_name == 'selective_color':
            color = params.get('color', 'reds')
            cyan_adj = params.get('cyan', 0) / 100
            magenta_adj = params.get('magenta', 0) / 100
            yellow_adj = params.get('yellow', 0) / 100
            black_adj = params.get('black', 0) / 100
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            if color == 'reds':
                mask = ((h < 10) | (h > 170)) & (s > 50)
            elif color == 'yellows':
                mask = (h >= 20) & (h < 40) & (s > 50)
            elif color == 'greens':
                mask = (h >= 40) & (h < 80) & (s > 50)
            elif color == 'cyans':
                mask = (h >= 80) & (h < 110) & (s > 50)
            elif color == 'blues':
                mask = (h >= 110) & (h < 130) & (s > 50)
            elif color == 'magentas':
                mask = (h >= 130) & (h < 170) & (s > 50)
            else:
                mask = np.zeros_like(h, dtype=bool)
            
            result = image.copy().astype(np.float32)
            
            if np.any(mask):
                result[mask, 2] *= (1 - cyan_adj)
                result[mask, 1] *= (1 - magenta_adj)
                result[mask, 0] *= (1 - yellow_adj)
                result[mask] *= (1 - black_adj)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        return image
    
    def apply_artistic_function(self, image, func_name, params):
        print(f"[LOG] Applying artistic function: {func_name}")
        
        if func_name == 'pencil_sketch':
            sigma_s = params.get('sigma_s', 60)
            sigma_r = params.get('sigma_r', 7) / 100
            shade_factor = params.get('shade_factor', 2) / 100
            
            gray, color = cv2.pencilSketch(image, sigma_s=sigma_s, sigma_r=sigma_r, 
                                          shade_factor=shade_factor)
            return color
            
        elif func_name == 'stylization':
            sigma_s = params.get('sigma_s', 60)
            sigma_r = params.get('sigma_r', 45) / 100
            return cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)
            
        elif func_name == 'oil_painting':
            size = params.get('size', 4)
            dyn_ratio = params.get('dyn_ratio', 1)
            return cv2.xphoto.oilPainting(image, size, dyn_ratio)
            
        elif func_name == 'cartoon':
            num_down = params.get('num_down', 2)
            num_bilateral = params.get('num_bilateral', 7)
            
            img_color = image
            for _ in range(num_down):
                img_color = cv2.pyrDown(img_color)
            
            for _ in range(num_bilateral):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                
            for _ in range(num_down):
                img_color = cv2.pyrUp(img_color)
                
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)
            
            img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 9, 10)
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
            
            h, w = image.shape[:2]
            img_color = cv2.resize(img_color, (w, h))
            
            return cv2.bitwise_and(img_color, img_edge)
            
        return image
    
    def create_motion_blur_kernel(self, size, angle):
        kernel = np.zeros((size, size))
        center = size // 2
        angle_rad = np.deg2rad(angle)
        
        for i in range(size):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
                
        return kernel / np.sum(kernel)
    
    def get_dark_channel(self, image, patch_size=15):
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def get_atmospheric_light(self, image, dark_channel, top_percent=0.001):
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape(-1, 3)
        num_pixels = len(flat_dark)
        num_brightest = max(int(num_pixels * top_percent), 1)
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        atmospheric_light = np.max(flat_image[indices], axis=0)
        return atmospheric_light
    
    def get_transmission(self, image, atmospheric_light, omega=0.95):
        normalized = image / atmospheric_light
        dark_channel = self.get_dark_channel(normalized)
        transmission = 1 - omega * dark_channel
        return transmission
    
    def recover_scene(self, image, transmission, atmospheric_light, t0=0.1):
        transmission = np.maximum(transmission, t0)
        recovered = np.empty_like(image)
        for i in range(3):
            recovered[:,:,i] = (image[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        return np.clip(recovered, 0, 255).astype(np.uint8)
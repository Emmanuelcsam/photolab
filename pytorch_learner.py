#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import json
import pickle
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ImageFeatureExtractor:
    def __init__(self):
        print("[LOG] Initializing ImageFeatureExtractor")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        self.fitted = False
        
    def extract_features(self, image_path):
        print(f"[LOG] Extracting features from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"[LOG] WARNING: Failed to load image: {image_path}")
            return None
            
        features = []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features.extend(self.get_color_statistics(img_rgb))
        features.extend(self.get_color_statistics(img_hsv))
        features.extend(self.get_color_statistics(img_lab))
        
        features.extend(self.get_histogram_features(img_gray))
        features.extend(self.get_histogram_features(img_hsv[:,:,0]))
        features.extend(self.get_histogram_features(img_hsv[:,:,1]))
        features.extend(self.get_histogram_features(img_hsv[:,:,2]))
        
        features.extend(self.get_texture_features(img_gray))
        
        features.extend(self.get_edge_features(img_gray))
        
        features.extend(self.get_quality_metrics(img))
        
        features.extend(self.get_composition_features(img))
        
        return np.array(features, dtype=np.float32)
    
    def get_color_statistics(self, img):
        features = []
        for i in range(img.shape[2]):
            channel = img[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 50),
                np.percentile(channel, 75),
                channel.flatten()[::1000].mean()
            ])
        return features
    
    def get_histogram_features(self, channel):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        features = []
        features.extend(hist[:16].tolist())
        
        features.append(np.sum(hist * np.arange(32)))
        features.append(np.sum(hist * (np.arange(32) ** 2)))
        features.append(-np.sum(hist * np.log(hist + 1e-7)))
        
        cumsum = np.cumsum(hist)
        features.extend([cumsum[7], cumsum[15], cumsum[23]])
        
        return features
    
    def get_texture_features(self, gray):
        features = []
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var())
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            gradient_magnitude.mean(),
            gradient_magnitude.std(),
            gradient_magnitude.max()
        ])
        
        glcm = self.compute_glcm(gray)
        features.extend(self.glcm_features(glcm))
        
        gabor_features = self.gabor_features(gray)
        features.extend(gabor_features)
        
        return features
    
    def compute_glcm(self, img, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        img_8bit = (img / 16).astype(np.uint8)
        h, w = img_8bit.shape
        glcm = np.zeros((16, 16))
        
        for d in distances[:1]:
            for angle in angles[:2]:
                dx = int(np.round(d * np.cos(angle)))
                dy = int(np.round(d * np.sin(angle)))
                
                for i in range(max(0, -dy), min(h, h - dy)):
                    for j in range(max(0, -dx), min(w, w - dx)):
                        glcm[img_8bit[i, j], img_8bit[i + dy, j + dx]] += 1
                        
        glcm = glcm / glcm.sum()
        return glcm
    
    def glcm_features(self, glcm):
        features = []
        
        i, j = np.mgrid[0:glcm.shape[0], 0:glcm.shape[1]]
        
        contrast = np.sum(glcm * (i - j)**2)
        features.append(contrast)
        
        homogeneity = np.sum(glcm / (1 + (i - j)**2))
        features.append(homogeneity)
        
        energy = np.sum(glcm**2)
        features.append(energy)
        
        correlation = np.sum(glcm * i * j) - np.sum(glcm * i) * np.sum(glcm * j)
        features.append(correlation)
        
        return features
    
    def gabor_features(self, img, frequencies=[0.1, 0.3]):
        features = []
        
        for freq in frequencies:
            for theta in [0, np.pi/4]:
                kernel = cv2.getGaborKernel((21, 21), 4, theta, 10/freq, 0.5, 0)
                filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
                features.extend([filtered.mean(), filtered.std()])
                
        return features
    
    def get_edge_features(self, gray):
        features = []
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.append(len(contours))
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features.extend([np.mean(areas), np.std(areas), np.max(areas)])
        else:
            features.extend([0, 0, 0])
            
        return features
    
    def get_quality_metrics(self, img):
        features = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        noise_level = self.estimate_noise(gray)
        features.append(noise_level)
        
        exposure_metric = self.calculate_exposure(img)
        features.extend(exposure_metric)
        
        return features
    
    def estimate_noise(self, gray):
        h, w = gray.shape
        crop = gray[h//4:3*h//4, w//4:3*w//4]
        return np.std(crop - cv2.GaussianBlur(crop, (5, 5), 0))
    
    def calculate_exposure(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        
        underexposed = np.sum(hist[:50])
        overexposed = np.sum(hist[205:])
        well_exposed = 1 - underexposed - overexposed
        
        return [underexposed, well_exposed, overexposed]
    
    def get_composition_features(self, img):
        features = []
        
        h, w = img.shape[:2]
        thirds_h, thirds_w = h // 3, w // 3
        
        regions = [
            img[:thirds_h, :thirds_w],
            img[:thirds_h, thirds_w:2*thirds_w],
            img[:thirds_h, 2*thirds_w:],
            img[thirds_h:2*thirds_h, :thirds_w],
            img[thirds_h:2*thirds_h, thirds_w:2*thirds_w],
            img[thirds_h:2*thirds_h, 2*thirds_w:],
            img[2*thirds_h:, :thirds_w],
            img[2*thirds_h:, thirds_w:2*thirds_w],
            img[2*thirds_h:, 2*thirds_w:]
        ]
        
        for region in regions:
            features.append(np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)))
            
        center_region = img[h//4:3*h//4, w//4:3*w//4]
        features.append(np.std(center_region))
        
        return features
    
    def fit_transform(self, features_list):
        print("[LOG] Fitting feature transformer")
        features_array = np.array(features_list)
        
        if len(features_array) < 2:
            print("[LOG] WARNING: Not enough samples for fitting")
            return features_array
            
        self.scaler.fit(features_array)
        scaled_features = self.scaler.transform(features_array)
        
        if len(features_array) < self.pca.n_components:
            self.pca.n_components = min(len(features_array) - 1, 20)
            
        self.pca.fit(scaled_features)
        self.fitted = True
        
        return self.pca.transform(scaled_features)
    
    def transform(self, features):
        if not self.fitted:
            print("[LOG] WARNING: Transformer not fitted, returning raw features")
            return features
            
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)

class EditSequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(EditSequenceEncoder, self).__init__()
        print(f"[LOG] Initializing EditSequenceEncoder with input_dim={input_dim}")
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.2
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        
        lstm_out, _ = self.lstm(x)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        pooled = torch.mean(attn_out, dim=1)
        
        output = self.output_projection(pooled)
        
        return output

class EditPredictor(nn.Module):
    def __init__(self, image_feature_dim, edit_encoding_dim, num_functions, max_params=20):
        super(EditPredictor, self).__init__()
        print(f"[LOG] Initializing EditPredictor")
        
        self.image_encoder = nn.Sequential(
            nn.Linear(image_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        self.edit_decoder = nn.Sequential(
            nn.Linear(64 + edit_encoding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_functions + max_params)
        )
        
        self.num_functions = num_functions
        self.max_params = max_params
        
    def forward(self, image_features, edit_encoding=None):
        img_encoded = self.image_encoder(image_features)
        
        if edit_encoding is not None:
            combined = torch.cat([img_encoded, edit_encoding], dim=-1)
        else:
            combined = torch.cat([img_encoded, torch.zeros(img_encoded.size(0), 64)], dim=-1)
            
        output = self.edit_decoder(combined)
        
        function_logits = output[:, :self.num_functions]
        param_values = torch.sigmoid(output[:, self.num_functions:])
        
        return function_logits, param_values

class EditLearner:
    def __init__(self):
        print("[LOG] Initializing EditLearner")
        self.feature_extractor = ImageFeatureExtractor()
        self.edit_history = []
        self.image_features_cache = {}
        self.models_trained = False
        
        self.function_names = []
        self.function_to_idx = {}
        self.idx_to_function = {}
        
        self.encoder = None
        self.predictor = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[LOG] Using device: {self.device}")
        
        self.save_path = Path('lightroom_ai_models')
        self.save_path.mkdir(exist_ok=True)
        
    def add_edit_sequence(self, image_path, edit_sequence):
        print(f"[LOG] Adding edit sequence for: {image_path}")
        
        if image_path not in self.image_features_cache:
            features = self.feature_extractor.extract_features(image_path)
            if features is not None:
                self.image_features_cache[image_path] = features
            else:
                print(f"[LOG] WARNING: Failed to extract features for: {image_path}")
                return
                
        self.edit_history.append({
            'image_path': image_path,
            'edit_sequence': edit_sequence,
            'timestamp': torch.tensor([len(self.edit_history)])
        })
        
        for edit in edit_sequence:
            func_name = edit['function']
            if func_name not in self.function_to_idx:
                idx = len(self.function_names)
                self.function_names.append(func_name)
                self.function_to_idx[func_name] = idx
                self.idx_to_function[idx] = func_name
                
        if len(self.edit_history) >= 5 and len(self.edit_history) % 5 == 0:
            print("[LOG] Auto-training models with accumulated data")
            self.train_models()
            
    def encode_edit_sequence(self, edit_sequence):
        if not self.function_names:
            return None
            
        encoded = []
        for edit in edit_sequence:
            func_name = edit['function']
            params = edit['params']
            
            func_encoding = np.zeros(len(self.function_names))
            if func_name in self.function_to_idx:
                func_encoding[self.function_to_idx[func_name]] = 1
                
            param_encoding = []
            for param_name, param_value in sorted(params.items()):
                if isinstance(param_value, bool):
                    param_encoding.append(float(param_value))
                elif isinstance(param_value, (int, float)):
                    param_encoding.append(float(param_value) / 255.0)
                elif isinstance(param_value, str):
                    param_encoding.append(hash(param_value) % 100 / 100.0)
                    
            param_encoding = param_encoding[:20]
            param_encoding.extend([0] * (20 - len(param_encoding)))
            
            edit_encoding = np.concatenate([func_encoding, param_encoding])
            encoded.append(edit_encoding)
            
        return np.array(encoded, dtype=np.float32)
    
    def train_models(self):
        if len(self.edit_history) < 3:
            print("[LOG] Not enough edit history to train")
            return
            
        print(f"[LOG] Training models with {len(self.edit_history)} examples")
        
        all_features = []
        for entry in self.edit_history:
            if entry['image_path'] in self.image_features_cache:
                all_features.append(self.image_features_cache[entry['image_path']])
                
        if len(all_features) < 2:
            print("[LOG] Not enough valid features to train")
            return
            
        transformed_features = self.feature_extractor.fit_transform(all_features)
        
        if self.encoder is None:
            input_dim = len(self.function_names) + 20
            self.encoder = EditSequenceEncoder(input_dim, hidden_dim=64, num_layers=1)
            self.encoder.to(self.device)
            
        if self.predictor is None:
            self.predictor = EditPredictor(
                image_feature_dim=transformed_features.shape[1],
                edit_encoding_dim=32,
                num_functions=len(self.function_names),
                max_params=20
            )
            self.predictor.to(self.device)
            
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        
        criterion_func = nn.CrossEntropyLoss()
        criterion_param = nn.MSELoss()
        
        self.encoder.train()
        self.predictor.train()
        
        num_epochs = min(50, len(self.edit_history) * 5)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i, entry in enumerate(self.edit_history):
                if entry['image_path'] not in self.image_features_cache:
                    continue
                    
                features = self.image_features_cache[entry['image_path']]
                transformed = self.feature_extractor.transform(features)
                
                encoded_sequence = self.encode_edit_sequence(entry['edit_sequence'])
                if encoded_sequence is None or len(encoded_sequence) == 0:
                    continue
                    
                img_tensor = torch.FloatTensor(transformed).to(self.device)
                seq_tensor = torch.FloatTensor(encoded_sequence).unsqueeze(0).to(self.device)
                
                edit_encoding = self.encoder(seq_tensor)
                
                func_logits, param_preds = self.predictor(img_tensor, edit_encoding)
                
                if entry['edit_sequence']:
                    first_edit = entry['edit_sequence'][0]
                    func_name = first_edit['function']
                    if func_name in self.function_to_idx:
                        target_func = torch.LongTensor([self.function_to_idx[func_name]]).to(self.device)
                        
                        target_params = torch.zeros(20).to(self.device)
                        param_values = list(first_edit['params'].values())[:20]
                        for j, val in enumerate(param_values):
                            if isinstance(val, (int, float)):
                                target_params[j] = float(val) / 255.0
                                
                        loss_func = criterion_func(func_logits, target_func)
                        loss_param = criterion_param(param_preds.squeeze(), target_params)
                        
                        loss = loss_func + 0.5 * loss_param
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters()) + list(self.predictor.parameters()), 
                            max_norm=1.0
                        )
                        optimizer.step()
                        
                        total_loss += loss.item()
                        
            if epoch % 10 == 0:
                print(f"[LOG] Training epoch {epoch}/{num_epochs}, Loss: {total_loss:.4f}")
                
        self.models_trained = True
        self.save_models()
        print("[LOG] Model training completed")
        
    def suggest_edits(self, image_path):
        print(f"[LOG] Suggesting edits for: {image_path}")
        
        if not self.models_trained:
            print("[LOG] Models not trained yet")
            return None
            
        features = self.feature_extractor.extract_features(image_path)
        if features is None:
            print("[LOG] Failed to extract features")
            return None
            
        try:
            transformed = self.feature_extractor.transform(features)
            img_tensor = torch.FloatTensor(transformed).to(self.device)
            
            self.encoder.eval()
            self.predictor.eval()
            
            with torch.no_grad():
                func_logits, param_preds = self.predictor(img_tensor.unsqueeze(0))
                
                suggested_edits = []
                
                func_probs = F.softmax(func_logits, dim=-1)
                top_k = min(3, len(self.function_names))
                top_functions = torch.topk(func_probs[0], top_k)
                
                for i in range(top_k):
                    func_idx = top_functions.indices[i].item()
                    func_prob = top_functions.values[i].item()
                    
                    if func_prob < 0.1:
                        continue
                        
                    if func_idx in self.idx_to_function:
                        func_name = self.idx_to_function[func_idx]
                        
                        params = {}
                        param_values = param_preds[0].cpu().numpy()
                        
                        if func_name in ['brightness', 'contrast', 'exposure']:
                            params[func_name.lower()] = int(param_values[0] * 200 - 100)
                        elif func_name in ['blur', 'gaussian_blur']:
                            params['kernel_size'] = int(param_values[0] * 20) * 2 + 1
                            if func_name == 'gaussian_blur':
                                params['sigma_x'] = int(param_values[1] * 10)
                        elif func_name == 'canny':
                            params['threshold1'] = int(param_values[0] * 200)
                            params['threshold2'] = int(param_values[1] * 300)
                            params['aperture_size'] = 3
                        else:
                            params['value'] = int(param_values[0] * 100)
                            
                        suggested_edits.append({
                            'function': func_name,
                            'params': params
                        })
                        
                        print(f"[LOG] Suggested: {func_name} with confidence {func_prob:.2f}")
                        
                return suggested_edits if suggested_edits else None
                
        except Exception as e:
            print(f"[LOG] ERROR in suggesting edits: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def save_models(self):
        print("[LOG] Saving models")
        
        checkpoint = {
            'encoder_state': self.encoder.state_dict() if self.encoder else None,
            'predictor_state': self.predictor.state_dict() if self.predictor else None,
            'function_names': self.function_names,
            'function_to_idx': self.function_to_idx,
            'idx_to_function': self.idx_to_function,
            'feature_scaler': self.feature_extractor.scaler,
            'feature_pca': self.feature_extractor.pca,
            'fitted': self.feature_extractor.fitted,
            'edit_history': self.edit_history[-100:]
        }
        
        torch.save(checkpoint, self.save_path / 'lightroom_ai_model.pth')
        
        with open(self.save_path / 'feature_cache.pkl', 'wb') as f:
            pickle.dump(dict(list(self.image_features_cache.items())[-100:]), f)
            
        print("[LOG] Models saved successfully")
        
    def load_models(self):
        print("[LOG] Loading models")
        
        model_path = self.save_path / 'lightroom_ai_model.pth'
        if not model_path.exists():
            print("[LOG] No saved models found")
            return False
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            self.function_names = checkpoint['function_names']
            self.function_to_idx = checkpoint['function_to_idx']
            self.idx_to_function = checkpoint['idx_to_function']
            self.feature_extractor.scaler = checkpoint['feature_scaler']
            self.feature_extractor.pca = checkpoint['feature_pca']
            self.feature_extractor.fitted = checkpoint['fitted']
            self.edit_history = checkpoint['edit_history']
            
            if checkpoint['encoder_state'] and self.function_names:
                input_dim = len(self.function_names) + 20
                self.encoder = EditSequenceEncoder(input_dim, hidden_dim=64, num_layers=1)
                self.encoder.load_state_dict(checkpoint['encoder_state'])
                self.encoder.to(self.device)
                
            if checkpoint['predictor_state'] and self.function_names:
                feature_dim = self.feature_extractor.pca.n_components if self.feature_extractor.fitted else 50
                self.predictor = EditPredictor(
                    image_feature_dim=feature_dim,
                    edit_encoding_dim=32,
                    num_functions=len(self.function_names),
                    max_params=20
                )
                self.predictor.load_state_dict(checkpoint['predictor_state'])
                self.predictor.to(self.device)
                
            cache_path = self.save_path / 'feature_cache.pkl'
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.image_features_cache = pickle.load(f)
                    
            self.models_trained = True
            print("[LOG] Models loaded successfully")
            return True
            
        except Exception as e:
            print(f"[LOG] ERROR loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
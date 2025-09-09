// William & Mary Photo Editor - Optimized with WebGL, Web Workers, and Advanced Features
// Production-ready professional photo editing application

class PhotoEditorOptimized {
    constructor() {
        this.canvas = document.getElementById('imageCanvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        
        // WebGL setup for GPU acceleration with better fallback
        this.glCanvas = document.createElement('canvas');
        try {
            this.gl = this.glCanvas.getContext('webgl2', { 
                preserveDrawingBuffer: true,
                antialias: false,
                depth: false,
                stencil: false,
                premultipliedAlpha: false
            });
            if (!this.gl) {
                this.gl = this.glCanvas.getContext('webgl', {
                    preserveDrawingBuffer: true,
                    antialias: false,
                    depth: false,
                    stencil: false,
                    premultipliedAlpha: false
                });
            }
            if (this.gl) {
                console.log('WebGL context created successfully');
            }
        } catch (e) {
            console.warn('WebGL not available:', e);
            this.gl = null;
        }
        
        this.currentImage = null;
        this.originalImage = null;
        this.originalImageData = null;
        this.previewImage = null;
        this.currentAdjustments = {};
        this.undoStack = [];
        this.redoStack = [];
        this.imageHistory = {};
        this.currentImagePath = null;
        this.zoomLevel = 1;
        this.isProcessing = false;
        
        // Performance optimization
        this.frameRequestId = null;
        this.pendingAdjustments = false;
        this.lastProcessTime = 0;
        this.targetFPS = 60;
        this.frameTime = 1000 / this.targetFPS;
        
        // Web Worker for heavy processing
        this.setupWebWorker();
        
        // WebGL shaders cache
        this.shaderPrograms = {};
        
        // Debounce timers
        this.debounceTimers = {};
        
        this.initializeFunctions();
        this.initializeUI();
        this.setupEventListeners();
        this.initializeWebGL();
    }
    
    setupWebWorker() {
        // Create Web Worker for background processing
        const workerCode = `
            // Image processing functions in Web Worker
            self.addEventListener('message', function(e) {
                const { action, data, width, height, params } = e.data;
                
                switch(action) {
                    case 'processFilters':
                        const result = applyFilters(data, width, height, params);
                        self.postMessage({ action: 'filtersProcessed', data: result });
                        break;
                    
                    case 'processHistogram':
                        const histogram = calculateHistogram(data, width, height, params);
                        self.postMessage({ action: 'histogramProcessed', data: histogram });
                        break;
                    
                    case 'processEdgeDetection':
                        const edges = detectEdges(data, width, height, params);
                        self.postMessage({ action: 'edgesProcessed', data: edges });
                        break;
                        
                    case 'processMorphology':
                        const morph = applyMorphology(data, width, height, params);
                        self.postMessage({ action: 'morphologyProcessed', data: morph });
                        break;
                }
            });
            
            function applyFilters(data, width, height, params) {
                const { filterType, kernelSize, sigma } = params;
                const output = new Uint8ClampedArray(data);
                
                switch(filterType) {
                    case 'gaussian':
                        return gaussianBlur(data, width, height, kernelSize, sigma);
                    case 'median':
                        return medianFilter(data, width, height, kernelSize);
                    case 'bilateral':
                        return bilateralFilter(data, width, height, kernelSize, sigma);
                    case 'unsharp':
                        return unsharpMask(data, width, height, kernelSize, sigma);
                }
                return output;
            }
            
            function gaussianBlur(data, width, height, size, sigma) {
                const output = new Uint8ClampedArray(data.length);
                const kernel = createGaussianKernel(size, sigma);
                const halfSize = Math.floor(size / 2);
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        let r = 0, g = 0, b = 0, kernelSum = 0;
                        
                        for (let ky = -halfSize; ky <= halfSize; ky++) {
                            for (let kx = -halfSize; kx <= halfSize; kx++) {
                                const px = Math.min(width - 1, Math.max(0, x + kx));
                                const py = Math.min(height - 1, Math.max(0, y + ky));
                                const idx = (py * width + px) * 4;
                                const weight = kernel[ky + halfSize][kx + halfSize];
                                
                                r += data[idx] * weight;
                                g += data[idx + 1] * weight;
                                b += data[idx + 2] * weight;
                                kernelSum += weight;
                            }
                        }
                        
                        const idx = (y * width + x) * 4;
                        output[idx] = r / kernelSum;
                        output[idx + 1] = g / kernelSum;
                        output[idx + 2] = b / kernelSum;
                        output[idx + 3] = data[idx + 3];
                    }
                }
                return output;
            }
            
            function createGaussianKernel(size, sigma) {
                const kernel = [];
                const halfSize = Math.floor(size / 2);
                
                for (let y = -halfSize; y <= halfSize; y++) {
                    const row = [];
                    for (let x = -halfSize; x <= halfSize; x++) {
                        const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
                        row.push(value);
                    }
                    kernel.push(row);
                }
                return kernel;
            }
            
            function medianFilter(data, width, height, size) {
                const output = new Uint8ClampedArray(data.length);
                const halfSize = Math.floor(size / 2);
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const rValues = [], gValues = [], bValues = [];
                        
                        for (let ky = -halfSize; ky <= halfSize; ky++) {
                            for (let kx = -halfSize; kx <= halfSize; kx++) {
                                const px = Math.min(width - 1, Math.max(0, x + kx));
                                const py = Math.min(height - 1, Math.max(0, y + ky));
                                const idx = (py * width + px) * 4;
                                
                                rValues.push(data[idx]);
                                gValues.push(data[idx + 1]);
                                bValues.push(data[idx + 2]);
                            }
                        }
                        
                        rValues.sort((a, b) => a - b);
                        gValues.sort((a, b) => a - b);
                        bValues.sort((a, b) => a - b);
                        
                        const medianIdx = Math.floor(rValues.length / 2);
                        const idx = (y * width + x) * 4;
                        
                        output[idx] = rValues[medianIdx];
                        output[idx + 1] = gValues[medianIdx];
                        output[idx + 2] = bValues[medianIdx];
                        output[idx + 3] = data[idx + 3];
                    }
                }
                return output;
            }
            
            function bilateralFilter(data, width, height, size, sigmaSpace, sigmaColor = 75) {
                const output = new Uint8ClampedArray(data.length);
                const halfSize = Math.floor(size / 2);
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const centerIdx = (y * width + x) * 4;
                        let r = 0, g = 0, b = 0, weightSum = 0;
                        
                        for (let ky = -halfSize; ky <= halfSize; ky++) {
                            for (let kx = -halfSize; kx <= halfSize; kx++) {
                                const px = Math.min(width - 1, Math.max(0, x + kx));
                                const py = Math.min(height - 1, Math.max(0, y + ky));
                                const idx = (py * width + px) * 4;
                                
                                // Spatial weight
                                const spatialDist = Math.sqrt(kx * kx + ky * ky);
                                const spatialWeight = Math.exp(-(spatialDist * spatialDist) / (2 * sigmaSpace * sigmaSpace));
                                
                                // Color weight
                                const colorDist = Math.sqrt(
                                    Math.pow(data[idx] - data[centerIdx], 2) +
                                    Math.pow(data[idx + 1] - data[centerIdx + 1], 2) +
                                    Math.pow(data[idx + 2] - data[centerIdx + 2], 2)
                                );
                                const colorWeight = Math.exp(-(colorDist * colorDist) / (2 * sigmaColor * sigmaColor));
                                
                                const weight = spatialWeight * colorWeight;
                                
                                r += data[idx] * weight;
                                g += data[idx + 1] * weight;
                                b += data[idx + 2] * weight;
                                weightSum += weight;
                            }
                        }
                        
                        output[centerIdx] = r / weightSum;
                        output[centerIdx + 1] = g / weightSum;
                        output[centerIdx + 2] = b / weightSum;
                        output[centerIdx + 3] = data[centerIdx + 3];
                    }
                }
                return output;
            }
            
            function unsharpMask(data, width, height, radius, amount = 1.5) {
                const blurred = gaussianBlur(data, width, height, radius * 2 + 1, radius);
                const output = new Uint8ClampedArray(data.length);
                
                for (let i = 0; i < data.length; i += 4) {
                    output[i] = Math.min(255, Math.max(0, data[i] + (data[i] - blurred[i]) * amount));
                    output[i + 1] = Math.min(255, Math.max(0, data[i + 1] + (data[i + 1] - blurred[i + 1]) * amount));
                    output[i + 2] = Math.min(255, Math.max(0, data[i + 2] + (data[i + 2] - blurred[i + 2]) * amount));
                    output[i + 3] = data[i + 3];
                }
                return output;
            }
            
            function calculateHistogram(data, width, height, params) {
                const { equalize } = params;
                
                if (equalize) {
                    return histogramEqualization(data, width, height);
                }
                
                // Calculate histogram for display
                const histogram = {
                    r: new Array(256).fill(0),
                    g: new Array(256).fill(0),
                    b: new Array(256).fill(0),
                    luminance: new Array(256).fill(0)
                };
                
                for (let i = 0; i < data.length; i += 4) {
                    histogram.r[data[i]]++;
                    histogram.g[data[i + 1]]++;
                    histogram.b[data[i + 2]]++;
                    const lum = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
                    histogram.luminance[lum]++;
                }
                
                return histogram;
            }
            
            function histogramEqualization(data, width, height) {
                const output = new Uint8ClampedArray(data.length);
                const totalPixels = width * height;
                
                // Calculate histograms for each channel
                const histR = new Array(256).fill(0);
                const histG = new Array(256).fill(0);
                const histB = new Array(256).fill(0);
                
                for (let i = 0; i < data.length; i += 4) {
                    histR[data[i]]++;
                    histG[data[i + 1]]++;
                    histB[data[i + 2]]++;
                }
                
                // Calculate CDFs
                const cdfR = calculateCDF(histR, totalPixels);
                const cdfG = calculateCDF(histG, totalPixels);
                const cdfB = calculateCDF(histB, totalPixels);
                
                // Apply equalization
                for (let i = 0; i < data.length; i += 4) {
                    output[i] = cdfR[data[i]];
                    output[i + 1] = cdfG[data[i + 1]];
                    output[i + 2] = cdfB[data[i + 2]];
                    output[i + 3] = data[i + 3];
                }
                
                return output;
            }
            
            function calculateCDF(histogram, totalPixels) {
                const cdf = new Array(256);
                cdf[0] = histogram[0];
                
                for (let i = 1; i < 256; i++) {
                    cdf[i] = cdf[i - 1] + histogram[i];
                }
                
                // Find minimum non-zero value
                const cdfMin = cdf.find(val => val > 0) || 0;
                
                // Normalize
                const scale = 255.0 / (totalPixels - cdfMin);
                return cdf.map(val => Math.round((val - cdfMin) * scale));
            }
            
            function detectEdges(data, width, height, params) {
                const { method, threshold1, threshold2 } = params;
                
                switch(method) {
                    case 'sobel':
                        return sobelEdgeDetection(data, width, height, threshold1);
                    case 'canny':
                        return cannyEdgeDetection(data, width, height, threshold1, threshold2);
                    case 'laplacian':
                        return laplacianEdgeDetection(data, width, height, threshold1);
                    default:
                        return data;
                }
            }
            
            function sobelEdgeDetection(data, width, height, threshold = 50) {
                const output = new Uint8ClampedArray(data.length);
                const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
                const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
                
                // Convert to grayscale first
                const gray = new Uint8ClampedArray(width * height);
                for (let i = 0; i < data.length; i += 4) {
                    gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                }
                
                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        let gx = 0, gy = 0;
                        
                        for (let ky = -1; ky <= 1; ky++) {
                            for (let kx = -1; kx <= 1; kx++) {
                                const idx = (y + ky) * width + (x + kx);
                                gx += gray[idx] * sobelX[ky + 1][kx + 1];
                                gy += gray[idx] * sobelY[ky + 1][kx + 1];
                            }
                        }
                        
                        const magnitude = Math.sqrt(gx * gx + gy * gy);
                        const idx = (y * width + x) * 4;
                        const edgeValue = magnitude > threshold ? 255 : 0;
                        
                        output[idx] = edgeValue;
                        output[idx + 1] = edgeValue;
                        output[idx + 2] = edgeValue;
                        output[idx + 3] = 255;
                    }
                }
                
                return output;
            }
            
            function cannyEdgeDetection(data, width, height, lowThreshold = 50, highThreshold = 150) {
                // Simplified Canny edge detection
                const blurred = gaussianBlur(data, width, height, 5, 1.4);
                const edges = sobelEdgeDetection(blurred, width, height, lowThreshold);
                
                // Apply hysteresis thresholding
                const output = new Uint8ClampedArray(edges.length);
                
                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        const idx = (y * width + x) * 4;
                        
                        if (edges[idx] === 255) {
                            // Strong edge
                            output[idx] = output[idx + 1] = output[idx + 2] = 255;
                            output[idx + 3] = 255;
                        } else if (edges[idx] > lowThreshold) {
                            // Weak edge - check if connected to strong edge
                            let hasStrongNeighbor = false;
                            for (let dy = -1; dy <= 1; dy++) {
                                for (let dx = -1; dx <= 1; dx++) {
                                    const nIdx = ((y + dy) * width + (x + dx)) * 4;
                                    if (edges[nIdx] === 255) {
                                        hasStrongNeighbor = true;
                                        break;
                                    }
                                }
                                if (hasStrongNeighbor) break;
                            }
                            
                            const value = hasStrongNeighbor ? 255 : 0;
                            output[idx] = output[idx + 1] = output[idx + 2] = value;
                            output[idx + 3] = 255;
                        } else {
                            output[idx + 3] = 255;
                        }
                    }
                }
                
                return output;
            }
            
            function laplacianEdgeDetection(data, width, height, threshold = 30) {
                const output = new Uint8ClampedArray(data.length);
                const kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]];
                
                // Convert to grayscale
                const gray = new Uint8ClampedArray(width * height);
                for (let i = 0; i < data.length; i += 4) {
                    gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                }
                
                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        let sum = 0;
                        
                        for (let ky = -1; ky <= 1; ky++) {
                            for (let kx = -1; kx <= 1; kx++) {
                                const idx = (y + ky) * width + (x + kx);
                                sum += gray[idx] * kernel[ky + 1][kx + 1];
                            }
                        }
                        
                        const idx = (y * width + x) * 4;
                        const edgeValue = Math.abs(sum) > threshold ? 255 : 0;
                        
                        output[idx] = edgeValue;
                        output[idx + 1] = edgeValue;
                        output[idx + 2] = edgeValue;
                        output[idx + 3] = 255;
                    }
                }
                
                return output;
            }
            
            function applyMorphology(data, width, height, params) {
                const { operation, kernelSize, shape } = params;
                
                // Convert to binary image first
                const binary = new Uint8ClampedArray(width * height);
                for (let i = 0; i < data.length; i += 4) {
                    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    binary[i / 4] = gray > 128 ? 255 : 0;
                }
                
                let result;
                switch(operation) {
                    case 'erosion':
                        result = morphErosion(binary, width, height, kernelSize, shape);
                        break;
                    case 'dilation':
                        result = morphDilation(binary, width, height, kernelSize, shape);
                        break;
                    case 'opening':
                        const eroded = morphErosion(binary, width, height, kernelSize, shape);
                        result = morphDilation(eroded, width, height, kernelSize, shape);
                        break;
                    case 'closing':
                        const dilated = morphDilation(binary, width, height, kernelSize, shape);
                        result = morphErosion(dilated, width, height, kernelSize, shape);
                        break;
                    case 'gradient':
                        const dil = morphDilation(binary, width, height, kernelSize, shape);
                        const ero = morphErosion(binary, width, height, kernelSize, shape);
                        result = new Uint8ClampedArray(width * height);
                        for (let i = 0; i < result.length; i++) {
                            result[i] = dil[i] - ero[i];
                        }
                        break;
                    default:
                        result = binary;
                }
                
                // Convert back to RGBA
                const output = new Uint8ClampedArray(data.length);
                for (let i = 0; i < result.length; i++) {
                    const idx = i * 4;
                    output[idx] = output[idx + 1] = output[idx + 2] = result[i];
                    output[idx + 3] = 255;
                }
                
                return output;
            }
            
            function createKernel(size, shape) {
                const kernel = [];
                const center = Math.floor(size / 2);
                
                for (let y = 0; y < size; y++) {
                    const row = [];
                    for (let x = 0; x < size; x++) {
                        if (shape === 'square') {
                            row.push(1);
                        } else if (shape === 'cross') {
                            row.push((x === center || y === center) ? 1 : 0);
                        } else { // circle
                            const dist = Math.sqrt((x - center) ** 2 + (y - center) ** 2);
                            row.push(dist <= center ? 1 : 0);
                        }
                    }
                    kernel.push(row);
                }
                return kernel;
            }
            
            function morphErosion(binary, width, height, kernelSize, shape) {
                const output = new Uint8ClampedArray(binary.length);
                const kernel = createKernel(kernelSize, shape);
                const halfSize = Math.floor(kernelSize / 2);
                
                for (let y = halfSize; y < height - halfSize; y++) {
                    for (let x = halfSize; x < width - halfSize; x++) {
                        let min = 255;
                        
                        for (let ky = 0; ky < kernelSize; ky++) {
                            for (let kx = 0; kx < kernelSize; kx++) {
                                if (kernel[ky][kx] === 1) {
                                    const idx = (y + ky - halfSize) * width + (x + kx - halfSize);
                                    min = Math.min(min, binary[idx]);
                                }
                            }
                        }
                        
                        output[y * width + x] = min;
                    }
                }
                
                return output;
            }
            
            function morphDilation(binary, width, height, kernelSize, shape) {
                const output = new Uint8ClampedArray(binary.length);
                const kernel = createKernel(kernelSize, shape);
                const halfSize = Math.floor(kernelSize / 2);
                
                for (let y = halfSize; y < height - halfSize; y++) {
                    for (let x = halfSize; x < width - halfSize; x++) {
                        let max = 0;
                        
                        for (let ky = 0; ky < kernelSize; ky++) {
                            for (let kx = 0; kx < kernelSize; kx++) {
                                if (kernel[ky][kx] === 1) {
                                    const idx = (y + ky - halfSize) * width + (x + kx - halfSize);
                                    max = Math.max(max, binary[idx]);
                                }
                            }
                        }
                        
                        output[y * width + x] = max;
                    }
                }
                
                return output;
            }
        `;
        
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        this.worker = new Worker(URL.createObjectURL(blob));
        
        this.worker.addEventListener('message', (e) => {
            this.handleWorkerMessage(e.data);
        });
    }
    
    handleWorkerMessage(message) {
        const { action, data } = message;
        
        switch(action) {
            case 'filtersProcessed':
            case 'histogramProcessed':
            case 'edgesProcessed':
            case 'morphologyProcessed':
                // Update canvas with processed data
                if (data instanceof Uint8ClampedArray) {
                    const imageData = new ImageData(data, this.canvas.width, this.canvas.height);
                    this.ctx.putImageData(imageData, 0, 0);
                    this.isProcessing = false;
                    this.updateStatus('Processing complete');
                }
                break;
        }
    }
    
    initializeWebGL() {
        if (!this.gl) {
            console.warn('WebGL not supported, falling back to Canvas 2D');
            return;
        }
        
        // Create vertex shader
        const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            varying vec2 v_texCoord;
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;
        
        // Base fragment shader for adjustments
        const fragmentShaderSource = `
            precision mediump float;
            
            uniform sampler2D u_image;
            uniform float u_brightness;
            uniform float u_contrast;
            uniform float u_saturation;
            uniform float u_exposure;
            uniform float u_highlights;
            uniform float u_shadows;
            uniform float u_vibrance;
            uniform float u_temperature;
            uniform float u_tint;
            uniform float u_clarity;
            uniform float u_dehaze;
            uniform vec2 u_resolution;
            
            varying vec2 v_texCoord;
            
            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }
            
            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }
            
            float luminance(vec3 color) {
                return dot(color, vec3(0.299, 0.587, 0.114));
            }
            
            vec3 adjustTemperature(vec3 color, float temp) {
                float temperature = temp / 100.0;
                color.r = clamp(color.r + temperature * 0.1, 0.0, 1.0);
                color.b = clamp(color.b - temperature * 0.1, 0.0, 1.0);
                return color;
            }
            
            vec3 adjustTint(vec3 color, float tint) {
                float t = tint / 100.0;
                color.g = clamp(color.g + t * 0.1, 0.0, 1.0);
                return color;
            }
            
            vec3 adjustHighlightsShadows(vec3 color, float highlights, float shadows) {
                float lum = luminance(color);
                float highlightMask = smoothstep(0.5, 1.0, lum);
                float shadowMask = 1.0 - smoothstep(0.0, 0.5, lum);
                
                color = mix(color, min(vec3(1.0), color * 1.5), highlightMask * highlights / 100.0);
                color = mix(color, max(vec3(0.0), color * 0.5), shadowMask * abs(shadows) / 100.0);
                
                return color;
            }
            
            vec3 adjustVibrance(vec3 color, float vibrance) {
                float average = (color.r + color.g + color.b) / 3.0;
                float mx = max(color.r, max(color.g, color.b));
                float amt = (mx - average) * (-vibrance / 100.0) * 3.0;
                color = mix(color, vec3(mx), amt);
                return color;
            }
            
            vec3 adjustClarity(vec3 color, vec2 texCoord) {
                // Local contrast enhancement
                vec3 sum = vec3(0.0);
                float kernel[9];
                kernel[0] = -1.0; kernel[1] = -1.0; kernel[2] = -1.0;
                kernel[3] = -1.0; kernel[4] = 9.0;  kernel[5] = -1.0;
                kernel[6] = -1.0; kernel[7] = -1.0; kernel[8] = -1.0;
                
                vec2 step = 1.0 / u_resolution;
                
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        vec2 offset = vec2(float(i), float(j)) * step;
                        vec3 sample = texture2D(u_image, texCoord + offset).rgb;
                        sum += sample * kernel[(i + 1) * 3 + (j + 1)];
                    }
                }
                
                return mix(color, sum, u_clarity / 100.0);
            }
            
            void main() {
                vec4 pixel = texture2D(u_image, v_texCoord);
                vec3 color = pixel.rgb;
                
                // Apply exposure
                color *= pow(2.0, u_exposure / 100.0);
                
                // Apply brightness
                color = color + u_brightness / 100.0;
                
                // Apply contrast
                color = (color - 0.5) * (1.0 + u_contrast / 100.0) + 0.5;
                
                // Apply highlights and shadows
                color = adjustHighlightsShadows(color, u_highlights, u_shadows);
                
                // Apply temperature and tint
                color = adjustTemperature(color, u_temperature);
                color = adjustTint(color, u_tint);
                
                // Apply vibrance
                color = adjustVibrance(color, u_vibrance);
                
                // Apply saturation
                vec3 gray = vec3(luminance(color));
                color = mix(gray, color, 1.0 + u_saturation / 100.0);
                
                // Apply clarity (local contrast)
                if (abs(u_clarity) > 0.01) {
                    color = adjustClarity(color, v_texCoord);
                }
                
                // Apply dehaze
                if (abs(u_dehaze) > 0.01) {
                    vec3 fogColor = vec3(0.5, 0.6, 0.7);
                    float dehazeFactor = u_dehaze / 100.0;
                    color = mix(color, color * 1.2 - fogColor * 0.2, dehazeFactor);
                }
                
                // Clamp final color
                color = clamp(color, 0.0, 1.0);
                
                gl_FragColor = vec4(color, pixel.a);
            }
        `;
        
        // Compile shaders
        this.vertexShader = this.compileShader(vertexShaderSource, this.gl.VERTEX_SHADER);
        this.fragmentShader = this.compileShader(fragmentShaderSource, this.gl.FRAGMENT_SHADER);
        
        // Create and link program
        this.glProgram = this.gl.createProgram();
        this.gl.attachShader(this.glProgram, this.vertexShader);
        this.gl.attachShader(this.glProgram, this.fragmentShader);
        this.gl.linkProgram(this.glProgram);
        
        if (!this.gl.getProgramParameter(this.glProgram, this.gl.LINK_STATUS)) {
            console.error('Unable to initialize shader program:', this.gl.getProgramInfoLog(this.glProgram));
            return;
        }
        
        // Set up buffers
        this.setupWebGLBuffers();
    }
    
    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    setupWebGLBuffers() {
        // Create position buffer
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]);
        
        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
        
        // Create texture coordinate buffer
        const texCoords = new Float32Array([
            0, 1,
            1, 1,
            0, 0,
            1, 0
        ]);
        
        this.texCoordBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);
        
        // Create texture
        this.texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
    }
    
    processImageWithWebGL() {
        if (!this.gl || !this.glProgram || !this.originalImage) {
            // Fallback to Canvas2D if WebGL is not available
            this.processImageWithCanvas();
            return;
        }
        
        // Set canvas size to match image
        this.glCanvas.width = this.originalImage.width;
        this.glCanvas.height = this.originalImage.height;
        this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        
        // Use program
        this.gl.useProgram(this.glProgram);
        
        // Set position attribute
        const positionLocation = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
        
        // Set texture coordinate attribute
        const texCoordLocation = this.gl.getAttribLocation(this.glProgram, 'a_texCoord');
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.enableVertexAttribArray(texCoordLocation);
        this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 0, 0);
        
        // Upload image to texture
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.originalImage);
        
        // Set uniforms
        this.setWebGLUniforms();
        
        // Draw
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        
        // Copy result to main canvas
        this.ctx.drawImage(this.glCanvas, 0, 0);
    }
    
    setWebGLUniforms() {
        // Set all adjustment uniforms
        const uniforms = {
            u_brightness: this.getAdjustmentValue('basic_brightness_brightness', 0),
            u_contrast: this.getAdjustmentValue('basic_contrast_contrast', 0),
            u_saturation: this.getAdjustmentValue('basic_saturation_saturation', 0),
            u_exposure: this.getAdjustmentValue('basic_exposure_exposure', 0),
            u_highlights: this.getAdjustmentValue('basic_highlights_highlights', 0),
            u_shadows: this.getAdjustmentValue('basic_shadows_shadows', 0),
            u_vibrance: this.getAdjustmentValue('basic_vibrance_vibrance', 0),
            u_temperature: this.getAdjustmentValue('basic_temperature_temperature', 0),
            u_tint: this.getAdjustmentValue('basic_tint_tint', 0),
            u_clarity: this.getAdjustmentValue('basic_clarity_clarity', 0),
            u_dehaze: this.getAdjustmentValue('basic_dehaze_amount', 0)
        };
        
        // Set resolution uniform
        const resolutionLocation = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
        this.gl.uniform2f(resolutionLocation, this.glCanvas.width, this.glCanvas.height);
        
        // Set all other uniforms
        for (const [name, value] of Object.entries(uniforms)) {
            const location = this.gl.getUniformLocation(this.glProgram, name);
            if (location !== null) {
                this.gl.uniform1f(location, value);
            }
        }
        
        // Set texture uniform
        const imageLocation = this.gl.getUniformLocation(this.glProgram, 'u_image');
        this.gl.uniform1i(imageLocation, 0);
    }
    
    getAdjustmentValue(key, defaultValue = 0) {
        return this.currentAdjustments[key] !== undefined ? this.currentAdjustments[key] : defaultValue;
    }
    
    debounce(func, wait) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }
    
    createDebouncedAdjustment(key, delay = 50) {
        if (!this.debounceTimers[key]) {
            this.debounceTimers[key] = this.debounce(() => {
                this.applyAllAdjustments();
            }, delay);
        }
        return this.debounceTimers[key];
    }
    
    initializeFunctions() {
        // Define all photo editing functions
        this.functions = {
            basic: {
                brightness: {
                    params: {
                        brightness: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                contrast: {
                    params: {
                        contrast: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                exposure: {
                    params: {
                        exposure: { type: 'slider', min: -200, max: 200, default: 0 }
                    }
                },
                highlights: {
                    params: {
                        highlights: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                shadows: {
                    params: {
                        shadows: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                whites: {
                    params: {
                        whites: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                blacks: {
                    params: {
                        blacks: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                vibrance: {
                    params: {
                        vibrance: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                saturation: {
                    params: {
                        saturation: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                temperature: {
                    params: {
                        temperature: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                tint: {
                    params: {
                        tint: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                clarity: {
                    params: {
                        clarity: { type: 'slider', min: 0, max: 100, default: 0 }
                    }
                },
                dehaze: {
                    params: {
                        amount: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                vignette: {
                    params: {
                        amount: { type: 'slider', min: -100, max: 100, default: 0 },
                        midpoint: { type: 'slider', min: 0, max: 100, default: 50 }
                    }
                },
                grain: {
                    params: {
                        amount: { type: 'slider', min: 0, max: 100, default: 0 },
                        size: { type: 'slider', min: 1, max: 5, default: 2 }
                    }
                }
            },
            filters: {
                gaussian_blur: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 },
                        sigma: { type: 'slider', min: 0.5, max: 10, default: 1 }
                    }
                },
                box_blur: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 }
                    }
                },
                median_filter: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 }
                    }
                },
                bilateral_filter: {
                    params: {
                        d: { type: 'slider', min: 5, max: 15, default: 9 },
                        sigma_color: { type: 'slider', min: 10, max: 200, default: 75 },
                        sigma_space: { type: 'slider', min: 10, max: 200, default: 75 }
                    }
                },
                sharpen: {
                    params: {
                        amount: { type: 'slider', min: 0, max: 300, default: 100 }
                    }
                },
                unsharp_mask: {
                    params: {
                        radius: { type: 'slider', min: 0.5, max: 5, default: 1 },
                        amount: { type: 'slider', min: 0, max: 200, default: 100 },
                        threshold: { type: 'slider', min: 0, max: 100, default: 0 }
                    }
                }
            },
            threshold: {
                simple_threshold: {
                    params: {
                        threshold: { type: 'slider', min: 0, max: 255, default: 128 }
                    }
                },
                adaptive_threshold: {
                    params: {
                        method: { type: 'combo', values: ['mean', 'gaussian'], default: 'mean' },
                        block_size: { type: 'slider', min: 3, max: 21, default: 11 },
                        c: { type: 'slider', min: -20, max: 20, default: 2 }
                    }
                },
                otsu_threshold: {
                    params: {
                        invert: { type: 'checkbox', default: false }
                    }
                }
            },
            morphology: {
                erosion: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 },
                        shape: { type: 'combo', values: ['square', 'cross', 'circle'], default: 'square' }
                    }
                },
                dilation: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 },
                        shape: { type: 'combo', values: ['square', 'cross', 'circle'], default: 'square' }
                    }
                },
                opening: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 },
                        shape: { type: 'combo', values: ['square', 'cross', 'circle'], default: 'square' }
                    }
                },
                closing: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 },
                        shape: { type: 'combo', values: ['square', 'cross', 'circle'], default: 'square' }
                    }
                },
                gradient: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 21, default: 5 },
                        shape: { type: 'combo', values: ['square', 'cross', 'circle'], default: 'square' }
                    }
                }
            },
            edge: {
                canny_edge: {
                    params: {
                        threshold1: { type: 'slider', min: 0, max: 255, default: 100 },
                        threshold2: { type: 'slider', min: 0, max: 255, default: 200 }
                    }
                },
                sobel_edge: {
                    params: {
                        ksize: { type: 'slider', min: 3, max: 7, default: 3 },
                        threshold: { type: 'slider', min: 0, max: 255, default: 50 }
                    }
                },
                laplacian_edge: {
                    params: {
                        ksize: { type: 'slider', min: 3, max: 7, default: 3 },
                        threshold: { type: 'slider', min: 0, max: 100, default: 30 }
                    }
                },
                prewitt_edge: {
                    params: {
                        threshold: { type: 'slider', min: 0, max: 255, default: 50 }
                    }
                }
            },
            enhance: {
                histogram_equalization: {
                    params: {
                        clip_limit: { type: 'slider', min: 1, max: 10, default: 2 }
                    }
                },
                adaptive_histogram: {
                    params: {
                        clip_limit: { type: 'slider', min: 1, max: 10, default: 3 },
                        grid_size: { type: 'slider', min: 2, max: 16, default: 8 }
                    }
                },
                gamma_correction: {
                    params: {
                        gamma: { type: 'slider', min: 10, max: 300, default: 100 }
                    }
                },
                contrast_stretching: {
                    params: {
                        min_percentile: { type: 'slider', min: 0, max: 50, default: 2 },
                        max_percentile: { type: 'slider', min: 50, max: 100, default: 98 }
                    }
                }
            },
            color: {
                hue_shift: {
                    params: {
                        shift: { type: 'slider', min: -180, max: 180, default: 0 }
                    }
                },
                color_balance: {
                    params: {
                        red: { type: 'slider', min: -100, max: 100, default: 0 },
                        green: { type: 'slider', min: -100, max: 100, default: 0 },
                        blue: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                channel_mixer: {
                    params: {
                        red_red: { type: 'slider', min: -200, max: 200, default: 100 },
                        red_green: { type: 'slider', min: -200, max: 200, default: 0 },
                        red_blue: { type: 'slider', min: -200, max: 200, default: 0 },
                        green_red: { type: 'slider', min: -200, max: 200, default: 0 },
                        green_green: { type: 'slider', min: -200, max: 200, default: 100 },
                        green_blue: { type: 'slider', min: -200, max: 200, default: 0 },
                        blue_red: { type: 'slider', min: -200, max: 200, default: 0 },
                        blue_green: { type: 'slider', min: -200, max: 200, default: 0 },
                        blue_blue: { type: 'slider', min: -200, max: 200, default: 100 }
                    }
                },
                selective_color: {
                    params: {
                        target: { type: 'combo', values: ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas'], default: 'reds' },
                        cyan: { type: 'slider', min: -100, max: 100, default: 0 },
                        magenta: { type: 'slider', min: -100, max: 100, default: 0 },
                        yellow: { type: 'slider', min: -100, max: 100, default: 0 },
                        black: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                }
            },
            artistic: {
                oil_painting: {
                    params: {
                        size: { type: 'slider', min: 3, max: 15, default: 7 },
                        dynRatio: { type: 'slider', min: 1, max: 10, default: 5 }
                    }
                },
                pencil_sketch: {
                    params: {
                        sigma_s: { type: 'slider', min: 10, max: 200, default: 60 },
                        sigma_r: { type: 'slider', min: 0.01, max: 1, default: 0.07 },
                        shade_factor: { type: 'slider', min: 0.01, max: 0.1, default: 0.05 }
                    }
                },
                cartoon: {
                    params: {
                        num_down: { type: 'slider', min: 1, max: 5, default: 2 },
                        num_bilateral: { type: 'slider', min: 1, max: 10, default: 7 },
                        edge_threshold: { type: 'slider', min: 10, max: 100, default: 20 }
                    }
                },
                watercolor: {
                    params: {
                        stylization: { type: 'slider', min: 0, max: 100, default: 50 },
                        edge_preservation: { type: 'slider', min: 0, max: 100, default: 50 }
                    }
                }
            }
        };
    }
    
    initializeUI() {
        // Create function widgets for each category
        Object.keys(this.functions).forEach(category => {
            const tabPane = document.getElementById(`${category}-tab`);
            if (!tabPane) return;
            
            Object.entries(this.functions[category]).forEach(([funcName, func]) => {
                const widget = this.createFunctionWidget(funcName, func.params, category);
                tabPane.appendChild(widget);
            });
        });
        
        // Setup tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Remove active from all tabs and panes
                document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                
                // Add active to clicked tab and corresponding pane
                e.target.classList.add('active');
                const tabId = `${e.target.dataset.tab}-tab`;
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    createFunctionWidget(funcName, params, category) {
        const widget = document.createElement('div');
        widget.className = 'function-widget';
        widget.dataset.function = funcName;
        widget.dataset.category = category;
        
        // Create header
        const header = document.createElement('div');
        header.className = 'function-header';
        
        const title = document.createElement('h4');
        title.textContent = this.formatFunctionName(funcName);
        title.className = 'function-name';
        header.appendChild(title);
        
        const resetBtn = document.createElement('button');
        resetBtn.className = 'reset-function-btn';
        resetBtn.innerHTML = '<i class="fas fa-undo"></i> Reset';
        resetBtn.title = 'Reset';
        resetBtn.onclick = () => this.resetFunction(funcName, params, category);
        header.appendChild(resetBtn);
        
        widget.appendChild(header);
        
        // Create parameter controls
        const paramsDiv = document.createElement('div');
        paramsDiv.className = 'function-params';
        
        Object.entries(params).forEach(([paramName, param]) => {
            const paramGroup = document.createElement('div');
            paramGroup.className = 'param-group';
            
            const label = document.createElement('label');
            label.textContent = this.formatParamName(paramName);
            label.className = 'param-label';
            paramGroup.appendChild(label);
            
            const control = document.createElement('div');
            control.className = 'param-control';
            
            const adjustmentKey = `${category}_${funcName}_${paramName}`;
            
            if (param.type === 'slider') {
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = param.min;
                slider.max = param.max;
                slider.value = param.default;
                slider.step = (param.max - param.min) > 100 ? 1 : 0.1;
                slider.id = `${funcName}_${paramName}`;
                
                const valueInput = document.createElement('input');
                valueInput.type = 'number';
                valueInput.min = param.min;
                valueInput.max = param.max;
                valueInput.value = param.default;
                valueInput.step = slider.step;
                valueInput.className = 'param-value-input';
                valueInput.id = `${funcName}_${paramName}_value`;
                
                // Initialize adjustment value
                this.currentAdjustments[adjustmentKey] = param.default;
                
                // Use debounced function for slider input
                const debouncedUpdate = this.createDebouncedAdjustment(adjustmentKey);
                
                slider.oninput = () => {
                    valueInput.value = slider.value;
                    this.currentAdjustments[adjustmentKey] = parseFloat(slider.value);
                    debouncedUpdate();
                };
                
                valueInput.oninput = () => {
                    let val = parseFloat(valueInput.value);
                    if (isNaN(val)) val = param.default;
                    val = Math.max(param.min, Math.min(param.max, val));
                    valueInput.value = val;
                    slider.value = val;
                    this.currentAdjustments[adjustmentKey] = val;
                    debouncedUpdate();
                };
                
                control.appendChild(slider);
                control.appendChild(valueInput);
            } else if (param.type === 'combo') {
                const select = document.createElement('select');
                select.id = `${funcName}_${paramName}`;
                
                param.values.forEach(val => {
                    const option = document.createElement('option');
                    option.value = val;
                    option.textContent = val;
                    if (val === param.default) option.selected = true;
                    select.appendChild(option);
                });
                
                // Initialize adjustment value
                this.currentAdjustments[adjustmentKey] = param.default;
                
                const debouncedUpdate = this.createDebouncedAdjustment(adjustmentKey);
                
                select.onchange = () => {
                    this.currentAdjustments[adjustmentKey] = select.value;
                    debouncedUpdate();
                };
                
                control.appendChild(select);
            } else if (param.type === 'checkbox') {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `${funcName}_${paramName}`;
                checkbox.checked = param.default;
                
                // Initialize adjustment value
                this.currentAdjustments[adjustmentKey] = param.default;
                
                const debouncedUpdate = this.createDebouncedAdjustment(adjustmentKey);
                
                checkbox.onchange = () => {
                    this.currentAdjustments[adjustmentKey] = checkbox.checked;
                    debouncedUpdate();
                };
                
                control.appendChild(checkbox);
            }
            
            paramGroup.appendChild(control);
            paramsDiv.appendChild(paramGroup);
        });
        
        widget.appendChild(paramsDiv);
        return widget;
    }
    
    resetFunction(funcName, params, category) {
        Object.keys(params).forEach(paramName => {
            const param = params[paramName];
            const elementId = `${funcName}_${paramName}`;
            const element = document.getElementById(elementId);
            const adjustmentKey = `${category}_${funcName}_${paramName}`;
            
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = param.default;
                } else {
                    element.value = param.default;
                    // Update value input for sliders
                    const valueInput = document.getElementById(`${funcName}_${paramName}_value`);
                    if (valueInput) {
                        valueInput.value = param.default;
                    }
                }
                this.currentAdjustments[adjustmentKey] = param.default;
            }
        });
        
        this.applyAllAdjustments();
    }
    
    formatFunctionName(name) {
        return name.replace(/_/g, ' ')
                   .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    formatParamName(name) {
        return name.replace(/_/g, ' ')
                   .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    setupEventListeners() {
        // File input
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files);
        });
        
        // Drag and drop
        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.canvas.classList.add('drag-over');
        });
        
        this.canvas.addEventListener('dragleave', () => {
            this.canvas.classList.remove('drag-over');
        });
        
        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            this.canvas.classList.remove('drag-over');
            this.handleFileSelect(e.dataTransfer.files);
        });
        
        // Zoom controls
        document.getElementById('zoomInBtn').addEventListener('click', () => this.zoom(1.25));
        document.getElementById('zoomOutBtn').addEventListener('click', () => this.zoom(0.8));
        document.getElementById('fitBtn').addEventListener('click', () => this.fitToScreen());
        
        // Control buttons
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetAllAdjustments());
        document.getElementById('exportBtn').addEventListener('click', () => this.showExportModal());
        document.getElementById('saveEditsBtn').addEventListener('click', () => this.saveEditPreset());
        document.getElementById('loadEditsBtn').addEventListener('click', () => this.loadEditPreset());
        document.getElementById('automaticModeBtn').addEventListener('click', () => this.applyAutomaticMode());
        
        // Export quality slider
        const qualitySlider = document.getElementById('exportQuality');
        const qualityValue = document.getElementById('qualityValue');
        if (qualitySlider && qualityValue) {
            qualitySlider.addEventListener('input', () => {
                qualityValue.textContent = qualitySlider.value + '%';
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'z':
                        e.preventDefault();
                        this.undo();
                        break;
                    case 'y':
                        e.preventDefault();
                        this.redo();
                        break;
                    case 's':
                        e.preventDefault();
                        this.showExportModal();
                        break;
                    case 'r':
                        e.preventDefault();
                        this.resetAllAdjustments();
                        break;
                    case '0':
                        e.preventDefault();
                        this.fitToScreen();
                        break;
                    case '=':
                    case '+':
                        e.preventDefault();
                        this.zoom(1.25);
                        break;
                    case '-':
                        e.preventDefault();
                        this.zoom(0.8);
                        break;
                }
            }
        });
        
        // Performance monitoring
        this.startPerformanceMonitoring();
        
        // Directory and Export buttons
        const openDirBtn = document.getElementById('openDirectoryBtn');
        if (openDirBtn) {
            openDirBtn.addEventListener('click', () => {
                document.getElementById('fileInput').click();
            });
        }
        
        const exportAllBtn = document.getElementById('exportAllBtn');
        if (exportAllBtn) {
            exportAllBtn.addEventListener('click', () => {
                this.exportAllImages();
            });
        }
    }
    
    startPerformanceMonitoring() {
        let lastTime = performance.now();
        let frames = 0;
        
        const updateFPS = () => {
            const currentTime = performance.now();
            frames++;
            
            if (currentTime >= lastTime + 1000) {
                const fps = Math.round((frames * 1000) / (currentTime - lastTime));
                document.getElementById('performanceInfo').textContent = `FPS: ${fps}`;
                frames = 0;
                lastTime = currentTime;
            }
            
            requestAnimationFrame(updateFPS);
        };
        
        updateFPS();
    }
    
    handleFileSelect(files) {
        const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
        
        if (imageFiles.length === 0) {
            this.updateStatus('No valid image files selected');
            return;
        }
        
        // Clear gallery
        const gallery = document.getElementById('imageGallery');
        gallery.innerHTML = '';
        
        // Add images to gallery
        imageFiles.forEach((file, index) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const thumbnail = document.createElement('div');
                thumbnail.className = 'image-thumbnail';
                if (index === 0) thumbnail.classList.add('active');
                
                const img = document.createElement('img');
                img.src = e.target.result;
                img.dataset.filename = file.name;
                
                thumbnail.appendChild(img);
                thumbnail.addEventListener('click', () => {
                    document.querySelectorAll('.image-thumbnail').forEach(t => t.classList.remove('active'));
                    thumbnail.classList.add('active');
                    this.loadImage(e.target.result);
                });
                
                gallery.appendChild(thumbnail);
                
                // Load first image automatically
                if (index === 0) {
                    this.loadImage(e.target.result);
                }
            };
            
            reader.readAsDataURL(file);
        });
        
        this.updateStatus(`Loaded ${imageFiles.length} image(s)`);
    }
    
    loadImage(src) {
        const img = new Image();
        img.onload = () => {
            this.originalImage = img;
            this.currentImage = img;
            
            // Set canvas size
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            
            // Draw image
            this.ctx.drawImage(img, 0, 0);
            
            // Store original image data
            this.originalImageData = this.ctx.getImageData(0, 0, img.width, img.height);
            
            // Update UI
            this.updateImageInfo(img.width, img.height);
            this.fitToScreen();
            
            // Reset adjustments for new image
            this.resetAllAdjustments();
            
            // Hide placeholder
            const placeholder = document.querySelector('.canvas-placeholder');
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        };
        img.src = src;
    }
    
    applyAllAdjustments() {
        if (!this.currentImage || this.isProcessing) return;
        
        this.isProcessing = true;
        
        // Cancel any pending frame request
        if (this.frameRequestId) {
            cancelAnimationFrame(this.frameRequestId);
        }
        
        // Use requestAnimationFrame for smoother updates
        this.frameRequestId = requestAnimationFrame(() => {
            const startTime = performance.now();
            
            // Always try Canvas2D first for reliability
            try {
                // Check if we should use WebGL or Canvas2D
                if (this.gl && this.glProgram && this.hasOnlyBasicAdjustments()) {
                    // Use WebGL for basic adjustments (much faster)
                    this.processImageWithWebGL();
                } else {
                    // Fallback to Canvas2D
                    this.processImageWithCanvas();
                }
            } catch (e) {
                console.warn('Processing error, using fallback:', e);
                this.processImageWithCanvas();
            }
            
            // Check if we need to apply filters or other effects with Web Worker
            const hasComplexFilters = this.hasComplexFilters();
            if (hasComplexFilters) {
                const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
                this.processWithWebWorker(imageData);
            }
            
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            
            // Update status
            this.updateStatus(`Processed in ${processingTime.toFixed(1)}ms`);
            
            this.isProcessing = false;
            this.frameRequestId = null;
        });
    }
    
    hasOnlyBasicAdjustments() {
        // Check if only basic adjustments are active (for WebGL optimization)
        const basicKeys = ['basic_brightness_brightness', 'basic_contrast_contrast', 
                          'basic_saturation_saturation', 'basic_exposure_exposure',
                          'basic_highlights_highlights', 'basic_shadows_shadows',
                          'basic_vibrance_vibrance', 'basic_temperature_temperature',
                          'basic_tint_tint', 'basic_clarity_clarity', 'basic_dehaze_amount'];
        
        return Object.keys(this.currentAdjustments).every(key => {
            if (!basicKeys.includes(key)) {
                const value = this.currentAdjustments[key];
                // Check if non-basic adjustment is at default
                const parts = key.split('_');
                const category = parts[0];
                const funcName = parts.slice(1, -1).join('_');
                const paramName = parts[parts.length - 1];
                
                if (this.functions[category] && this.functions[category][funcName]) {
                    const param = this.functions[category][funcName].params[paramName];
                    return param && value === param.default;
                }
            }
            return true;
        });
    }
    
    hasComplexFilters() {
        // Check if any complex filters are active
        const complexCategories = ['filters', 'edge', 'morphology', 'artistic'];
        
        return Object.entries(this.currentAdjustments).some(([key, value]) => {
            const [category] = key.split('_');
            if (!complexCategories.includes(category)) return false;
            
            // Check if value is different from default
            const parts = key.split('_');
            const funcName = parts.slice(1, -1).join('_');
            const paramName = parts[parts.length - 1];
            
            if (this.functions[category] && this.functions[category][funcName]) {
                const param = this.functions[category][funcName].params[paramName];
                return param && value !== param.default;
            }
            return false;
        });
    }
    
    processWithWebWorker(imageData) {
        // Determine which filters to apply
        const filters = this.getActiveFilters();
        
        if (filters.length > 0) {
            this.worker.postMessage({
                action: 'processFilters',
                data: imageData.data,
                width: imageData.width,
                height: imageData.height,
                params: filters[0] // Process one at a time for now
            });
        }
    }
    
    getActiveFilters() {
        const filters = [];
        
        Object.entries(this.currentAdjustments).forEach(([key, value]) => {
            const [category, ...rest] = key.split('_');
            const funcName = rest.slice(0, -1).join('_');
            const paramName = rest[rest.length - 1];
            
            if (this.functions[category] && this.functions[category][funcName]) {
                const param = this.functions[category][funcName].params[paramName];
                if (param && value !== param.default) {
                    filters.push({
                        category,
                        funcName,
                        paramName,
                        value
                    });
                }
            }
        });
        
        return filters;
    }
    
    processImageWithCanvas() {
        if (!this.originalImageData) return;
        
        // Fallback Canvas2D processing
        const imageData = new ImageData(
            new Uint8ClampedArray(this.originalImageData.data),
            this.originalImageData.width,
            this.originalImageData.height
        );
        
        // Apply basic adjustments
        this.applyBasicAdjustments(imageData.data, imageData.width, imageData.height);
        
        // Apply complex filters if needed
        const complexFilters = this.getActiveFilters().filter(f => f.category !== 'basic');
        if (complexFilters.length > 0) {
            // Apply filters directly on canvas for now
            this.applyCanvasFilters(imageData, complexFilters);
        }
        
        // Put processed image back
        this.ctx.putImageData(imageData, 0, 0);
    }
    
    applyCanvasFilters(imageData, filters) {
        // Apply filters that don't need Web Worker
        filters.forEach(filter => {
            if (filter.category === 'filters' && filter.funcName === 'sharpen') {
                this.applySharpenFilter(imageData, filter.value);
            }
            // Add more filter implementations as needed
        });
    }
    
    applySharpenFilter(imageData, amount) {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        const factor = amount / 100;
        
        const output = new Uint8ClampedArray(data);
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                for (let c = 0; c < 3; c++) {
                    const idx = (y * width + x) * 4 + c;
                    const center = data[idx] * (1 + 4 * factor);
                    const neighbors = (
                        data[((y - 1) * width + x) * 4 + c] +
                        data[((y + 1) * width + x) * 4 + c] +
                        data[(y * width + (x - 1)) * 4 + c] +
                        data[(y * width + (x + 1)) * 4 + c]
                    ) * (-factor);
                    
                    output[idx] = Math.min(255, Math.max(0, center + neighbors));
                }
            }
        }
        
        // Copy output back to imageData
        for (let i = 0; i < data.length; i++) {
            data[i] = output[i];
        }
    }
    
    applyBasicAdjustments(data, width, height) {
        // Get adjustment values
        const brightness = this.getAdjustmentValue('basic_brightness_brightness', 0) / 100 * 255;
        const contrast = this.getAdjustmentValue('basic_contrast_contrast', 0) / 100;
        const saturation = this.getAdjustmentValue('basic_saturation_saturation', 0) / 100;
        const exposure = Math.pow(2, this.getAdjustmentValue('basic_exposure_exposure', 0) / 100);
        
        // Process pixels
        for (let i = 0; i < data.length; i += 4) {
            // Apply exposure
            data[i] = Math.min(255, data[i] * exposure);
            data[i + 1] = Math.min(255, data[i + 1] * exposure);
            data[i + 2] = Math.min(255, data[i + 2] * exposure);
            
            // Apply brightness
            data[i] = Math.min(255, Math.max(0, data[i] + brightness));
            data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + brightness));
            data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + brightness));
            
            // Apply contrast
            data[i] = Math.min(255, Math.max(0, (data[i] - 128) * (1 + contrast) + 128));
            data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] - 128) * (1 + contrast) + 128));
            data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] - 128) * (1 + contrast) + 128));
            
            // Apply saturation
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            data[i] = Math.min(255, Math.max(0, gray + (data[i] - gray) * (1 + saturation)));
            data[i + 1] = Math.min(255, Math.max(0, gray + (data[i + 1] - gray) * (1 + saturation)));
            data[i + 2] = Math.min(255, Math.max(0, gray + (data[i + 2] - gray) * (1 + saturation)));
        }
    }
    
    undo() {
        if (this.undoStack.length > 0) {
            const state = this.undoStack.pop();
            this.redoStack.push(this.currentAdjustments);
            this.currentAdjustments = { ...state };
            this.restoreAdjustments();
            this.applyAllAdjustments();
            this.updateStatus('Undo applied');
        }
    }
    
    redo() {
        if (this.redoStack.length > 0) {
            const state = this.redoStack.pop();
            this.undoStack.push(this.currentAdjustments);
            this.currentAdjustments = { ...state };
            this.restoreAdjustments();
            this.applyAllAdjustments();
            this.updateStatus('Redo applied');
        }
    }
    
    restoreAdjustments() {
        // Update all UI controls to match current adjustments
        Object.entries(this.currentAdjustments).forEach(([key, value]) => {
            const parts = key.split('_');
            const category = parts[0];
            const funcName = parts.slice(1, -1).join('_');
            const paramName = parts[parts.length - 1];
            
            const elementId = `${funcName}_${paramName}`;
            const element = document.getElementById(elementId);
            
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                    // Update value input for sliders
                    const valueInput = document.getElementById(`${funcName}_${paramName}_value`);
                    if (valueInput) {
                        valueInput.value = value;
                    }
                }
            }
        });
    }
    
    resetAllAdjustments() {
        // Save current state to undo stack
        this.undoStack.push({ ...this.currentAdjustments });
        
        // Reset all adjustments
        Object.keys(this.functions).forEach(category => {
            Object.entries(this.functions[category]).forEach(([funcName, func]) => {
                this.resetFunction(funcName, func.params, category);
            });
        });
        
        this.updateStatus('All adjustments reset');
    }
    
    zoom(factor) {
        this.zoomLevel *= factor;
        this.zoomLevel = Math.max(0.1, Math.min(5, this.zoomLevel));
        this.updateZoom();
    }
    
    fitToScreen() {
        if (!this.currentImage) return;
        
        const container = this.canvas.parentElement;
        const containerPadding = 40; // Account for padding
        const maxWidth = container.clientWidth - containerPadding;
        const maxHeight = container.clientHeight - containerPadding;
        
        const scaleX = maxWidth / this.currentImage.width;
        const scaleY = maxHeight / this.currentImage.height;
        
        this.zoomLevel = Math.min(scaleX, scaleY, 1);
        this.updateZoom();
    }
    
    updateZoom() {
        this.canvas.style.transform = `scale(${this.zoomLevel})`;
        this.canvas.style.transformOrigin = 'center center';
        const zoomDisplay = document.getElementById('zoomLevel');
        if (zoomDisplay) {
            zoomDisplay.textContent = Math.round(this.zoomLevel * 100) + '%';
        }
    }
    
    showExportModal() {
        if (!this.currentImage) {
            this.updateStatus('No image to export');
            return;
        }
        
        const modal = document.getElementById('exportModal');
        if (modal) {
            modal.classList.add('active');
        }
    }
    
    performExport() {
        const fileName = document.getElementById('exportFileName').value || 'edited_image.jpg';
        const format = document.getElementById('exportFormat').value || 'jpeg';
        const qualityInput = document.getElementById('exportQuality');
        const quality = qualityInput ? qualityInput.value / 100 : 0.95;
        
        // Ensure the file extension matches the format
        let finalFileName = fileName;
        const extensions = { 'jpeg': '.jpg', 'png': '.png', 'webp': '.webp', 'bmp': '.bmp' };
        const expectedExt = extensions[format];
        
        // Add or replace extension
        if (!finalFileName.toLowerCase().endsWith(expectedExt) && !finalFileName.toLowerCase().endsWith('.jpeg')) {
            // Remove any existing extension and add the correct one
            finalFileName = finalFileName.replace(/\.[^/.]+$/, '') + expectedExt;
        }
        
        // Convert canvas to blob
        const mimeType = format === 'jpeg' ? 'image/jpeg' : format === 'png' ? 'image/png' : format === 'webp' ? 'image/webp' : 'image/bmp';
        
        try {
            this.canvas.toBlob((blob) => {
                if (!blob) {
                    this.updateStatus('Export failed: Unable to create image');
                    return;
                }
                
                // Create download link
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = finalFileName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                setTimeout(() => URL.revokeObjectURL(url), 100);
                this.closeModal('exportModal');
                this.updateStatus(`Exported ${finalFileName}`);
            }, mimeType, quality);
        } catch (error) {
            console.error('Export error:', error);
            this.updateStatus('Export failed');
        }
    }
    
    saveEditPreset() {
        const preset = {
            adjustments: this.currentAdjustments,
            timestamp: new Date().toISOString()
        };
        
        const json = JSON.stringify(preset, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `preset_${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.updateStatus('Preset saved');
    }
    
    loadEditPreset() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const preset = JSON.parse(event.target.result);
                    
                    // Save current state to undo stack
                    this.undoStack.push({ ...this.currentAdjustments });
                    
                    // Apply preset
                    this.currentAdjustments = { ...preset.adjustments };
                    this.restoreAdjustments();
                    this.applyAllAdjustments();
                    
                    this.updateStatus('Preset loaded');
                } catch (err) {
                    this.updateStatus('Error loading preset');
                    console.error(err);
                }
            };
            
            reader.readAsText(file);
        };
        
        input.click();
    }
    
    applyAutomaticMode() {
        if (!this.currentImage || !this.originalImageData) {
            this.updateStatus('No image loaded');
            return;
        }
        
        this.updateStatus('Analyzing image...');
        
        // Save current state
        this.undoStack.push({ ...this.currentAdjustments });
        
        // Analyze image and determine optimal adjustments
        setTimeout(() => {
            const analysis = this.analyzeImage();
            
            // Apply intelligent adjustments based on analysis
            const autoAdjustments = {
                'basic_brightness_brightness': analysis.brightness,
                'basic_contrast_contrast': analysis.contrast,
                'basic_exposure_exposure': analysis.exposure,
                'basic_highlights_highlights': analysis.highlights,
                'basic_shadows_shadows': analysis.shadows,
                'basic_vibrance_vibrance': analysis.vibrance,
                'basic_saturation_saturation': analysis.saturation,
                'basic_clarity_clarity': analysis.clarity,
                'basic_dehaze_amount': analysis.dehaze,
                'basic_temperature_temperature': analysis.temperature,
                'basic_tint_tint': analysis.tint
            };
            
            // Apply adjustments
            Object.entries(autoAdjustments).forEach(([key, value]) => {
                this.currentAdjustments[key] = value;
            });
            
            this.restoreAdjustments();
            this.applyAllAdjustments();
            this.updateStatus('Automatic adjustments applied');
        }, 100);
    }
    
    analyzeImage() {
        const imageData = this.originalImageData;
        const data = imageData.data;
        const pixelCount = data.length / 4;
        
        // Calculate comprehensive statistics
        const histogram = new Array(256).fill(0);
        const colorStats = { r: 0, g: 0, b: 0, rMin: 255, rMax: 0, gMin: 255, gMax: 0, bMin: 255, bMax: 0 };
        let totalBrightness = 0;
        let totalSaturation = 0;
        let minBrightness = 255;
        let maxBrightness = 0;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Update color statistics
            colorStats.r += r;
            colorStats.g += g;
            colorStats.b += b;
            colorStats.rMin = Math.min(colorStats.rMin, r);
            colorStats.rMax = Math.max(colorStats.rMax, r);
            colorStats.gMin = Math.min(colorStats.gMin, g);
            colorStats.gMax = Math.max(colorStats.gMax, g);
            colorStats.bMin = Math.min(colorStats.bMin, b);
            colorStats.bMax = Math.max(colorStats.bMax, b);
            
            // Calculate brightness
            const brightness = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            histogram[brightness]++;
            totalBrightness += brightness;
            minBrightness = Math.min(minBrightness, brightness);
            maxBrightness = Math.max(maxBrightness, brightness);
            
            // Calculate saturation
            const max = Math.max(r, g, b);
            const min = Math.min(r, g, b);
            const sat = max === 0 ? 0 : (max - min) / max;
            totalSaturation += sat;
        }
        
        const avgBrightness = totalBrightness / pixelCount;
        const avgSaturation = totalSaturation / pixelCount;
        const range = maxBrightness - minBrightness;
        
        // Calculate color averages
        const avgR = colorStats.r / pixelCount;
        const avgG = colorStats.g / pixelCount;
        const avgB = colorStats.b / pixelCount;
        
        // Calculate standard deviation for contrast assessment
        let stdDev = 0;
        for (let i = 0; i < data.length; i += 4) {
            const brightness = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            stdDev += Math.pow(brightness - avgBrightness, 2);
        }
        stdDev = Math.sqrt(stdDev / pixelCount);
        
        // Calculate recommendations
        const analysis = {
            brightness: 0,
            contrast: 0,
            exposure: 0,
            highlights: 0,
            shadows: 0,
            vibrance: 0,
            saturation: 0,
            dehaze: 0,
            clarity: 0,
            temperature: 0,
            tint: 0
        };
        
        // Brightness and exposure adjustments
        if (avgBrightness < 85) {
            analysis.brightness = Math.min(30, (100 - avgBrightness) * 0.3);
            analysis.exposure = Math.min(50, (100 - avgBrightness) * 0.5);
            analysis.shadows = Math.min(40, (100 - avgBrightness) * 0.4);
        } else if (avgBrightness > 170) {
            analysis.brightness = Math.max(-30, (150 - avgBrightness) * 0.3);
            analysis.highlights = Math.max(-40, (150 - avgBrightness) * 0.4);
        }
        
        // Contrast adjustments based on standard deviation
        if (stdDev < 40) {
            analysis.contrast = Math.min(30, (50 - stdDev) * 0.8);
            analysis.clarity = 10;
        } else if (stdDev > 80) {
            analysis.contrast = Math.max(-20, (70 - stdDev) * 0.3);
        }
        
        // Saturation and vibrance adjustments
        if (avgSaturation < 0.3) {
            analysis.vibrance = Math.min(25, (0.4 - avgSaturation) * 50);
            analysis.saturation = Math.min(15, (0.4 - avgSaturation) * 30);
        } else if (avgSaturation > 0.7) {
            analysis.vibrance = Math.max(-15, (0.6 - avgSaturation) * 30);
        }
        
        // Color temperature and tint adjustments
        const greenMagentaBias = (avgG - (avgR + avgB) / 2) / 128;
        const blueYellowBias = (avgB - avgR) / 128;
        
        if (Math.abs(blueYellowBias) > 0.05) {
            analysis.temperature = -blueYellowBias * 30;
        }
        
        if (Math.abs(greenMagentaBias) > 0.05) {
            analysis.tint = -greenMagentaBias * 20;
        }
        
        // Check for haze (low contrast in midtones)
        const midtoneStart = Math.floor(histogram.length * 0.25);
        const midtoneEnd = Math.floor(histogram.length * 0.75);
        let midtoneCount = 0;
        
        for (let i = midtoneStart; i < midtoneEnd; i++) {
            midtoneCount += histogram[i];
        }
        
        if (midtoneCount > pixelCount * 0.7 && stdDev < 50) {
            analysis.dehaze = Math.min(20, (0.8 - midtoneCount / pixelCount) * 100);
        }
        
        // Round all values
        Object.keys(analysis).forEach(key => {
            analysis[key] = Math.round(analysis[key]);
        });
        
        return analysis;
    }
    
    updateStatus(message) {
        document.getElementById('statusMessage').textContent = message;
    }
    
    updateImageInfo(width, height) {
        document.getElementById('imageInfo').textContent = `${width} × ${height} pixels`;
    }
    
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('active');
        }
    }
    
    // Cleanup method to prevent memory leaks
    exportAllImages() {
        const thumbnails = document.querySelectorAll('.image-thumbnail img');
        if (thumbnails.length === 0) {
            this.updateStatus('No images to export');
            return;
        }
        
        this.updateStatus('Exporting all images...');
        
        thumbnails.forEach((img, index) => {
            setTimeout(() => {
                // Load and process each image
                this.loadImage(img.src);
                setTimeout(() => {
                    this.canvas.toBlob((blob) => {
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `edited_${img.dataset.filename || 'image_' + index + '.jpg'}`;
                        a.click();
                        URL.revokeObjectURL(url);
                    }, 'image/jpeg', 0.95);
                }, 100);
            }, index * 200);
        });
    }
    
    cleanup() {
        // Remove event listeners
        if (this.fileInput) {
            this.fileInput.removeEventListener('change', this.handleFileSelectBound);
        }
        
        // Clear WebGL resources
        if (this.gl) {
            // Delete shaders and programs
            Object.values(this.shaderPrograms).forEach(program => {
                if (program) {
                    this.gl.deleteProgram(program);
                }
            });
            
            // Clear buffers
            if (this.positionBuffer) this.gl.deleteBuffer(this.positionBuffer);
            if (this.texCoordBuffer) this.gl.deleteBuffer(this.texCoordBuffer);
            if (this.texture) this.gl.deleteTexture(this.texture);
            
            // Lose context
            const loseContext = this.gl.getExtension('WEBGL_lose_context');
            if (loseContext) {
                loseContext.loseContext();
            }
        }
        
        // Clear Web Worker
        if (this.worker) {
            this.worker.terminate();
        }
        
        // Clear timers
        Object.values(this.debounceTimers).forEach(timer => {
            clearTimeout(timer);
        });
        
        // Clear animation frame
        if (this.frameRequestId) {
            cancelAnimationFrame(this.frameRequestId);
        }
        
        // Clear references
        this.currentImage = null;
        this.originalImage = null;
        this.originalImageData = null;
        this.previewImage = null;
        this.undoStack = [];
        this.redoStack = [];
        this.imageHistory = {};
        
        console.log('Cleanup completed');
    }
}

// Helper function for modal closing
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

function performExport() {
    window.photoEditor.performExport();
}

// Initialize the optimized application
document.addEventListener('DOMContentLoaded', () => {
    window.photoEditor = new PhotoEditorOptimized();
    console.log('William & Mary Photo Editor Optimized - WebGL + Web Workers Enabled');
});

// Cleanup on page unload to prevent memory leaks
window.addEventListener('beforeunload', () => {
    if (window.photoEditor && typeof window.photoEditor.cleanup === 'function') {
        window.photoEditor.cleanup();
    }
});
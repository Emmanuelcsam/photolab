// William & Mary Photo Editor - Main Application JavaScript
// All photo editing functions from the original Python implementation

class PhotoEditor {
    constructor() {
        this.canvas = document.getElementById('imageCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.currentImage = null;
        this.originalImage = null;
        this.previewImage = null;
        this.currentAdjustments = {}; // Store current slider values
        this.undoStack = [];
        this.imageHistory = {};
        this.currentImagePath = null;
        this.zoomLevel = 1;
        this.isProcessing = false;
        
        this.initializeFunctions();
        this.initializeUI();
        this.setupEventListeners();
    }
    
    initializeFunctions() {
        // Define all photo editing functions matching the Python implementation
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
                        size: { type: 'slider', min: 1, max: 100, default: 25 }
                    }
                }
            },
            filters: {
                blur: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 99, default: 5 }
                    }
                },
                gaussian_blur: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 99, default: 5 },
                        sigma_x: { type: 'slider', min: 0, max: 50, default: 0 }
                    }
                },
                median_blur: {
                    params: {
                        kernel_size: { type: 'slider', min: 3, max: 99, default: 5 }
                    }
                },
                bilateral_filter: {
                    params: {
                        d: { type: 'slider', min: 1, max: 30, default: 9 },
                        sigma_color: { type: 'slider', min: 1, max: 200, default: 75 },
                        sigma_space: { type: 'slider', min: 1, max: 200, default: 75 }
                    }
                },
                motion_blur: {
                    params: {
                        size: { type: 'slider', min: 1, max: 100, default: 15 },
                        angle: { type: 'slider', min: 0, max: 360, default: 0 }
                    }
                },
                sharpen: {
                    params: {
                        amount: { type: 'slider', min: 0, max: 300, default: 100 }
                    }
                },
                unsharp_mask: {
                    params: {
                        radius: { type: 'slider', min: 0, max: 100, default: 1 },
                        amount: { type: 'slider', min: 0, max: 300, default: 100 },
                        threshold: { type: 'slider', min: 0, max: 255, default: 0 }
                    }
                },
                denoise: {
                    params: {
                        h: { type: 'slider', min: 1, max: 100, default: 10 },
                        template_window_size: { type: 'slider', min: 3, max: 21, default: 7 },
                        search_window_size: { type: 'slider', min: 7, max: 35, default: 21 }
                    }
                },
                emboss: {
                    params: {
                        strength: { type: 'slider', min: 0, max: 10, default: 1 }
                    }
                }
            },
            threshold: {
                threshold_binary: {
                    params: {
                        threshold_value: { type: 'slider', min: 0, max: 255, default: 127 },
                        max_value: { type: 'slider', min: 0, max: 255, default: 255 }
                    }
                },
                threshold_binary_inv: {
                    params: {
                        threshold_value: { type: 'slider', min: 0, max: 255, default: 127 },
                        max_value: { type: 'slider', min: 0, max: 255, default: 255 }
                    }
                },
                threshold_trunc: {
                    params: {
                        threshold_value: { type: 'slider', min: 0, max: 255, default: 127 }
                    }
                },
                threshold_tozero: {
                    params: {
                        threshold_value: { type: 'slider', min: 0, max: 255, default: 127 }
                    }
                },
                threshold_tozero_inv: {
                    params: {
                        threshold_value: { type: 'slider', min: 0, max: 255, default: 127 }
                    }
                },
                threshold_otsu: {
                    params: {}
                },
                adaptive_threshold_mean: {
                    params: {
                        block_size: { type: 'slider', min: 3, max: 99, default: 11 },
                        c: { type: 'slider', min: -50, max: 50, default: 2 }
                    }
                },
                adaptive_threshold_gaussian: {
                    params: {
                        block_size: { type: 'slider', min: 3, max: 99, default: 11 },
                        c: { type: 'slider', min: -50, max: 50, default: 2 }
                    }
                }
            },
            morphology: {
                erosion: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 3 },
                        iterations: { type: 'slider', min: 1, max: 10, default: 1 }
                    }
                },
                dilation: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 3 },
                        iterations: { type: 'slider', min: 1, max: 10, default: 1 }
                    }
                },
                opening: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 3 }
                    }
                },
                closing: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 3 }
                    }
                },
                morphology_gradient: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 3 }
                    }
                },
                tophat: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 9 }
                    }
                },
                blackhat: {
                    params: {
                        kernel_size: { type: 'slider', min: 1, max: 21, default: 9 }
                    }
                }
            },
            edge: {
                canny: {
                    params: {
                        threshold1: { type: 'slider', min: 0, max: 500, default: 100 },
                        threshold2: { type: 'slider', min: 0, max: 500, default: 200 }
                    }
                },
                sobel_x: {
                    params: {
                        kernel_size: { type: 'combo', values: [1, 3, 5, 7], default: 3 }
                    }
                },
                sobel_y: {
                    params: {
                        kernel_size: { type: 'combo', values: [1, 3, 5, 7], default: 3 }
                    }
                },
                sobel_combined: {
                    params: {
                        kernel_size: { type: 'combo', values: [1, 3, 5, 7], default: 3 }
                    }
                },
                laplacian: {
                    params: {
                        kernel_size: { type: 'combo', values: [1, 3, 5, 7], default: 1 }
                    }
                },
                scharr_x: {
                    params: {}
                },
                scharr_y: {
                    params: {}
                },
                prewitt: {
                    params: {}
                }
            },
            enhance: {
                histogram_equalization: {
                    params: {}
                },
                clahe: {
                    params: {
                        clip_limit: { type: 'slider', min: 1, max: 100, default: 2 },
                        tile_grid_size: { type: 'slider', min: 1, max: 50, default: 8 }
                    }
                },
                gamma_correction: {
                    params: {
                        gamma: { type: 'slider', min: 1, max: 500, default: 100 }
                    }
                },
                log_transformation: {
                    params: {
                        c: { type: 'slider', min: 1, max: 255, default: 1 }
                    }
                },
                power_law_transformation: {
                    params: {
                        gamma: { type: 'slider', min: 1, max: 500, default: 100 },
                        c: { type: 'slider', min: 1, max: 255, default: 1 }
                    }
                },
                auto_brightness_contrast: {
                    params: {
                        clip_percent: { type: 'slider', min: 0, max: 100, default: 1 }
                    }
                }
            },
            color: {
                hsv_adjustment: {
                    params: {
                        hue_shift: { type: 'slider', min: -180, max: 180, default: 0 },
                        saturation_scale: { type: 'slider', min: 0, max: 200, default: 100 },
                        value_scale: { type: 'slider', min: 0, max: 200, default: 100 }
                    }
                },
                rgb_adjustment: {
                    params: {
                        red_scale: { type: 'slider', min: 0, max: 200, default: 100 },
                        green_scale: { type: 'slider', min: 0, max: 200, default: 100 },
                        blue_scale: { type: 'slider', min: 0, max: 200, default: 100 }
                    }
                },
                color_balance: {
                    params: {
                        cyan_red: { type: 'slider', min: -100, max: 100, default: 0 },
                        magenta_green: { type: 'slider', min: -100, max: 100, default: 0 },
                        yellow_blue: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                channel_mixer: {
                    params: {
                        red_from_red: { type: 'slider', min: -200, max: 200, default: 100 },
                        red_from_green: { type: 'slider', min: -200, max: 200, default: 0 },
                        red_from_blue: { type: 'slider', min: -200, max: 200, default: 0 }
                    }
                },
                selective_color: {
                    params: {
                        target_hue: { type: 'slider', min: 0, max: 360, default: 0 },
                        hue_range: { type: 'slider', min: 1, max: 180, default: 30 },
                        adjustment: { type: 'slider', min: -100, max: 100, default: 50 }
                    }
                },
                color_lookup_table: {
                    params: {
                        preset: { type: 'combo', values: ['autumn', 'winter', 'spring', 'summer', 'vintage', 'cinematic'], default: 'autumn' }
                    }
                },
                split_toning: {
                    params: {
                        highlight_hue: { type: 'slider', min: 0, max: 360, default: 30 },
                        highlight_saturation: { type: 'slider', min: 0, max: 100, default: 20 },
                        shadow_hue: { type: 'slider', min: 0, max: 360, default: 220 },
                        shadow_saturation: { type: 'slider', min: 0, max: 100, default: 20 },
                        balance: { type: 'slider', min: -100, max: 100, default: 0 }
                    }
                },
                duotone: {
                    params: {
                        color1_hue: { type: 'slider', min: 0, max: 360, default: 30 },
                        color2_hue: { type: 'slider', min: 0, max: 360, default: 220 }
                    }
                }
            },
            artistic: {
                oil_painting: {
                    params: {
                        size: { type: 'slider', min: 1, max: 20, default: 7 },
                        dyn_ratio: { type: 'slider', min: 1, max: 10, default: 1 }
                    }
                },
                pencil_sketch: {
                    params: {
                        sigma_s: { type: 'slider', min: 1, max: 200, default: 60 },
                        sigma_r: { type: 'slider', min: 0, max: 100, default: 7 },
                        shade_factor: { type: 'slider', min: 0, max: 100, default: 2 }
                    }
                },
                stylization: {
                    params: {
                        sigma_s: { type: 'slider', min: 1, max: 200, default: 60 },
                        sigma_r: { type: 'slider', min: 0, max: 100, default: 6 }
                    }
                },
                edge_preserving_filter: {
                    params: {
                        flags: { type: 'combo', values: ['RECURS_FILTER', 'NORMCONV_FILTER'], default: 'RECURS_FILTER' },
                        sigma_s: { type: 'slider', min: 1, max: 200, default: 60 },
                        sigma_r: { type: 'slider', min: 0, max: 100, default: 4 }
                    }
                },
                cartoon: {
                    params: {
                        num_down: { type: 'slider', min: 1, max: 10, default: 2 },
                        num_bilateral: { type: 'slider', min: 1, max: 20, default: 7 }
                    }
                },
                anisotropic_diffusion: {
                    params: {
                        alpha: { type: 'slider', min: 0, max: 100, default: 10 },
                        k: { type: 'slider', min: 1, max: 100, default: 20 },
                        iterations: { type: 'slider', min: 1, max: 50, default: 10 }
                    }
                },
                watercolor: {
                    params: {
                        sigma_s: { type: 'slider', min: 1, max: 200, default: 60 },
                        sigma_r: { type: 'slider', min: 0, max: 100, default: 4 }
                    }
                },
                HDR_tone_mapping: {
                    params: {
                        gamma: { type: 'slider', min: 0, max: 500, default: 100 },
                        saturation: { type: 'slider', min: 0, max: 200, default: 100 }
                    }
                }
            }
        };
    }
    
    initializeUI() {
        // Build function panels
        Object.keys(this.functions).forEach(category => {
            const tabPane = document.getElementById(`${category}-tab`);
            if (!tabPane) return;
            
            Object.keys(this.functions[category]).forEach(funcName => {
                const func = this.functions[category][funcName];
                const widget = this.createFunctionWidget(funcName, func.params, category);
                tabPane.appendChild(widget);
            });
        });
        
        this.updateStatus('Ready');
    }
    
    createFunctionWidget(funcName, params, category) {
        const widget = document.createElement('div');
        widget.className = 'function-widget';
        
        // Header
        const header = document.createElement('div');
        header.className = 'function-header';
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'function-name';
        nameSpan.textContent = this.formatFunctionName(funcName);
        
        // Reset button for this function
        const resetBtn = document.createElement('button');
        resetBtn.className = 'reset-function-btn';
        resetBtn.innerHTML = '<i class="fas fa-undo"></i>';
        resetBtn.title = 'Reset to default';
        resetBtn.onclick = () => this.resetFunction(funcName, params, category);
        
        header.appendChild(nameSpan);
        header.appendChild(resetBtn);
        widget.appendChild(header);
        
        // Parameters
        const paramsDiv = document.createElement('div');
        paramsDiv.className = 'function-params';
        
        // Check if there are no parameters
        if (Object.keys(params).length === 0) {
            // Add an apply button for functions with no parameters
            const applyBtn = document.createElement('button');
            applyBtn.className = 'btn-primary';
            applyBtn.style.width = '100%';
            applyBtn.style.marginTop = '0.5rem';
            applyBtn.textContent = 'Apply ' + this.formatFunctionName(funcName);
            applyBtn.onclick = () => {
                // For functions with no parameters, apply directly
                if (!this.currentImage) {
                    this.updateStatus('Please load an image first');
                    return;
                }
                
                // Apply the effect
                const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
                const adjustment = {
                    category: category,
                    function: funcName,
                    params: {}
                };
                this.applyEffect(imageData, adjustment);
                this.ctx.putImageData(imageData, 0, 0);
                
                // Save to undo stack
                this.undoStack.push({...this.currentAdjustments});
                
                this.updateStatus(`Applied ${this.formatFunctionName(funcName)}`);
            };
            paramsDiv.appendChild(applyBtn);
        }
        
        Object.keys(params).forEach(paramName => {
            const param = params[paramName];
            const paramGroup = document.createElement('div');
            paramGroup.className = 'param-group';
            
            const label = document.createElement('label');
            label.className = 'param-label';
            label.textContent = this.formatParamName(paramName);
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
                slider.id = `${funcName}_${paramName}`;
                
                const valueInput = document.createElement('input');
                valueInput.type = 'number';
                valueInput.className = 'param-value-input';
                valueInput.min = param.min;
                valueInput.max = param.max;
                valueInput.value = param.default;
                valueInput.id = `${funcName}_${paramName}_value`;
                
                // Initialize adjustment value
                this.currentAdjustments[adjustmentKey] = param.default;
                
                slider.oninput = () => {
                    valueInput.value = slider.value;
                    this.currentAdjustments[adjustmentKey] = parseFloat(slider.value);
                    this.applyAllAdjustments();
                };
                
                valueInput.oninput = () => {
                    let val = parseFloat(valueInput.value);
                    if (isNaN(val)) val = param.default;
                    val = Math.max(param.min, Math.min(param.max, val));
                    valueInput.value = val;
                    slider.value = val;
                    this.currentAdjustments[adjustmentKey] = val;
                    this.applyAllAdjustments();
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
                
                select.onchange = () => {
                    this.currentAdjustments[adjustmentKey] = select.value;
                    this.applyAllAdjustments();
                };
                
                control.appendChild(select);
            } else if (param.type === 'checkbox') {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `${funcName}_${paramName}`;
                checkbox.checked = param.default;
                
                // Initialize adjustment value
                this.currentAdjustments[adjustmentKey] = param.default;
                
                checkbox.onchange = () => {
                    this.currentAdjustments[adjustmentKey] = checkbox.checked;
                    this.applyAllAdjustments();
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
        
        // Open Directory button (uses file input with webkitdirectory for folder selection)
        document.getElementById('openDirectoryBtn').addEventListener('click', () => {
            // Create a temporary input for directory selection
            const dirInput = document.createElement('input');
            dirInput.type = 'file';
            dirInput.multiple = true;
            dirInput.accept = 'image/*';
            dirInput.webkitdirectory = true;
            dirInput.directory = true;
            
            dirInput.addEventListener('change', (e) => {
                this.handleFileSelect(e.target.files);
            });
            
            // Fallback to regular multi-file selection if directory selection not supported
            dirInput.addEventListener('click', (e) => {
                // If webkitdirectory is not supported, use regular file input
                if (!('webkitdirectory' in dirInput)) {
                    e.preventDefault();
                    document.getElementById('fileInput').click();
                }
            });
            
            dirInput.click();
        });
        
        // Export All button
        document.getElementById('exportAllBtn').addEventListener('click', () => this.exportAllImages());
        
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.switchTab(btn.dataset.tab);
            });
        });
        
        // Control buttons
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetImage());
        document.getElementById('exportBtn').addEventListener('click', () => this.showExportModal());
        document.getElementById('saveEditsBtn').addEventListener('click', () => this.savePreset());
        document.getElementById('loadEditsBtn').addEventListener('click', () => this.loadPreset());
        document.getElementById('automaticModeBtn').addEventListener('click', () => this.applyAutomaticMode());
        
        // Zoom controls
        document.getElementById('zoomInBtn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoomOutBtn').addEventListener('click', () => this.zoomOut());
        document.getElementById('fitBtn').addEventListener('click', () => this.fitToCanvas());
        
        // Export modal
        document.getElementById('exportQuality').addEventListener('input', (e) => {
            document.getElementById('qualityValue').textContent = e.target.value + '%';
        });
        
        // Canvas mouse events for panning
        this.setupCanvasInteraction();
    }
    
    setupCanvasInteraction() {
        let isPanning = false;
        let startX, startY;
        
        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                isPanning = true;
                startX = e.clientX;
                startY = e.clientY;
                this.canvas.style.cursor = 'grabbing';
            }
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (isPanning) {
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                // Implement panning logic here
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            isPanning = false;
            this.canvas.style.cursor = 'grab';
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoom(delta);
        });
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // Update tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.toggle('active', pane.id === `${tabName}-tab`);
        });
    }
    
    handleFileSelect(files) {
        const gallery = document.getElementById('imageGallery');
        gallery.innerHTML = '';
        
        Array.from(files).forEach(file => {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    this.addImageToGallery(e.target.result, file.name);
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    addImageToGallery(src, name) {
        const gallery = document.getElementById('imageGallery');
        
        const thumbnail = document.createElement('div');
        thumbnail.className = 'image-thumbnail';
        thumbnail.onclick = () => this.loadImage(src, name);
        
        const img = document.createElement('img');
        img.src = src;
        img.alt = name;
        
        thumbnail.appendChild(img);
        gallery.appendChild(thumbnail);
        
        // Auto-load first image
        if (gallery.children.length === 1) {
            this.loadImage(src, name);
            thumbnail.classList.add('active');
        }
    }
    
    loadImage(src, name) {
        const img = new Image();
        img.onload = () => {
            this.currentImage = img;
            this.originalImage = img;
            this.currentImagePath = name;
            this.undoStack = [];
            
            // Store original image data for quick resets
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = img.width;
            tempCanvas.height = img.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(img, 0, 0);
            this.originalImageData = tempCtx.getImageData(0, 0, img.width, img.height);
            
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            this.ctx.drawImage(img, 0, 0);
            
            this.fitToCanvas();
            this.updateStatus(`Loaded: ${name}`);
            this.updateImageInfo(img.width, img.height);
            
            // Update gallery selection
            document.querySelectorAll('.image-thumbnail').forEach(thumb => {
                thumb.classList.toggle('active', thumb.querySelector('img').src === src);
            });
            
            // Show canvas, hide placeholder
            this.canvas.style.display = 'block';
            document.querySelector('.canvas-placeholder').style.display = 'none';
        };
        img.src = src;
    }
    
    applyAllAdjustments() {
        if (!this.currentImage || this.isProcessing) return;
        
        this.isProcessing = true;
        
        // Use requestAnimationFrame for smoother updates
        requestAnimationFrame(() => {
            // Start with original image data
            const imageData = new ImageData(
                new Uint8ClampedArray(this.originalImageData.data),
                this.originalImageData.width,
                this.originalImageData.height
            );
            
            // Group adjustments by category and function for efficient processing
            const adjustmentGroups = {};
            
            Object.entries(this.currentAdjustments).forEach(([key, value]) => {
                // Split key more carefully to handle underscores in function/param names
                const parts = key.split('_');
                const category = parts[0];
                let funcName, paramName;
                
                // Find where function name ends and param name begins
                // Try to match with actual function definitions
                for (let i = 1; i < parts.length; i++) {
                    const possibleFunc = parts.slice(1, i + 1).join('_');
                    const possibleParam = parts.slice(i + 1).join('_');
                    
                    if (this.functions[category] && 
                        this.functions[category][possibleFunc] && 
                        this.functions[category][possibleFunc].params[possibleParam]) {
                        funcName = possibleFunc;
                        paramName = possibleParam;
                        break;
                    }
                }
                
                if (!funcName || !paramName) return;
                
                const groupKey = `${category}_${funcName}`;
                
                if (!adjustmentGroups[groupKey]) {
                    adjustmentGroups[groupKey] = {
                        category: category,
                        function: funcName,
                        params: {}
                    };
                }
                
                // Check if value is different from default
                const defaultValue = this.getDefaultValue(category, funcName, paramName);
                if (value !== defaultValue && value != defaultValue) {
                    adjustmentGroups[groupKey].params[paramName] = value;
                }
            });
            
            // Apply only non-default adjustments
            Object.values(adjustmentGroups).forEach(adjustment => {
                if (Object.keys(adjustment.params).length > 0) {
                    this.applyEffect(imageData, adjustment);
                }
            });
            
            // Update canvas
            this.ctx.putImageData(imageData, 0, 0);
            this.isProcessing = false;
        });
    }
    
    getDefaultValue(category, funcName, paramName) {
        if (this.functions[category] && 
            this.functions[category][funcName] && 
            this.functions[category][funcName].params[paramName]) {
            return this.functions[category][funcName].params[paramName].default;
        }
        return null;
    }
    
    applyEffect(imageData, edit) {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        
        // Implement effects based on category and function
        switch(edit.category) {
            case 'basic':
                this.applyBasicAdjustment(data, width, height, edit.function, edit.params);
                break;
            case 'filters':
                this.applyFilter(data, width, height, edit.function, edit.params);
                break;
            case 'color':
                this.applyColorAdjustment(data, edit.function, edit.params);
                break;
            case 'threshold':
                this.applyThreshold(data, width, height, edit.function, edit.params);
                break;
            case 'morphology':
                this.applyMorphology(data, width, height, edit.function, edit.params);
                break;
            case 'edge':
                this.applyEdgeDetection(data, width, height, edit.function, edit.params);
                break;
            case 'enhance':
                this.applyEnhancement(data, width, height, edit.function, edit.params);
                break;
            case 'artistic':
                this.applyArtistic(data, width, height, edit.function, edit.params);
                break;
        }
    }
    
    applyBasicAdjustment(data, width, height, funcName, params) {
        
        for (let i = 0; i < data.length; i += 4) {
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            
            switch(funcName) {
                case 'brightness':
                    const brightness = parseFloat(params.brightness) || 0;
                    r = Math.min(255, Math.max(0, r + brightness));
                    g = Math.min(255, Math.max(0, g + brightness));
                    b = Math.min(255, Math.max(0, b + brightness));
                    break;
                    
                case 'contrast':
                    const contrast = (parseFloat(params.contrast) || 0);
                    const factor = (259 * (contrast + 255)) / (255 * (259 - contrast));
                    r = Math.min(255, Math.max(0, factor * (r - 128) + 128));
                    g = Math.min(255, Math.max(0, factor * (g - 128) + 128));
                    b = Math.min(255, Math.max(0, factor * (b - 128) + 128));
                    break;
                    
                case 'exposure':
                    const exposure = parseFloat(params.exposure) || 0;
                    const expFactor = Math.pow(2, exposure / 100);
                    r = Math.min(255, Math.max(0, r * expFactor));
                    g = Math.min(255, Math.max(0, g * expFactor));
                    b = Math.min(255, Math.max(0, b * expFactor));
                    break;
                    
                case 'saturation':
                    const saturation = (parseFloat(params.saturation) || 0) / 100;
                    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    r = Math.min(255, Math.max(0, gray + (1 + saturation) * (r - gray)));
                    g = Math.min(255, Math.max(0, gray + (1 + saturation) * (g - gray)));
                    b = Math.min(255, Math.max(0, gray + (1 + saturation) * (b - gray)));
                    break;
                    
                case 'vibrance':
                    const vibrance = (parseFloat(params.vibrance) || 0) / 100;
                    const avg = (r + g + b) / 3;
                    const mx = Math.max(r, g, b);
                    const amt = ((Math.abs(mx - avg) * 2 / 255) * -vibrance) * 0.8;
                    if (r !== mx) r = Math.min(255, Math.max(0, r + (mx - r) * amt));
                    if (g !== mx) g = Math.min(255, Math.max(0, g + (mx - g) * amt));
                    if (b !== mx) b = Math.min(255, Math.max(0, b + (mx - b) * amt));
                    break;
                    
                case 'temperature':
                    const temp = (parseFloat(params.temperature) || 0) / 100;
                    r = Math.min(255, Math.max(0, r + temp * 30));
                    b = Math.min(255, Math.max(0, b - temp * 30));
                    break;
                    
                case 'tint':
                    const tint = (parseFloat(params.tint) || 0) / 100;
                    g = Math.min(255, Math.max(0, g + tint * 30));
                    break;
                    
                case 'highlights':
                    const highlights = (parseFloat(params.highlights) || 0) / 100;
                    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
                    if (luminance > 180) {
                        const factor = 1 + highlights * 0.5;
                        r = Math.min(255, Math.max(0, r * factor));
                        g = Math.min(255, Math.max(0, g * factor));
                        b = Math.min(255, Math.max(0, b * factor));
                    }
                    break;
                    
                case 'shadows':
                    const shadows = (parseFloat(params.shadows) || 0) / 100;
                    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    if (lum < 80) {
                        const factor = 1 + shadows * 0.5;
                        r = Math.min(255, Math.max(0, r * factor));
                        g = Math.min(255, Math.max(0, g * factor));
                        b = Math.min(255, Math.max(0, b * factor));
                    }
                    break;
                    
                case 'whites':
                    const whites = (parseFloat(params.whites) || 0) / 100;
                    if (r > 200 || g > 200 || b > 200) {
                        r = Math.min(255, Math.max(0, r + whites * 50));
                        g = Math.min(255, Math.max(0, g + whites * 50));
                        b = Math.min(255, Math.max(0, b + whites * 50));
                    }
                    break;
                    
                case 'blacks':
                    const blacks = (parseFloat(params.blacks) || 0) / 100;
                    if (r < 50 || g < 50 || b < 50) {
                        r = Math.min(255, Math.max(0, r + blacks * 50));
                        g = Math.min(255, Math.max(0, g + blacks * 50));
                        b = Math.min(255, Math.max(0, b + blacks * 50));
                    }
                    break;
                    
                case 'clarity':
                    // Simple clarity effect using local contrast
                    const clarity = (parseFloat(params.clarity) || 0) / 100;
                    const localContrast = 1 + clarity * 0.5;
                    const mid = (r + g + b) / 3;
                    r = Math.min(255, Math.max(0, mid + (r - mid) * localContrast));
                    g = Math.min(255, Math.max(0, mid + (g - mid) * localContrast));
                    b = Math.min(255, Math.max(0, mid + (b - mid) * localContrast));
                    break;
                    
                case 'dehaze':
                    const amount = (parseFloat(params.amount) || 0) / 100;
                    // Simple dehaze by increasing contrast and reducing atmospheric haze
                    const dehazeFactor = 1 + amount * 0.3;
                    r = Math.min(255, Math.max(0, (r - 128) * dehazeFactor + 128));
                    g = Math.min(255, Math.max(0, (g - 128) * dehazeFactor + 128));
                    b = Math.min(255, Math.max(0, (b - 128) * dehazeFactor + 128));
                    break;
                    
                case 'vignette':
                    const vignetteAmount = (parseFloat(params.amount) || 0) / 100;
                    const midpoint = (parseFloat(params.midpoint) || 50) / 100;
                    // Calculate distance from center
                    const pixelIndex = i / 4;
                    const x = pixelIndex % width;
                    const y = Math.floor(pixelIndex / width);
                    const centerX = width / 2;
                    const centerY = height / 2;
                    const maxDist = Math.sqrt(centerX * centerX + centerY * centerY);
                    const dist = Math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                    const vignetteFactor = 1 - (vignetteAmount * Math.pow(dist / maxDist, 2 - midpoint));
                    r = Math.min(255, Math.max(0, r * vignetteFactor));
                    g = Math.min(255, Math.max(0, g * vignetteFactor));
                    b = Math.min(255, Math.max(0, b * vignetteFactor));
                    break;
                    
                case 'grain':
                    const grainAmount = (parseFloat(params.amount) || 0) / 100;
                    const grainSize = (parseFloat(params.size) || 25) / 100;
                    if (grainAmount > 0) {
                        const noise = (Math.random() - 0.5) * grainAmount * 50 * grainSize;
                        r = Math.min(255, Math.max(0, r + noise));
                        g = Math.min(255, Math.max(0, g + noise));
                        b = Math.min(255, Math.max(0, b + noise));
                    }
                    break;
            }
            
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
    }
    
    applyThreshold(data, width, height, funcName, params) {
        for (let i = 0; i < data.length; i += 4) {
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            let value = 0;
            
            switch(funcName) {
                case 'threshold_binary':
                    const threshold = parseFloat(params.threshold_value) || 127;
                    const maxVal = parseFloat(params.max_value) || 255;
                    value = gray > threshold ? maxVal : 0;
                    data[i] = data[i + 1] = data[i + 2] = value;
                    break;
                    
                case 'threshold_binary_inv':
                    const thresholdInv = parseFloat(params.threshold_value) || 127;
                    const maxValInv = parseFloat(params.max_value) || 255;
                    value = gray > thresholdInv ? 0 : maxValInv;
                    data[i] = data[i + 1] = data[i + 2] = value;
                    break;
                    
                case 'threshold_otsu':
                    // Simple Otsu's method approximation
                    value = gray > 128 ? 255 : 0;
                    data[i] = data[i + 1] = data[i + 2] = value;
                    break;
            }
        }
    }
    
    applyMorphology(data, width, height, funcName, params) {
        const kernelSize = parseInt(params.kernel_size) || 3;
        const iterations = parseInt(params.iterations) || 1;
        const tempData = new Uint8ClampedArray(data);
        
        for (let iter = 0; iter < iterations; iter++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = (y * width + x) * 4;
                    
                    switch(funcName) {
                        case 'erosion':
                            // Find minimum in kernel
                            let minVal = 255;
                            for (let ky = -kernelSize; ky <= kernelSize; ky++) {
                                for (let kx = -kernelSize; kx <= kernelSize; kx++) {
                                    const px = Math.min(width - 1, Math.max(0, x + kx));
                                    const py = Math.min(height - 1, Math.max(0, y + ky));
                                    const kidx = (py * width + px) * 4;
                                    const gray = 0.299 * tempData[kidx] + 0.587 * tempData[kidx + 1] + 0.114 * tempData[kidx + 2];
                                    minVal = Math.min(minVal, gray);
                                }
                            }
                            data[idx] = data[idx + 1] = data[idx + 2] = minVal;
                            break;
                            
                        case 'dilation':
                            // Find maximum in kernel
                            let maxVal = 0;
                            for (let ky = -kernelSize; ky <= kernelSize; ky++) {
                                for (let kx = -kernelSize; kx <= kernelSize; kx++) {
                                    const px = Math.min(width - 1, Math.max(0, x + kx));
                                    const py = Math.min(height - 1, Math.max(0, y + ky));
                                    const kidx = (py * width + px) * 4;
                                    const gray = 0.299 * tempData[kidx] + 0.587 * tempData[kidx + 1] + 0.114 * tempData[kidx + 2];
                                    maxVal = Math.max(maxVal, gray);
                                }
                            }
                            data[idx] = data[idx + 1] = data[idx + 2] = maxVal;
                            break;
                    }
                }
            }
            if (iter < iterations - 1) {
                tempData.set(data);
            }
        }
    }
    
    applyEdgeDetection(data, width, height, funcName, params) {
        const tempData = new Uint8ClampedArray(data);
        
        switch(funcName) {
            case 'canny':
                const threshold1 = parseFloat(params.threshold1) || 100;
                const threshold2 = parseFloat(params.threshold2) || 200;
                
                // Simple edge detection (Sobel approximation)
                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        const idx = (y * width + x) * 4;
                        
                        // Sobel X
                        let gx = 0;
                        gx -= tempData[((y - 1) * width + (x - 1)) * 4];
                        gx += tempData[((y - 1) * width + (x + 1)) * 4];
                        gx -= 2 * tempData[(y * width + (x - 1)) * 4];
                        gx += 2 * tempData[(y * width + (x + 1)) * 4];
                        gx -= tempData[((y + 1) * width + (x - 1)) * 4];
                        gx += tempData[((y + 1) * width + (x + 1)) * 4];
                        
                        // Sobel Y
                        let gy = 0;
                        gy -= tempData[((y - 1) * width + (x - 1)) * 4];
                        gy -= 2 * tempData[((y - 1) * width + x) * 4];
                        gy -= tempData[((y - 1) * width + (x + 1)) * 4];
                        gy += tempData[((y + 1) * width + (x - 1)) * 4];
                        gy += 2 * tempData[((y + 1) * width + x) * 4];
                        gy += tempData[((y + 1) * width + (x + 1)) * 4];
                        
                        const magnitude = Math.sqrt(gx * gx + gy * gy);
                        const edge = magnitude > threshold1 ? 255 : 0;
                        
                        data[idx] = data[idx + 1] = data[idx + 2] = edge;
                    }
                }
                break;
                
            case 'sobel_x':
            case 'sobel_y':
            case 'sobel_combined':
                // Simplified Sobel implementation
                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        const idx = (y * width + x) * 4;
                        let gradient = 0;
                        
                        if (funcName === 'sobel_x' || funcName === 'sobel_combined') {
                            // Horizontal gradient
                            gradient += Math.abs(
                                -tempData[((y - 1) * width + (x - 1)) * 4] +
                                tempData[((y - 1) * width + (x + 1)) * 4] +
                                -2 * tempData[(y * width + (x - 1)) * 4] +
                                2 * tempData[(y * width + (x + 1)) * 4] +
                                -tempData[((y + 1) * width + (x - 1)) * 4] +
                                tempData[((y + 1) * width + (x + 1)) * 4]
                            );
                        }
                        
                        if (funcName === 'sobel_y' || funcName === 'sobel_combined') {
                            // Vertical gradient
                            gradient += Math.abs(
                                -tempData[((y - 1) * width + (x - 1)) * 4] +
                                -2 * tempData[((y - 1) * width + x) * 4] +
                                -tempData[((y - 1) * width + (x + 1)) * 4] +
                                tempData[((y + 1) * width + (x - 1)) * 4] +
                                2 * tempData[((y + 1) * width + x) * 4] +
                                tempData[((y + 1) * width + (x + 1)) * 4]
                            );
                        }
                        
                        gradient = Math.min(255, gradient);
                        data[idx] = data[idx + 1] = data[idx + 2] = gradient;
                    }
                }
                break;
        }
    }
    
    applyEnhancement(data, width, height, funcName, params) {
        switch(funcName) {
            case 'histogram_equalization':
                // Calculate histogram
                const histogram = new Array(256).fill(0);
                for (let i = 0; i < data.length; i += 4) {
                    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
                    histogram[gray]++;
                }
                
                // Calculate CDF
                const cdf = new Array(256);
                cdf[0] = histogram[0];
                for (let i = 1; i < 256; i++) {
                    cdf[i] = cdf[i - 1] + histogram[i];
                }
                
                // Normalize CDF
                const cdfMin = cdf.find(val => val > 0);
                const totalPixels = width * height;
                const mapping = cdf.map(val => Math.round(((val - cdfMin) / (totalPixels - cdfMin)) * 255));
                
                // Apply mapping
                for (let i = 0; i < data.length; i += 4) {
                    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
                    const ratio = mapping[gray] / (gray || 1);
                    data[i] = Math.min(255, data[i] * ratio);
                    data[i + 1] = Math.min(255, data[i + 1] * ratio);
                    data[i + 2] = Math.min(255, data[i + 2] * ratio);
                }
                break;
                
            case 'gamma_correction':
                const gamma = (parseFloat(params.gamma) || 100) / 100;
                const invGamma = 1.0 / gamma;
                
                for (let i = 0; i < data.length; i += 4) {
                    data[i] = Math.pow(data[i] / 255, invGamma) * 255;
                    data[i + 1] = Math.pow(data[i + 1] / 255, invGamma) * 255;
                    data[i + 2] = Math.pow(data[i + 2] / 255, invGamma) * 255;
                }
                break;
        }
    }
    
    applyArtistic(data, width, height, funcName, params) {
        switch(funcName) {
            case 'oil_painting':
                const size = parseInt(params.size) || 7;
                const tempData = new Uint8ClampedArray(data);
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = (y * width + x) * 4;
                        const intensityCount = new Array(256).fill(0);
                        const avgR = new Array(256).fill(0);
                        const avgG = new Array(256).fill(0);
                        const avgB = new Array(256).fill(0);
                        
                        // Sample neighborhood
                        for (let ky = -size; ky <= size; ky++) {
                            for (let kx = -size; kx <= size; kx++) {
                                const px = Math.min(width - 1, Math.max(0, x + kx));
                                const py = Math.min(height - 1, Math.max(0, y + ky));
                                const kidx = (py * width + px) * 4;
                                
                                const intensity = Math.round((tempData[kidx] + tempData[kidx + 1] + tempData[kidx + 2]) / 3);
                                intensityCount[intensity]++;
                                avgR[intensity] += tempData[kidx];
                                avgG[intensity] += tempData[kidx + 1];
                                avgB[intensity] += tempData[kidx + 2];
                            }
                        }
                        
                        // Find most frequent intensity
                        let maxCount = 0;
                        let maxIntensity = 0;
                        for (let i = 0; i < 256; i++) {
                            if (intensityCount[i] > maxCount) {
                                maxCount = intensityCount[i];
                                maxIntensity = i;
                            }
                        }
                        
                        data[idx] = avgR[maxIntensity] / maxCount;
                        data[idx + 1] = avgG[maxIntensity] / maxCount;
                        data[idx + 2] = avgB[maxIntensity] / maxCount;
                    }
                }
                break;
                
            case 'pencil_sketch':
                // Convert to grayscale and invert
                for (let i = 0; i < data.length; i += 4) {
                    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    const inverted = 255 - gray;
                    // Apply some contrast
                    const sketch = Math.min(255, Math.max(0, inverted * 1.5 - 100));
                    data[i] = data[i + 1] = data[i + 2] = sketch;
                }
                break;
        }
    }
    
    applyFilter(data, width, height, funcName, params) {
        // Implement filter effects
        switch(funcName) {
            case 'blur':
                // Simple box blur implementation
                const kernelSize = parseInt(params.kernel_size);
                const tempData = new Uint8ClampedArray(data);
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        let r = 0, g = 0, b = 0, count = 0;
                        
                        for (let ky = -kernelSize; ky <= kernelSize; ky++) {
                            for (let kx = -kernelSize; kx <= kernelSize; kx++) {
                                const px = Math.min(width - 1, Math.max(0, x + kx));
                                const py = Math.min(height - 1, Math.max(0, y + ky));
                                const idx = (py * width + px) * 4;
                                
                                r += tempData[idx];
                                g += tempData[idx + 1];
                                b += tempData[idx + 2];
                                count++;
                            }
                        }
                        
                        const idx = (y * width + x) * 4;
                        data[idx] = r / count;
                        data[idx + 1] = g / count;
                        data[idx + 2] = b / count;
                    }
                }
                break;
                
            // Add more filters
        }
    }
    
    applyColorAdjustment(data, funcName, params) {
        for (let i = 0; i < data.length; i += 4) {
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            
            switch(funcName) {
                case 'hsv_adjustment':
                    const hueShift = parseFloat(params.hue_shift) || 0;
                    const satScale = (parseFloat(params.saturation_scale) || 100) / 100;
                    const valScale = (parseFloat(params.value_scale) || 100) / 100;
                    
                    // Convert RGB to HSV
                    const max = Math.max(r, g, b) / 255;
                    const min = Math.min(r, g, b) / 255;
                    const delta = max - min;
                    
                    let h = 0, s = 0, v = max;
                    
                    if (delta !== 0) {
                        s = delta / max;
                        
                        if (max === r / 255) {
                            h = ((g - b) / 255 / delta + (g < b ? 6 : 0)) * 60;
                        } else if (max === g / 255) {
                            h = ((b - r) / 255 / delta + 2) * 60;
                        } else {
                            h = ((r - g) / 255 / delta + 4) * 60;
                        }
                    }
                    
                    // Apply adjustments
                    h = (h + hueShift + 360) % 360;
                    s = Math.min(1, Math.max(0, s * satScale));
                    v = Math.min(1, Math.max(0, v * valScale));
                    
                    // Convert back to RGB
                    const c = v * s;
                    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
                    const m = v - c;
                    
                    let r1 = 0, g1 = 0, b1 = 0;
                    
                    if (h < 60) { r1 = c; g1 = x; b1 = 0; }
                    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
                    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
                    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
                    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
                    else { r1 = c; g1 = 0; b1 = x; }
                    
                    r = Math.round((r1 + m) * 255);
                    g = Math.round((g1 + m) * 255);
                    b = Math.round((b1 + m) * 255);
                    break;
                    
                case 'rgb_adjustment':
                    const redScale = (parseFloat(params.red_scale) || 100) / 100;
                    const greenScale = (parseFloat(params.green_scale) || 100) / 100;
                    const blueScale = (parseFloat(params.blue_scale) || 100) / 100;
                    
                    r = Math.min(255, Math.max(0, r * redScale));
                    g = Math.min(255, Math.max(0, g * greenScale));
                    b = Math.min(255, Math.max(0, b * blueScale));
                    break;
                    
                case 'color_balance':
                    const cyanRed = (parseFloat(params.cyan_red) || 0) / 100;
                    const magentaGreen = (parseFloat(params.magenta_green) || 0) / 100;
                    const yellowBlue = (parseFloat(params.yellow_blue) || 0) / 100;
                    
                    r = Math.min(255, Math.max(0, r + cyanRed * 30));
                    g = Math.min(255, Math.max(0, g + magentaGreen * 30));
                    b = Math.min(255, Math.max(0, b + yellowBlue * 30));
                    break;
                    
                case 'split_toning':
                    const highlightHue = parseFloat(params.highlight_hue) || 30;
                    const highlightSat = (parseFloat(params.highlight_saturation) || 20) / 100;
                    const shadowHue = parseFloat(params.shadow_hue) || 220;
                    const shadowSat = (parseFloat(params.shadow_saturation) || 20) / 100;
                    const balance = (parseFloat(params.balance) || 0) / 100;
                    
                    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
                    const isHighlight = luminance > 128;
                    
                    if (isHighlight) {
                        // Apply highlight toning
                        const tintR = Math.cos(highlightHue * Math.PI / 180) * highlightSat * 50;
                        const tintG = Math.sin(highlightHue * Math.PI / 180) * highlightSat * 50;
                        r = Math.min(255, Math.max(0, r + tintR * (1 + balance)));
                        g = Math.min(255, Math.max(0, g + tintG * (1 + balance)));
                    } else {
                        // Apply shadow toning
                        const tintR = Math.cos(shadowHue * Math.PI / 180) * shadowSat * 50;
                        const tintB = Math.sin(shadowHue * Math.PI / 180) * shadowSat * 50;
                        r = Math.min(255, Math.max(0, r + tintR * (1 - balance)));
                        b = Math.min(255, Math.max(0, b + tintB * (1 - balance)));
                    }
                    break;
            }
            
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
    }
    
    undo() {
        if (!this.currentImage) return;
        
        // Save current adjustments to undo stack
        this.undoStack.push({...this.currentAdjustments});
        
        // Reset all adjustments to default
        this.resetAllAdjustments();
        this.updateStatus('Undo successful');
    }
    
    resetImage() {
        if (!this.currentImage) return;
        
        // Reset all adjustments to default values
        this.resetAllAdjustments();
        this.updateStatus('Image reset to original');
    }
    
    resetAllAdjustments() {
        // Reset all sliders and controls to default values
        Object.keys(this.functions).forEach(category => {
            Object.keys(this.functions[category]).forEach(funcName => {
                const params = this.functions[category][funcName].params;
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
            });
        });
        
        // Redraw original image
        this.ctx.putImageData(this.originalImageData, 0, 0);
    }
    
    zoomIn() {
        this.setZoom(this.zoomLevel * 1.25);
    }
    
    zoomOut() {
        this.setZoom(this.zoomLevel * 0.8);
    }
    
    zoom(factor) {
        this.setZoom(this.zoomLevel * factor);
    }
    
    setZoom(newZoom) {
        this.zoomLevel = Math.min(5, Math.max(0.1, newZoom));
        document.getElementById('zoomLevel').textContent = Math.round(this.zoomLevel * 100) + '%';
        
        this.canvas.style.transform = `scale(${this.zoomLevel})`;
        this.canvas.style.transformOrigin = 'center center';
    }
    
    fitToCanvas() {
        if (!this.currentImage) return;
        
        const container = document.querySelector('.canvas-container');
        const containerWidth = container.clientWidth - 40; // Account for padding
        const containerHeight = container.clientHeight - 40;
        const scaleX = containerWidth / this.canvas.width;
        const scaleY = containerHeight / this.canvas.height;
        const scale = Math.min(scaleX, scaleY, 1);
        
        this.setZoom(scale);
    }
    
    exportAllImages() {
        const gallery = document.getElementById('imageGallery');
        const images = gallery.querySelectorAll('.image-thumbnail img');
        
        if (images.length === 0) {
            this.updateStatus('No images to export');
            return;
        }
        
        // Create a zip file or export images one by one
        let exportCount = 0;
        images.forEach((img, index) => {
            setTimeout(() => {
                // Load image
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                const tempImg = new Image();
                
                tempImg.onload = () => {
                    tempCanvas.width = tempImg.width;
                    tempCanvas.height = tempImg.height;
                    tempCtx.drawImage(tempImg, 0, 0);
                    
                    // Apply saved edits if any
                    const imageName = img.alt || `image_${index}`;
                    if (this.imageHistory[imageName]) {
                        // Apply edits from history
                        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                        this.imageHistory[imageName].forEach(edit => {
                            this.applyEffect(imageData, edit);
                        });
                        tempCtx.putImageData(imageData, 0, 0);
                    }
                    
                    // Export
                    const link = document.createElement('a');
                    link.download = `edited_${imageName}`;
                    link.href = tempCanvas.toDataURL('image/jpeg', 0.95);
                    link.click();
                    
                    exportCount++;
                    if (exportCount === images.length) {
                        this.updateStatus(`Exported ${exportCount} images`);
                    }
                };
                
                tempImg.src = img.src;
            }, index * 500); // Stagger exports to prevent browser blocking
        });
    }
    
    showExportModal() {
        document.getElementById('exportModal').classList.add('active');
    }
    
    performExport() {
        const fileName = document.getElementById('exportFileName').value;
        const format = document.getElementById('exportFormat').value;
        const quality = document.getElementById('exportQuality').value / 100;
        
        const link = document.createElement('a');
        link.download = fileName;
        link.href = this.canvas.toDataURL(`image/${format}`, quality);
        link.click();
        
        this.closeModal('exportModal');
        this.updateStatus(`Exported: ${fileName}`);
    }
    
    savePreset() {
        // Only save non-default adjustments
        const preset = {};
        
        Object.entries(this.currentAdjustments).forEach(([key, value]) => {
            const [category, funcName, paramName] = key.split('_');
            const defaultValue = this.getDefaultValue(category, funcName, paramName);
            
            if (value !== defaultValue && value != defaultValue) {
                preset[key] = value;
            }
        });
        
        const presetData = {
            adjustments: preset,
            timestamp: new Date().toISOString(),
            imageInfo: {
                width: this.canvas.width,
                height: this.canvas.height
            }
        };
        
        const blob = new Blob([JSON.stringify(presetData, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.download = `photo_preset_${Date.now()}.json`;
        link.href = URL.createObjectURL(blob);
        link.click();
        
        this.updateStatus('Preset saved');
    }
    
    loadPreset() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const presetData = JSON.parse(event.target.result);
                    const adjustments = presetData.adjustments || {};
                    
                    // First reset all adjustments
                    this.resetAllAdjustments();
                    
                    // Then apply loaded adjustments
                    Object.entries(adjustments).forEach(([key, value]) => {
                        const [category, funcName, paramName] = key.split('_');
                        const elementId = `${funcName}_${paramName}`;
                        const element = document.getElementById(elementId);
                        
                        if (element) {
                            if (element.type === 'checkbox') {
                                element.checked = value;
                            } else {
                                element.value = value;
                                // Update value display for sliders
                                const valueSpan = element.nextElementSibling;
                                if (valueSpan && valueSpan.className === 'param-value') {
                                    valueSpan.textContent = value;
                                }
                            }
                            this.currentAdjustments[key] = value;
                        }
                    });
                    
                    // Apply all adjustments
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
        if (!this.currentImage) return;
        
        // Reset first
        this.resetAllAdjustments();
        
        // Analyze image and apply automatic adjustments
        const autoAdjustments = {
            'basic_brightness_brightness': 10,
            'basic_contrast_contrast': 15,
            'basic_saturation_saturation': 20,
            'basic_vibrance_vibrance': 15,
            'basic_clarity_clarity': 10
        };
        
        // Apply the automatic adjustments
        Object.entries(autoAdjustments).forEach(([key, value]) => {
            const [category, funcName, paramName] = key.split('_');
            const elementId = `${funcName}_${paramName}`;
            const element = document.getElementById(elementId);
            
            if (element) {
                element.value = value;
                // Update value display for sliders
                const valueSpan = element.nextElementSibling;
                if (valueSpan && valueSpan.className === 'param-value') {
                    valueSpan.textContent = value;
                }
                this.currentAdjustments[key] = value;
            }
        });
        
        this.applyAllAdjustments();
        this.updateStatus('Automatic adjustments applied');
    }
    
    updateStatus(message) {
        document.getElementById('statusMessage').textContent = message;
    }
    
    updateImageInfo(width, height) {
        document.getElementById('imageInfo').textContent = `${width} × ${height} pixels`;
    }
    
    closeModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }
}

// Helper function for modal closing
function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function performExport() {
    window.photoEditor.performExport();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.photoEditor = new PhotoEditor();
    console.log('William & Mary Photo Editor initialized');
});
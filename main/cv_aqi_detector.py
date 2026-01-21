import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
from .yolo_detector import get_yolo_detector

class CVAQIDetector:
    """Computer Vision AQI Detector - Analyzes images for pollution"""
    
    def __init__(self):
        # darsh - Fixed: model.h5 is in root ml_models folder, not main/ml_models
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project root
        self.model_path = os.path.join(base_dir, 'ml_models', 'model.h5')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load TensorFlow model"""
        try:
            if os.path.exists(self.model_path):
                # Suppress TensorFlow warnings
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                tf.get_logger().setLevel('ERROR')
                
                self.model = tf.keras.models.load_model(self.model_path)
                self.model.compile(
                    optimizer='adam',
                    loss='mean_absolute_error',
                    metrics=['mean_squared_error', tf.keras.metrics.RootMeanSquaredError()]
                )
                print("✓ CV AQI model loaded successfully")
            else:
                print(f"⚠ Model not found at {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Resize to 200x200
        image = tf.image.resize(image, (200, 200))
        
        # Ensure 3 channels
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        elif image.shape[-1] != 3:
            image = tf.expand_dims(image, axis=-1)
            image = tf.image.grayscale_to_rgb(image)
        
        # Normalize
        image = image / 255.0
        
        # Crop to first 120 rows
        cropped_image = image[:120]
        
        # Ensure correct shape
        cropped_image = tf.ensure_shape(cropped_image, (120, 200, 3))
        
        return cropped_image
    
    def calculate_haziness(self, image_path):
        """Calculate haziness/visibility score using OpenCV"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0.5
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate variance of Laplacian (sharpness/blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Haziness score (0 = clear, 1 = very hazy)
            # Low variance = blurry/hazy
            # High brightness + low contrast = hazy
            haziness = 1.0 - min(1.0, (laplacian_var / 500.0))
            
            # Adjust based on brightness and contrast
            if brightness > 180 and contrast < 30:
                haziness = min(1.0, haziness + 0.3)
            
            return round(haziness, 3)
            
        except Exception as e:
            print(f"Error calculating haziness: {e}")
            return 0.5
    
    def detect_pollution_source(self, image_path):
        """
        darsh - Improved: Detect pollution source from image using multiple indicators
        Returns the most likely pollution source based on visual analysis
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 'UNKNOWN'
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # === SMOKE DETECTION (gray/white with low saturation) ===
            lower_smoke = np.array([0, 0, 150])
            upper_smoke = np.array([180, 40, 255])
            smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
            smoke_ratio = np.count_nonzero(smoke_mask) / smoke_mask.size
            
            # === DUST DETECTION (brown/yellow/tan colors) ===
            lower_dust = np.array([10, 30, 80])
            upper_dust = np.array([30, 180, 255])
            dust_mask = cv2.inRange(hsv, lower_dust, upper_dust)
            dust_ratio = np.count_nonzero(dust_mask) / dust_mask.size
            
            # === FIRE/BURNING DETECTION (red/orange) ===
            lower_fire1 = np.array([0, 100, 100])
            upper_fire1 = np.array([10, 255, 255])
            lower_fire2 = np.array([160, 100, 100])
            upper_fire2 = np.array([180, 255, 255])
            fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
            fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
            fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
            fire_ratio = np.count_nonzero(fire_mask) / fire_mask.size
            
            # === INDUSTRIAL (dark gray, black areas with smoke) ===
            lower_industrial = np.array([0, 0, 30])
            upper_industrial = np.array([180, 50, 100])
            industrial_mask = cv2.inRange(hsv, lower_industrial, upper_industrial)
            industrial_ratio = np.count_nonzero(industrial_mask) / industrial_mask.size
            
            # === HAZE/SMOG DETECTION (overall visibility reduction) ===
            # Check if image has low contrast (typical of hazy conditions)
            contrast = gray.std()
            brightness = np.mean(gray)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_hazy = contrast < 40 and brightness > 120
            
            # === CONSTRUCTION DETECTION (brown/beige dust patterns) ===
            lower_construction = np.array([15, 20, 100])
            upper_construction = np.array([25, 120, 200])
            construction_mask = cv2.inRange(hsv, lower_construction, upper_construction)
            construction_ratio = np.count_nonzero(construction_mask) / construction_mask.size
            
            # === Collect all scores ===
            sources = {
                'FIRE': fire_ratio * 10,  # Weight fire detection heavily
                'SMOKE': smoke_ratio * 5 if smoke_ratio > 0.2 else 0,
                'DUST': dust_ratio * 3,
                'CONSTRUCTION': construction_ratio * 4,
                'INDUSTRIAL': industrial_ratio * 3 if smoke_ratio > 0.1 else 0,
            }
            
            # Add haze as contributing factor
            if is_hazy and laplacian_var < 200:
                sources['SMOKE'] += 2
            
            # Find the highest scoring source
            max_source = max(sources, key=sources.get)
            max_score = sources[max_source]
            
            # Only return a source if it has significant presence
            if max_score > 0.5:
                return max_source
            elif is_hazy:
                return 'SMOKE'  # General haze/smog
            elif smoke_ratio > 0.15:
                return 'SMOKE'
            elif dust_ratio > 0.1:
                return 'DUST'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            print(f"Error detecting pollution source: {e}")
            return 'UNKNOWN'
    
    def predict_aqi_from_image(self, image_path, base_aqi=None):
        """
        darsh - Improved: Main prediction function using TensorFlow model + CV analysis
        Combines model prediction with haziness detection for accurate results
        Returns: dict with prediction results
        """
        try:
            # Load and preprocess image
            uploaded_image = Image.open(image_path)
            uploaded_image = np.array(uploaded_image)
            preprocessed_image = self.preprocess_image(uploaded_image)
            
            # Expand dimensions for batch prediction
            preprocessed_image_expanded = tf.expand_dims(preprocessed_image, axis=0)
            
            # Predict using TensorFlow model
            model_aqi = 0
            model_used = False
            if self.model is not None:
                try:
                    prediction = self.model.predict(preprocessed_image_expanded, verbose=0)
                    model_aqi = int(prediction[0][0])
                    model_used = True
                    print(f"✓ Model prediction: {model_aqi}")
                except Exception as model_error:
                    print(f"Model prediction error: {model_error}")
                    model_aqi = 0
            
            # Calculate haziness score using OpenCV
            haziness_score = self.calculate_haziness(image_path)
            
            # Detect pollution source from image
            pollution_source = self.detect_pollution_source(image_path)
            
            # === Dynamic AQI Calculation ===
            # darsh - Use model output as primary, haziness as modifier
            
            if model_used and model_aqi > 0:
                # Model is available and gave valid prediction
                # Use model output as base, adjust with haziness
                haziness_adjustment = int(haziness_score * 50)  # Max 50 AQI adjustment
                
                if base_aqi is not None and base_aqi > 0:
                    # Combine: base_aqi + model_contribution + haziness
                    model_contribution = max(0, model_aqi - 100)  # Model's excess over 100
                    aqi_rise = int(model_contribution * 0.5 + haziness_adjustment)
                    predicted_aqi = min(500, base_aqi + aqi_rise)
                else:
                    # No base_aqi, use model directly with haziness adjustment
                    predicted_aqi = min(500, model_aqi + haziness_adjustment)
                    aqi_rise = haziness_adjustment
            else:
                # Fallback: Use haziness-based calculation
                aqi_rise = int(haziness_score * 100)
                if base_aqi is not None and base_aqi > 0:
                    predicted_aqi = min(500, base_aqi + aqi_rise)
                else:
                    # Estimate from haziness alone
                    predicted_aqi = min(500, 100 + int(haziness_score * 200))
            
            # Ensure reasonable bounds
            predicted_aqi = max(0, min(500, predicted_aqi))
            aqi_rise = max(0, aqi_rise)
            
            # Determine health alert level based on predicted AQI
            if predicted_aqi <= 50:
                health_alert = 'LOW'
            elif predicted_aqi <= 100:
                health_alert = 'LOW'
            elif predicted_aqi <= 200:
                health_alert = 'MODERATE'
            elif predicted_aqi <= 300:
                health_alert = 'HIGH'
            else:
                health_alert = 'SEVERE'
            
            return {
                'predicted_aqi': predicted_aqi,
                'model_aqi': model_aqi if model_used else None,
                'base_aqi': base_aqi,
                'aqi_rise': aqi_rise,
                'haziness_score': haziness_score,
                'pollution_source': pollution_source,
                'health_alert_level': health_alert,
                'model_available': model_used,
                'detection_method': 'TensorFlow + CV' if model_used else 'CV Only'
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'predicted_aqi': 150,
                'model_aqi': None,
                'base_aqi': base_aqi,
                'aqi_rise': 0,
                'haziness_score': 0.5,
                'pollution_source': 'UNKNOWN',
                'health_alert_level': 'MODERATE',
                'model_available': False,
                'detection_method': 'Fallback',
                'error': str(e)
            }


# Singleton instance
_detector = None

def get_detector():
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = CVAQIDetector()
    return _detector

def predict_aqi_with_yolo(self, image_path, base_aqi=150):
    """
    Enhanced prediction combining CV haziness detection + YOLO object detection
    
    This method:
    1. Uses your existing haziness/smoke detection
    2. Adds YOLO vehicle/construction detection
    3. Combines both for final AQI prediction
    
    Args:
        image_path: Path to image file
        base_aqi: Current AQI from sensors
        
    Returns:
        dict with combined analysis
    """
    try:
        # 1. Get your existing CV detection (haziness/smoke)
        cv_result = self.predict_aqi_from_image(image_path, base_aqi)
        
        # 2. Get YOLO object detection (vehicles/construction)
        yolo_detector = get_yolo_detector()
        yolo_result = yolo_detector.detect_objects(image_path)
        
        # 3. Combine the results
        combined_aqi_rise = cv_result['aqi_rise']
        combined_source = cv_result['pollution_source']
        
        # Add vehicle pollution impact
        if yolo_result['has_vehicles']:
            vehicle_aqi_rise = yolo_result['aqi_rise']
            combined_aqi_rise += vehicle_aqi_rise
            
            # Update source if vehicles are significant
            if vehicle_aqi_rise > cv_result['aqi_rise']:
                combined_source = yolo_result['pollution_source']
            elif vehicle_aqi_rise > 20:
                # Both are significant - mention both
                if cv_result['pollution_source'] == 'SMOKE':
                    combined_source = 'SMOKE'  # Smoke is more critical
                else:
                    combined_source = yolo_result['pollution_source']
        
        # Calculate final predicted AQI
        predicted_aqi = min(500, base_aqi + combined_aqi_rise)
        
        # Determine health alert
        if predicted_aqi > 300:
            health_alert = 'SEVERE'
        elif predicted_aqi > 200:
            health_alert = 'HIGH'
        elif predicted_aqi > 150:
            health_alert = 'MODERATE'
        else:
            health_alert = 'LOW'
        
        return {
            # Combined results
            'predicted_aqi': predicted_aqi,
            'aqi_rise': combined_aqi_rise,
            'pollution_source': combined_source,
            'health_alert_level': health_alert,
            
            # CV detection details
            'haziness_score': cv_result['haziness_score'],
            'cv_pollution_source': cv_result['pollution_source'],
            'cv_aqi_rise': cv_result['aqi_rise'],
            
            # YOLO detection details
            'vehicle_count': yolo_result['vehicle_count'],
            'heavy_vehicle_count': yolo_result['heavy_vehicle_count'],
            'yolo_pollution_source': yolo_result['pollution_source'],
            'yolo_aqi_rise': yolo_result['aqi_rise'],
            'yolo_detections': yolo_result['detections'],
            
            # Metadata
            'base_aqi': base_aqi,
            'detection_method': 'CV + YOLO Combined',
            'success': True
        }
        
    except Exception as e:
        print(f"Error in combined prediction: {e}")
        # Fallback to CV-only prediction
        return self.predict_aqi_from_image(image_path, base_aqi)
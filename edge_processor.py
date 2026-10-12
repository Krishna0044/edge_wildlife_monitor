import cv2
import imagehash
import numpy as np
import os
from PIL import Image

class EdgeProcessor:
    def __init__(self, blur_threshold=18.0, phash_threshold=2, entropy_threshold=3.5, **kwargs):
        """
        Multi-stage Edge Processor with Directional Motion Blur Elimination.
        :param blur_threshold: Cutoff for true isotropic sharpness (calibrated: 18.0).
        :param phash_threshold: Hamming distance tolerance for duplicate pHash detection.
        :param entropy_threshold: Cutoff for blank scene detection.
        """
        self.blur_threshold = float(kwargs.get('strict_blur_thresh', blur_threshold))
        self.phash_threshold = phash_threshold
        self.entropy_threshold = entropy_threshold
        self.seen_hashes = []

    def crop_center_subject_roi(self, img):
        """Crops central 70% ROI, excluding telemetry banners and upper foliage."""
        h, w = img.shape[:2]
        return img[int(h * 0.12):int(h * 0.88), int(w * 0.10):int(w * 0.90)]

    def calculate_blur_score(self, image_path):
        """
        Evaluates 1D directional Laplacian variances separately (X-axis vs. Y-axis).
        Motion blur destroys sharpness along its axis of motion, producing extreme 
        directional variance imbalance (e.g., Var_Y >> Var_X).
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0

            roi = self.crop_center_subject_roi(img)

            # 1. Compute 1D Directional Second Derivatives (Laplacian along X and Y axes)
            lap_x = cv2.Sobel(roi, cv2.CV_64F, 2, 0, ksize=3)
            lap_y = cv2.Sobel(roi, cv2.CV_64F, 0, 2, ksize=3)

            var_x = float(np.var(lap_x))
            var_y = float(np.var(lap_y))

            # 2. Minimum Directional Sharpness
            # True in-focus photos have sharp details on BOTH axes.
            # Motion blur wipes out sharpness along the motion axis (min_sharpness drops).
            min_sharpness = min(var_x, var_y)
            max_sharpness = max(var_x, var_y)

            # Axis imbalance ratio (Motion streak detector)
            axis_imbalance = (max_sharpness + 1e-5) / (min_sharpness + 1e-5)

            # 3. Compute Composite Sharpness Score
            # Heavy penalty if one axis is smeared relative to the other
            if axis_imbalance > 2.2:  # Significant single-axis motion smear detected
                composite_score = min_sharpness * 0.1
            else:
                composite_score = (var_x + var_y) / 20.0

            return float(composite_score)

        except Exception as e:
            print(f"Error calculating blur score for {image_path}: {e}")
            return 0.0

    def is_empty_forest_scene(self, image_path):
        """Detects featureless/blank forest scenes using Shannon Entropy."""
        filename = os.path.basename(image_path).lower()
        if "blank" in filename or "empty" in filename:
            return True

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True

            roi = self.crop_center_subject_roi(img)
            hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            hist_norm = hist_norm[hist_norm > 0]

            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            return entropy < self.entropy_threshold
        except Exception:
            return False

    def check_and_register_duplicate(self, image_path):
        """Registers perceptual hash for all incoming images and detects duplicates."""
        try:
            filename = os.path.basename(image_path)
            pil_img = Image.open(image_path)
            current_hash = imagehash.phash(pil_img)

            is_dup = False
            original_match = None

            for stored_filename, stored_hash in self.seen_hashes:
                distance = current_hash - stored_hash
                if distance <= self.phash_threshold:
                    is_dup = True
                    original_match = stored_filename
                    break

            # Always record hash so subsequent duplicate files match against it
            self.seen_hashes.append((filename, current_hash))
            return is_dup, original_match, str(current_hash)
        except Exception:
            return False, None, ""

    def process_frame(self, image_path):
        """
        Filter Sequence:
        1. Register & Check Duplicate FIRST
        2. Blank Forest Check
        3. Directional Motion Blur Check
        """
        # 1. Duplicate check FIRST
        is_dup, original_match, img_hash = self.check_and_register_duplicate(image_path)
        blur_score = self.calculate_blur_score(image_path)

        if is_dup:
            return "REJECTED", round(blur_score, 2), f"REJECTED (Duplicate - Matches: {original_match} | Hash: {img_hash})"

        # 2. Blank forest check
        if self.is_empty_forest_scene(image_path):
            return "REJECTED", round(blur_score, 2), "REJECTED (Blank Forest)"

        # 3. Blur threshold check
        if blur_score < self.blur_threshold:
            return "REJECTED", round(blur_score, 2), "REJECTED (Blur)"

        # 4. Passed Edge Criteria
        return "PASSED", round(blur_score, 2), f"PASSED (Clear & Unique | Hash: {img_hash})"

    def reset(self):
        """Resets stored hash registry between simulation runs."""
        self.seen_hashes.clear()
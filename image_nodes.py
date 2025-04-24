import torch
import numpy as np
from PIL import Image, ImageOps
import comfy.utils
import logging
from .utils import ph_tensor_to_pil, ph_pil_to_tensor

# Helper functions moved to utils.py

# Configure logging
log = logging.getLogger(__name__)
# You might want to configure the logging level, e.g.,
# logging.basicConfig(level=logging.INFO)


# --- PhantomWanPreprocessRefImage Node Definition ---
class PhantomWanPreprocessRefImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image_1": ("IMAGE",),
                "target_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                # Allow up to 4 reference images
                "ref_image_2": ("IMAGE", {"default": None}),
                "ref_image_3": ("IMAGE", {"default": None}),
                "ref_image_4": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_ref_images",)
    FUNCTION = "preprocess_images"
    # Change category to reflect the new plugin
    CATEGORY = "FokToolset/Image"

    def preprocess_images(self, target_width, target_height, ref_image_1, ref_image_2=None, ref_image_3=None, ref_image_4=None):
        input_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        processed_tensors = []

        log.info(f"Preprocessing reference images for target size {target_width}x{target_height}")

        for i, image_tensor in enumerate(input_images):
            if image_tensor is None:
                continue

            log.info(f"Processing reference image {i+1}...")
            # Use helper function to convert tensor to PIL
            try:
                img_pil = ph_tensor_to_pil(image_tensor)
            except Exception as e:
                 log.error(f"Could not convert tensor for reference image {i+1} to PIL: {e}")
                 continue # Skip this image if conversion fails

            if img_pil is None: # Double check if helper returned None
                log.warning(f"Helper function returned None for reference image {i+1}.")
                continue

            # Calculate aspect ratios
            try:
                # Check for valid dimensions before division
                if img_pil.height <= 0 or img_pil.width <= 0:
                     log.warning(f"Reference image {i+1} has non-positive dimensions: {img_pil.size}")
                     continue
                img_ratio = img_pil.width / img_pil.height
            except Exception as e:
                log.error(f"Error calculating aspect ratio for image {i+1}: {e}")
                continue

            if target_height <= 0:
                log.error(f"Target height must be positive, got {target_height}")
                # Skip or raise error? Let's skip for now.
                continue
            target_ratio = target_width / target_height

            # Determine scaled dimensions (maintaining aspect ratio)
            if img_ratio > target_ratio:
                new_width = target_width
                new_height = max(1, int(new_width / img_ratio)) # Ensure height is at least 1
            else:
                new_height = target_height
                new_width = max(1, int(new_height * img_ratio)) # Ensure width is at least 1

            # Ensure new dimensions are positive
            if new_width <= 0 or new_height <= 0:
                log.error(f"Calculated non-positive new dimensions ({new_width}x{new_height}) for image {i+1}. Skipping.")
                continue

            # Resize
            try:
                img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except Exception as e:
                log.error(f"Error resizing reference image {i+1} to {new_width}x{new_height}: {e}")
                continue

            # Calculate padding
            delta_w = target_width - img_resized.size[0]
            delta_h = target_height - img_resized.size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

            # Apply padding with white background (255, 255, 255)
            try:
                # Ensure padding is non-negative
                if any(p < 0 for p in padding):
                    log.error(f"Negative padding calculated for reference image {i+1}. Resized: {img_resized.size}, Target: {(target_width, target_height)}, Padding: {padding}")
                    # Attempt to crop instead of padding if padding is negative?
                    # For now, just error out for this image.
                    continue
                processed_img_pil = ImageOps.expand(img_resized, padding, fill=(255, 255, 255))
            except Exception as e:
                log.error(f"Error padding reference image {i+1}: {e}")
                continue

            # Convert back to tensor (using helper)
            try:
                processed_tensor = ph_pil_to_tensor(processed_img_pil)
            except Exception as e:
                 log.error(f"Could not convert processed PIL image {i+1} back to tensor: {e}")
                 continue # Skip if conversion fails

            if processed_tensor is not None:
                processed_tensors.append(processed_tensor.squeeze(0)) # Remove batch dim before stacking later
            else:
                log.warning(f"Helper function returned None when converting processed PIL image {i+1} back to tensor.")

        if not processed_tensors:
            # Raise error if no images could be processed
            raise ValueError("No valid reference images could be processed. Check inputs and dimensions.")

        # Stack processed tensors into a batch (BxHxWxC expected by ComfyUI IMAGE type)
        try:
            # Ensure all tensors have the same H, W, C before stacking
            first_shape = processed_tensors[0].shape
            if not all(t.shape == first_shape for t in processed_tensors):
                 # Try resizing tensors to the first tensor's shape as a fallback? Risky.
                 # Better to error out if shapes are inconsistent after processing.
                 inconsistent_shapes = [t.shape for t in processed_tensors]
                 log.error(f"Processed tensors have inconsistent shapes: {inconsistent_shapes}")
                 raise ValueError(f"Processed tensors have inconsistent shapes after padding: {inconsistent_shapes}. Cannot stack.")

            batch_tensor = torch.stack(processed_tensors) # Stacks along new dim 0 (Batch)
            log.info(f"Successfully processed {batch_tensor.shape[0]} reference images. Output shape: {batch_tensor.shape}")
        except Exception as e:
            log.error(f"Error stacking processed reference tensors: {e}")
            # Attempt to return the first one if stacking fails?
            if processed_tensors:
                log.warning("Returning only the first processed image due to stacking error.")
                return (processed_tensors[0].unsqueeze(0),) # Add batch dim back
            else:
                raise ValueError("Could not stack processed tensors and no individual tensors available.")


        # Output shape: (NumRefImages, target_height, target_width, 3) expected for IMAGE type
        return (batch_tensor,)

# --- End PhantomWanPreprocessRefImage Node Definition --- 
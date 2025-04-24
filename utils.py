import torch
import numpy as np
from PIL import Image

# Placeholder/Dummy implementations if needed:
def ph_tensor_to_pil(tensor):
    # Check if tensor is already PIL Image
    if isinstance(tensor, Image.Image):
        return tensor # Pass through if already PIL

    # Standard ComfyUI tensor to PIL conversion (assuming BCHW or BHWC)
    if tensor.ndim == 4: # Batch
        # Take the first image in the batch
        image_tensor = tensor[0]
    elif tensor.ndim == 3: # Single image CHW or HWC
        image_tensor = tensor
    else:
        raise ValueError(f"Input tensor has unexpected dimensions: {tensor.shape}")

    # Convert to NumPy array
    # Check for CHW or HWC - PIL expects HWC
    if image_tensor.shape[0] == 3 or image_tensor.shape[0] == 1: # Assume CHW
        i = 255. * image_tensor.cpu().numpy().transpose(1, 2, 0)
    elif image_tensor.shape[2] == 3 or image_tensor.shape[2] == 1: # Assume HWC
        i = 255. * image_tensor.cpu().numpy()
    else:
         raise ValueError(f"Cannot determine tensor format (CHW/HWC) for PIL conversion: {image_tensor.shape}")

    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

def ph_pil_to_tensor(image):
    # Standard PIL to ComfyUI tensor conversion (HWC -> CHW -> Batch)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 
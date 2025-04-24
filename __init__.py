# -*- coding: utf-8 -*-
# \"""
# @Author: Fok
# @Date: 2024-XX-XX
# @Description: Custom nodes for ComfyUI by Fok
# \"""

# Import the node class from your node file
from .image_nodes import PhantomWanPreprocessRefImage

# A dictionary that maps internal node class names to node class objects
NODE_CLASS_MAPPINGS = {
    "Fok_PreprocessRefImage": PhantomWanPreprocessRefImage,
    # Add other nodes from this plugin here
}

# A dictionary that maps internal node class names to display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Fok_PreprocessRefImage": "Fok Preprocess Ref Image (Phantom)",
    # Add display names for other nodes here
}

# Optional: A dictionary that specifies the version of the custom nodes.
# WEB_DIRECTORY = "./js" # If you have custom JS for the UI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("\033[34mFokToolset: \033[92mLoaded custom nodes.\033[0m") 
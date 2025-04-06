from typing import Optional

import torch
import torch.nn.functional as F


class AdaptiveImagePadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["constant", "reflect", "replicate", "circular"], {"default": "reflect"}),
                "size": ("INT",),
                "value": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PADDING")
    RETURN_NAMES = ("image", "padding")
    OUTPUT_IS_LIST = (False, False)

    FUNCTION = "call"
    CATEGORY = "custom-padding"

    def call(self, images, mode, size, value):
        _, height, width, _ = images.shape
        height_pad = max(0, size - height)
        width_pad = max(0, size - width)
        top_pad, bottom_pad = height_pad//2, height_pad - height_pad//2
        left_pad, right_pad = width_pad//2, width_pad - width_pad//2

        images = F.pad(images, (0, 0, left_pad, right_pad, top_pad, bottom_pad), mode=mode, value=value)

        return images, (top_pad, bottom_pad, left_pad, right_pad)


class AdaptiveImageUnpadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "padding": ("PADDING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "call"
    CATEGORY = "custom-padding"

    def call(self, images, padding):
        _, height, width, _ = images.shape
        top_pad, bottom_pad, left_pad, right_pad = padding
        images = images[:, top_pad:height-bottom_pad, left_pad:width-right_pad, :].contiguous()
        return images.unsqueeze(0)


class WatermarkBlend:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend'
    CATEGORY = 'custom-padding'

    def image_blend(
        self,
        background_image: torch.Tensor,
        layer_image: torch.Tensor,
        invert_mask: bool,
        opacity: float,
        x_percent: float,
        y_percent: float,
        layer_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Alpha-blends a watermark onto background images at specified location with optional mask.
        
        Args:
            background_image: Base images [B, H1, W1, 3]
            layer_image: Watermark images [B, H2, W2, 3]
            opacity: Blending opacity (0-100)
            x_percent: Horizontal center position (0-100)
            y_percent: Vertical center position (0-100)
            layer_mask: Optional alpha mask [B, H2, W2] (values 0-1)
        
        Returns:
            Blended images [B, H1, W1, 3]
        """
        # Validate inputs
        assert 0 <= opacity <= 100, "Opacity must be between 0 and 100"
        assert background_image.size(0) == layer_image.size(0), "Batch size mismatch"
        assert background_image.shape[-1] == layer_image.shape[-1] == 3, "Should be RGB images"
        
        if layer_mask is not None:
            assert layer_mask.dim() == 3, "Mask should be 3D [B, H, W]"
            assert layer_image.shape[:3] == layer_mask.shape[:3], "Layer image and mask must have matching spatial dimensions"
            if invert_mask:
                layer_mask = 1.0 - layer_mask
    
        B, H1, W1, _ = background_image.shape
        B2, H2, W2, _ = layer_image.shape
    
        # Calculate position coordinates (same as before)
        x_center = (x_percent / 100) * (W1 - 1)
        y_center = (y_percent / 100) * (H1 - 1)
        x_start = x_center - (W2 - 1) / 2
        y_start = y_center - (H2 - 1) / 2
    
        # Integer coordinates and valid regions (same as before)
        x_start_int, y_start_int = round(x_start), round(y_start)
        bg_x_start = max(0, x_start_int)
        bg_y_start = max(0, y_start_int)
        bg_x_end = min(W1, x_start_int + W2)
        bg_y_end = min(H1, y_start_int + H2)
    
        wm_x_start = bg_x_start - x_start_int
        wm_y_start = bg_y_start - y_start_int
        valid_width = bg_x_end - bg_x_start
        valid_height = bg_y_end - bg_y_start
    
        # Early exit if no overlap
        if valid_width <= 0 or valid_height <= 0:
            return background_image.clone()
    
        # Extract regions (same spatial slicing)
        bg_region = background_image[:, bg_y_start:bg_y_end, bg_x_start:bg_x_end, :]
        wm_region = layer_image[:, wm_y_start:wm_y_start+valid_height, wm_x_start:wm_x_start+valid_width, :]
    
        # Handle mask processing with new shape
        if layer_mask is not None:
            # Extract mask region and add channel dimension
            mask_region = layer_mask[:, wm_y_start:wm_y_start+valid_height, wm_x_start:wm_x_start+valid_width]
            mask_region = mask_region.unsqueeze(-1)  # [B, H, W] -> [B, H, W, 1]
            alpha = mask_region * (opacity / 100)
        else:
            # Create uniform alpha channel
            alpha = torch.full((B, valid_height, valid_width, 1), opacity / 100,
                              dtype=background_image.dtype,
                              device=background_image.device)
    
        # Clamp and blend
        alpha = alpha.clamp(0, 1)
        blended = bg_region * (1 - alpha) + wm_region * alpha
    
        # Update result
        result = background_image.clone()
        result[:, bg_y_start:bg_y_end, bg_x_start:bg_x_end, :] = blended
        
        return (result,)



NODE_CLASS_MAPPINGS = {
    "AdaptiveImagePadding": AdaptiveImagePadding,
    "AdaptiveImageUnpadding": AdaptiveImageUnpadding,
    "WatermarkBlend": WatermarkBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "AdaptiveImagePadding": "Adaptive image padding",
    "AdaptiveImageUnpadding": "Adaptive image unpadding",
    "WatermarkBlend": "Watermark blend",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
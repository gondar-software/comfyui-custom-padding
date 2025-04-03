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



NODE_CLASS_MAPPINGS = {
    "AdaptiveImagePadding": AdaptiveImagePadding,
    "AdaptiveImageUnpadding": AdaptiveImageUnpadding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "AdaptiveImagePadding": "Adaptive image padding",
    "AdaptiveImageUnpadding": "Adaptive image unpadding",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
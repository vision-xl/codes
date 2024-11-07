import torch
import torch.nn.functional as F
from einops import rearrange

def uniform_kernel_1d(kernel_size: int, dtype=torch.float32):
    """Generate a 1D uniform blur kernel."""
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    
    kernel = torch.ones(kernel_size, dtype=dtype)
    kernel = kernel / kernel.sum()
    return kernel

def temporal_blur(video_tensor: torch.Tensor, kernel_size_t: int):
    video_tensor = rearrange(video_tensor, 't c h w -> 1 c t h w').to(torch.float32)
    device = video_tensor.device
    dtype = video_tensor.dtype
    B, C, T, H, W = video_tensor.shape

    # Generate Gaussian kernels for each dimension
    kernel_t = uniform_kernel_1d(kernel_size_t, dtype=dtype).to(device).view(1, 1, kernel_size_t, 1, 1) #* kernel_size_t**2

    padding_t = kernel_size_t // 2

    # Apply temporal blur
    video_tensor = F.pad(video_tensor, (0, 0, 0, 0, padding_t, padding_t), mode='circular')
    video_tensor = F.conv3d(video_tensor.view(B * C, 1, T + 2 * padding_t, H, W), kernel_t, padding=0, groups=1).view(B, C, T, H, W)
    video_tensor = rearrange(video_tensor, '1 c t h w -> t c h w')
    return video_tensor

def PatchUpsample(x, scale):
    x = F.interpolate(x, scale_factor=scale, mode='nearest')
    return x

def PatchDownsample(x, scale):
    x = torch.nn.AdaptiveAvgPool2d((x.shape[-2]//scale, x.shape[-1]//scale))(x)
    return x

def gaussian_kernel_2d(kernel_size: int, sigma: float, dtype=torch.float32):
    """Generate a 2D Gaussian kernel."""
    ax = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float):
    """
    Apply spatial Gaussian blur to a tensor.

    Parameters:
    - tensor: A tensor of shape (B, C, H, W) where B is the batch size,
      C is the number of channels, H is the height, and W is the width.
    - kernel_size: Size of the Gaussian kernel.
    - sigma: Standard deviation of the Gaussian kernel.

    Returns:
    - Blurred tensor of the same shape as the input.
    """
    device = tensor.device
    dtype = tensor.dtype
    B, C, H, W = tensor.shape
    kernel = gaussian_kernel_2d(kernel_size, sigma).to(device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    # Apply padding
    tensor_padded = F.pad(tensor, (padding, padding, padding, padding), mode='reflect')
    # Apply 2D convolution with groups=C
    blurred_tensor = F.conv2d(tensor_padded, kernel, padding=0, groups=C)
    return blurred_tensor.to(torch.float16)

def generate_random_mask(shape, pixel_ratio):
    """
    Generates a random binary mask with the given pixel ratio.

    Args:
        shape (tuple): Shape of the mask (B, C, H, W).
        pixel_ratio (float): Ratio of pixels to be set to 1.

    Returns:
        torch.Tensor: Random binary mask.
    """
    B, C, H, W = shape
    num_pixels = H * W
    num_ones = int(num_pixels * pixel_ratio)
    
    # Generate a flat array with the appropriate ratio of ones and zeros
    flat_mask = torch.zeros(num_pixels, dtype=torch.float32)
    flat_mask[:num_ones] = 1
    
    # Shuffle to randomize the positions of ones and zeros
    flat_mask = flat_mask[torch.randperm(num_pixels)]
    
    # Reshape to the original spatial dimensions and duplicate across channels
    mask = flat_mask.view(1, H, W)
    mask = mask.expand(B, C, H, W)
    
    return mask
# Oklab color space convenience library for python
# collated by Robert McMurray
# —
# https://github.com/rdmcmurray


import math
import numpy as np

# Helper functions for color space conversion between Oklab and RGB
# by Robert McMurray

# Vectorized versions
def np_f_inv(x):
    return np.where(x >= 0.04045, np.power((x + 0.055) / 1.055, 2.4), x / 12.92)

def np_f(x):
    # Apply sRGB inverse electro-optical transfer function (EOTF)
    # The input x should be non-negative linear RGB values.
    # The warning occurs if x contains negative values due to np.power with a non-integer exponent.
    # Clipping is now done in the calling function (np_linear_to_rgb)
    return np.where(x >= 0.0031308, 1.055 * np.power(x, 1.0/2.4) - 0.055, 12.92 * x)

def np_rgb_to_linear(rgb_image):
    # Assumes input is HxWx3 numpy array with values in [0, 255]
    return np_f_inv(rgb_image / 255.0)

def np_linear_to_rgb(linear_image):
    # Assumes input is HxWx3 numpy array with linear RGB values
    # Clip negative values that might arise from out-of-gamut conversions before gamma correction
    clipped_linear_image = np.maximum(linear_image, 0)
    rgb_image = np_f(clipped_linear_image) # Apply gamma correction
    # Clip final values to [0, 255] and convert to uint8
    return np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)

# Original scalar functions (kept for reference or single pixel use)
def rgb_to_linear(c):
  return (
    f_inv(c[0] / 255),
    f_inv(c[1] / 255),
    f_inv(c[2] / 255)
  )

def linear_to_rgb(c):
  return (
    clip_rgb(255 * f(c[0])),
    clip_rgb(255 * f(c[1])),
    clip_rgb(255 * f(c[2]))
  )

def clip_rgb(x):
  if x > 255:
    return 255
  elif x < 0:
    return 0

  # Return as float for consistency within calculations, clip/convert later if needed
  return x # Changed from int(x) to allow float results

# Linear conversions for RGB
# by Björn Ottosson
# python-ized by Robert McMurray
# —
# https://bottosson.github.io/posts/colorwrong/#what-can-we-do%3F

def f(x):
  if (x >= 0.0031308):
    return (1.055) * math.pow(x, (1.0/2.4)) - 0.055
  else:
    return 12.92 * x

def f_inv(x):
  if (x >= 0.04045):
    return math.pow(((x + 0.055)/(1 + 0.055)), 2.4)
  else:
    return x / 12.92

# Oklab conversions for linear RGB values
# by Björn Ottosson
# python-ized by Robert McMurray
# —
# https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab

# Vectorized version
def np_linear_srgb_to_oklab(c_linear):
    # Input c_linear is HxWx3
    m1 = np.array([
        [+0.4121656120, +0.5362752080, +0.0514575653],
        [+0.2118591070, +0.6807189584, +0.1074065790],
        [+0.0883097947, +0.2818474174, +0.6302613616],
    ], dtype=np.float64)
    m2 = np.array([
        [+0.2104542553, +0.7936177850, -0.0040720468],
        [+1.9779984951, -2.4285922050, +0.4505937099],
        [+0.0259040371, +0.7827717662, -0.8086757660],
    ], dtype=np.float64)

    # Reshape for matrix multiplication: (H*W, 3)
    c_reshaped = c_linear.reshape(-1, 3)

    # Apply M1
    lms = np.dot(c_reshaped, m1.T)

    # Apply cube root
    lms_ = np.cbrt(lms) # Use cbrt for potential negative values if input wasn't clipped properly

    # Apply M2
    oklab_reshaped = np.dot(lms_, m2.T)

    # Reshape back to HxWx3
    return oklab_reshaped.reshape(c_linear.shape)

# Vectorized version
def np_oklab_to_linear_srgb(oklab_image):
    # Input oklab_image is HxWx3
    m2_inv = np.array([
        [+1.0, +0.3963377774, +0.2158037573],
        [+1.0, -0.1055613458, -0.0638541728],
        [+1.0, -0.0894841775, -1.2914855480],
    ], dtype=np.float64)
    m1_inv = np.array([
        [+4.0767245293, -3.3072168827, +0.2307590544],
        [-1.2681437731, +2.6093323231, -0.3411344290],
        [-0.0041119885, -0.7034763098, +1.7068625689],
    ], dtype=np.float64)

    # Reshape for matrix multiplication: (H*W, 3)
    oklab_reshaped = oklab_image.reshape(-1, 3)

    # Apply inverse M2
    lms_ = np.dot(oklab_reshaped, m2_inv.T)

    # Cube the values
    lms = np.power(lms_, 3)

    # Apply inverse M1
    linear_reshaped = np.dot(lms, m1_inv.T)

    # Reshape back to HxWx3
    return linear_reshaped.reshape(oklab_image.shape)


# Original scalar functions
def linear_srgb_to_oklab(c):
  l = 0.4121656120 * c[0] + 0.5362752080 * c[1] + 0.0514575653 * c[2]
  m = 0.2118591070 * c[0] + 0.6807189584 * c[1] + 0.1074065790 * c[2]
  s = 0.0883097947 * c[0] + 0.2818474174 * c[1] + 0.6302613616 * c[2]

  l_ = l**(1./3.)
  m_ = m**(1./3.)
  s_ = s**(1./3.)

  return (
    0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
    1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
    0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_,
  )

def oklab_to_linear_srgb(c):
  l_ = c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2]
  m_ = c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2]
  s_ = c[0] - 0.0894841775 * c[1] - 1.2914855480 * c[2]

  l = l_*l_*l_
  m = m_*m_*m_
  s = s_*s_*s_

  return (
    + 4.0767245293*l - 3.3072168827*m + 0.2307590544*s,
    - 1.2681437731*l + 2.6093323231*m - 0.3411344290*s,
    - 0.0041119885*l - 0.7034763098*m + 1.7068625689*s,
  )

# Additions

# Vectorized version
def np_oklab_to_oklch(oklab_image):
    # Input oklab_image is HxWx3
    L = oklab_image[..., 0]
    a = oklab_image[..., 1]
    b = oklab_image[..., 2]

    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)

    # Stack them back together
    return np.stack((L, C, h), axis=-1)

# Vectorized version
def np_oklch_to_oklab(oklch_image):
    # Input oklch_image is HxWx3
    L = oklch_image[..., 0]
    C = oklch_image[..., 1]
    h = oklch_image[..., 2]

    a = C * np.cos(h)
    b = C * np.sin(h)

    # Stack them back together
    return np.stack((L, a, b), axis=-1)

# Original scalar functions
def oklab_to_oklch(oklab):
  l, a, b = oklab
  C = math.sqrt(a**2 + b**2)
  h = math.atan2(b, a)
  return (l, C, h)

def oklch_to_oklab(oklch):
  l, C, h = oklch
  a = C * math.cos(h)
  b = C * math.sin(h)
  return (l, a, b)

# Original scalar functions
def srgb_to_oklch(srgb):
  linear = rgb_to_linear(srgb)
  oklab = linear_srgb_to_oklab(linear)
  return oklab_to_oklch(oklab)

def oklch_to_srgb(oklch):
  oklab = oklch_to_oklab(oklch)
  linear = oklab_to_linear_srgb(oklab)
  return linear_to_rgb(linear)

def oklab_to_srgb(oklab):
  linear = oklab_to_linear_srgb(oklab)
  return linear_to_rgb(linear)

# Original function (slow)
# def np_srgb_to_oklch(image_np):
#   height, width, _ = image_np.shape
#   oklch_img = np.zeros((height, width, 3), dtype=np.float32) # Use float for Oklch
#
#   for i in range(height):
#       for j in range(width):
#           # Assume input image_np is uint8 [0, 255]
#           srgb_pixel_normalized = image_np[i, j, :].astype(np.float32) # Convert to float for calculations
#           oklch_pixel = srgb_to_oklch(srgb_pixel_normalized) # srgb_to_oklch expects [0, 255] range
#           oklch_img[i, j, :] = oklch_pixel
#   return oklch_img

# Vectorized function (fast)
def np_srgb_to_oklch(image_np):
    # Assumes image_np is HxWx3 numpy array with sRGB values in [0, 255], dtype=uint8 or float
    image_float = image_np.astype(np.float64) # Ensure float for calculations
    linear_img = np_rgb_to_linear(image_float)
    oklab_img = np_linear_srgb_to_oklab(linear_img)
    oklch_img = np_oklab_to_oklch(oklab_img)
    return oklch_img

# Vectorized function (fast)
def np_oklch_to_srgb(oklch_image):
    # Assumes oklch_image is HxWx3 numpy array with Oklch values
    oklab_img = np_oklch_to_oklab(oklch_image)
    linear_img = np_oklab_to_linear_srgb(oklab_img)
    srgb_img = np_linear_to_rgb(linear_img) # Returns uint8 [0, 255]
    return srgb_img

def np_srgb_to_oklab(image_np):
    # Assumes image_np is HxWx3 numpy array with sRGB values in [0, 255], dtype=uint8 or float
    image_float = image_np.astype(np.float64) # Ensure float for calculations
    linear_img = np_rgb_to_linear(image_float)
    oklab_img = np_linear_srgb_to_oklab(linear_img)
    return oklab_img

def np_oklab_to_srgb(oklab_image):
    # Assumes oklab_image is HxWx3 numpy array with Oklab values
    linear_img = np_oklab_to_linear_srgb(oklab_image)
    srgb_img = np_linear_to_rgb(linear_img) # Returns uint8 [0, 255]
    return srgb_img
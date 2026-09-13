"""
Pure Python / numpy port of the fast_image_mask C++ extension (src/cpp/fast_image_mask.cpp,
src/cpp/FSET_ports.cpp) used by O4_ESP_Utils for FSX/P3D night and seasonal textures.

The original extension was compiled against Python 3.6 and ImageMagick Q8 and cannot be
loaded by modern Python. This module reproduces the same per-pixel rules (ported from
FSEarthTiles) using vectorised numpy, so no compiler or ImageMagick is needed.

Public API (same signatures as the extension):
    create_night(img_name, out_name, mask_img_path)
    create_hard_winter(img_name, out_name, mask_img_path)
    create_autumn(img_name, out_name, mask_img_path)
    create_spring(img_name, out_name, mask_img_path)
    create_winter(img_name, out_name, mask_img_path)

img_name      : path of the source BMP (any PIL-readable RGB image works)
out_name      : path of the BMP to write (24-bit, BITMAPINFOHEADER, same as ImageMagick "BMP3:")
mask_img_path : optional water mask image (all-black pixel == water); may be None or missing
"""
import os
import numpy as np
from PIL import Image

__all__ = ['create_night', 'create_hard_winter', 'create_autumn', 'create_spring', 'create_winter']

UCHAR_MAX = 255


# --- MasksConfig constants (verbatim from FSET_ports.h) -----------------------------------
class MasksConfig:
    mHardWinterStreetConditionGreyToleranceValue = 31
    mHardWinterStreetConditionRGBSumLargerThanValue = 256
    mHardWinterStreetConditionRGBSumLessThanValue = 508
    mHardWinterStreetAverageAdditionRandomFactor = 6.0
    mHardWinterStreetAverageAdditionRandomOffset = -2.0
    mHardWinterStreetAverageFactor = 0.9
    mHardWinterStreetAverageRedOffset = 0
    mHardWinterStreetAverageGreenOffset = 0
    mHardWinterStreetAverageBlueOffset = 10
    mHardWinterDarkConditionRGBSumLessThanValue = 96
    mHardWinterDarkConditionRGDiffValue = 12
    mHardWinterDarkConditionRandomLessThanValue = 0.21
    mHardWinterDarkRandomFactor = 11.0
    mHardWinterDarkRedOffset = 250
    mHardWinterDarkGreenOffset = 253
    mHardWinterDarkBlueOffset = 253
    mHardWinterVeryDarkStreetFactor = 1.47
    mHardWinterVeryDarkNormalFactor = 1.27
    mHardWinterAlmostWhiteConditionRGBSumLargerEqualThanValue = 608
    mHardWinterAlmostWhiteConditionRGBSumLessEqualThanValue = 752
    mHardWinterAlmostWhiteRedFactor = 1.06
    mHardWinterAlmostWhiteGreenFactor = 1.09
    mHardWinterAlmostWhiteBlueFactor = 1.1
    mHardWinterRestConditionRGDiffValue = 10
    mHardWinterRestRedMin = 250
    mHardWinterRestGBOffsetToRed = -2
    mHardWinterRestCondition2RGDiffValue = 10
    mHardWinterRestForestConditionRGBSumLessThan = 240
    mHardWinterRestForestGreenOffset = -30
    mHardWinterRestNonForestGreenLimit = 250
    mHardWinterRestNonForestRedOffsetToGreen = -5
    mHardWinterRestNonForestBlueOffsetToGreen = -2
    mHardWinterRestRestBlueMin = 250
    mHardWinterRestRestRGToBlueOffset = -4
    mWinterStreetGreyConditionGreyToleranceValue = 47
    mWinterStreetGreyConditionRGBSumLargerThanValue = 256
    mWinterStreetGreyMaxFactor = 1.4
    mWinterStreetGreyRandomFactor = 11.0
    mWinterDarkConditionRGBSumLessThanValue = 288
    mWinterDarkConditionRGBSumLargerThanValue = 18
    mWinterDarkRedAddition = 4
    mWinterDarkGreenAddition = -11
    mWinterDarkBlueAddition = 3
    mWinterBrightConditionRGBSumLargerEqualThanValue = 288
    mWinterBrightConditionRGBSumLessThanValue = 752
    mWinterBrightRedAddition = -20
    mWinterBrightGreenAddition = -14
    mWinterBrightBlueAddition = -12
    mWinterGreenishConditionBlueIntegerFactor = 7
    mWinterGreenishConditionGreenIntegerFactor = 5
    mWinterGreenishRedAddition = -13
    mWinterGreenishGreenAddition = -25
    mWinterGreenishBlueAddition = 0
    mWinterRestRedAddition = 0
    mWinterRestGreenAddition = -12
    mWinterRestBlueAddition = 0
    mAutumnDarkConditionRGBSumLessThanValue = 288
    mAutumnDarkConditionRGBSumLargerThanValue = 18
    mAutumnDarkRedAddition = 9
    mAutumnDarkGreenAddition = -8
    mAutumnDarkBlueAddition = 8
    mAutumnBrightConditionRGBSumLargerEqualThanValue = 288
    mAutumnBrightConditionRGBSumLessThanValue = 752
    mAutumnBrightRedAddition = -16
    mAutumnBrightGreenAddition = -10
    mAutumnBrightBlueAddition = -7
    mAutumnGreenishConditionBlueIntegerFactor = 7
    mAutumnGreenishConditionGreenIntegerFactor = 5
    mAutumnGreenishRedAddition = -9
    mAutumnGreenishGreenAddition = -20
    mAutumnGreenishBlueAddition = 0
    mAutumnRestRedAddition = 0
    mAutumnRestGreenAddition = -16
    mAutumnRestBlueAddition = 0
    mSpringDarkConditionRGBSumLessThanValue = 288
    mSpringDarkConditionRGBSumLargerThanValue = 18
    mSpringDarkRedAddition = 9
    mSpringDarkGreenAddition = -8
    mSpringDarkBlueAddition = 8
    mSpringBrightConditionRGBSumLargerEqualThanValue = 288
    mSpringBrightConditionRGBSumLessThanValue = 752
    mSpringBrightRedAddition = 15
    mSpringBrightGreenAddition = 10
    mSpringBrightBlueAddition = -10
    mSpringGreenishConditionBlueIntegerFactor = 7
    mSpringGreenishConditionGreenIntegerFactor = 5
    mSpringGreenishRedAddition = 10
    mSpringGreenishGreenAddition = 5
    mSpringGreenishBlueAddition = -5
    mSpringRestRedAddition = 0
    mSpringRestGreenAddition = 0
    mSpringRestBlueAddition = 0
    mNightStreetGreyConditionGreyToleranceValue = 11
    mNightStreetConditionRGBSumLessEqualThanValue = 510
    mNightStreetConditionRGBSumLargerThanValue = 0
    mNightStreetLightDots1DitherProbabily = 0.01
    mNightStreetLightDots2DitherProbabily = 0.02
    mNightStreetLightDots3DitherProbabily = 0.05
    mNightStreetLightDot1Red = UCHAR_MAX
    mNightStreetLightDot1Green = UCHAR_MAX
    mNightStreetLightDot1Blue = UCHAR_MAX
    mNightStreetLightDot2Red = UCHAR_MAX
    mNightStreetLightDot2Green = 200
    mNightStreetLightDot2Blue = 140
    mNightStreetLightDot3Red = UCHAR_MAX
    mNightStreetLightDot3Green = 180
    mNightStreetLightDot3Blue = 80
    mNightStreetRedAddition = 100
    mNightStreetGreenAddition = 50
    mNightStreetBlueAddition = -50
    mNightNonStreetLightness = 0.5
    mSpareOutWaterForSeasonsGeneration = False
    mNoSnowInWaterForWinterAndHardWinter = False
    mHardWinterStreetsConditionOn = True


C = MasksConfig


# --- helpers -----------------------------------------------------------------------------

def _load_rgb(img_name):
    """Return (H, W, 3) int32 array of the image."""
    with Image.open(img_name) as im:
        return np.asarray(im.convert('RGB'), dtype=np.int32)


def _load_water_mask(mask_img_path, shape_hw):
    """Boolean (H, W) array, True where the mask pixel is all black (== water).
    Mirrors WaterPixelChecker: a None or missing mask means 'nothing is water'."""
    if not mask_img_path or not os.path.isfile(mask_img_path):
        return np.zeros(shape_hw, dtype=bool)
    with Image.open(mask_img_path) as im:
        im = im.convert('RGB')
        if im.size != (shape_hw[1], shape_hw[0]):
            # The C++ code indexed the mask with the texture's coordinates and assumed equal
            # size; resize (nearest) so the check stays defined for mismatched inputs.
            im = im.resize((shape_hw[1], shape_hw[0]), Image.NEAREST)
        m = np.asarray(im)
    return (m[..., 0] == 0) & (m[..., 1] == 0) & (m[..., 2] == 0)


def _trunc(x):
    """C-style (int32_t)(float) conversion: truncation toward zero."""
    return np.trunc(x).astype(np.int32)


def _rand(shape):
    return np.random.random_sample(shape)


def _grey_condition(r, g, b, tol):
    """|r-b| < tol and |r-g| < tol and |g-b| < tol (strict, like the C++ code)."""
    return (np.abs(r - b) < tol) & (np.abs(r - g) < tol) & (np.abs(g - b) < tol)


def _save_bmp(rgb, out_name):
    """limitRGBValues + write a 24-bit BMP (equivalent of ImageMagick's BMP3: output)."""
    out = np.clip(rgb, 0, UCHAR_MAX).astype(np.uint8)
    Image.fromarray(out, 'RGB').save(out_name, format='BMP')


def _add(r, g, b, mask, dr, dg, db):
    r[mask] += dr
    g[mask] += dg
    b[mask] += db


# --- the five transforms (ports of c_create_* in FSET_ports.cpp) ---------------------------

def create_night(img_name, out_name, mask_img_path):
    try:
        rgb = _load_rgb(img_name)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        water = _load_water_mask(mask_img_path, r.shape)
        s = r + g + b

        street = (~water
                  & _grey_condition(r, g, b, C.mNightStreetGreyConditionGreyToleranceValue)
                  & (s > C.mNightStreetConditionRGBSumLargerThanValue)
                  & (s <= C.mNightStreetConditionRGBSumLessEqualThanValue))

        # default: normal land / water, factor 2 darker
        nr = _trunc(C.mNightNonStreetLightness * r.astype(np.float32))
        ng = _trunc(C.mNightNonStreetLightness * g.astype(np.float32))
        nb = _trunc(C.mNightNonStreetLightness * b.astype(np.float32))

        # street pixels: three sequential independent dither draws, else brighten / orange
        d1 = _rand(r.shape) < C.mNightStreetLightDots1DitherProbabily
        d2 = _rand(r.shape) < C.mNightStreetLightDots2DitherProbabily
        d3 = _rand(r.shape) < C.mNightStreetLightDots3DitherProbabily
        dot1 = street & d1
        dot2 = street & ~d1 & d2
        dot3 = street & ~d1 & ~d2 & d3
        rest = street & ~d1 & ~d2 & ~d3

        nr[rest] = r[rest] + C.mNightStreetRedAddition
        ng[rest] = g[rest] + C.mNightStreetGreenAddition
        nb[rest] = b[rest] + C.mNightStreetBlueAddition
        nr[dot1], ng[dot1], nb[dot1] = C.mNightStreetLightDot1Red, C.mNightStreetLightDot1Green, C.mNightStreetLightDot1Blue
        nr[dot2], ng[dot2], nb[dot2] = C.mNightStreetLightDot2Red, C.mNightStreetLightDot2Green, C.mNightStreetLightDot2Blue
        nr[dot3], ng[dot3], nb[dot3] = C.mNightStreetLightDot3Red, C.mNightStreetLightDot3Green, C.mNightStreetLightDot3Blue

        _save_bmp(np.stack([nr, ng, nb], axis=-1), out_name)
    except Exception as e:
        print("Caught exception:", e)


def create_hard_winter(img_name, out_name, mask_img_path):
    try:
        rgb = _load_rgb(img_name)
        r, g, b = rgb[..., 0].copy(), rgb[..., 1].copy(), rgb[..., 2].copy()
        r0, g0, b0 = r.copy(), g.copy(), b.copy()
        water = _load_water_mask(mask_img_path, r.shape)
        s = r0 + g0 + b0
        shape = r.shape

        dont_alter = C.mSpareOutWaterForSeasonsGeneration & water
        snow_allowed = ~(C.mNoSnowInWaterForWinterAndHardWinter & water)
        active = ~dont_alter
        v_streets = True  # constant in the C++ source

        # branch 1: greyish street pixels
        street = (active & C.mHardWinterStreetsConditionOn
                  & _grey_condition(r0, g0, b0, C.mHardWinterStreetConditionGreyToleranceValue)
                  & (s > C.mHardWinterStreetConditionRGBSumLargerThanValue)
                  & (s < C.mHardWinterStreetConditionRGBSumLessThanValue))
        # branch 2: very dark
        dark = active & ~street & (s < C.mHardWinterDarkConditionRGBSumLessThanValue)
        # branch 3: almost white
        almost_white = active & ~street & ~dark & (s >= C.mHardWinterAlmostWhiteConditionRGBSumLargerEqualThanValue)
        # branch 4: rest
        rest = active & ~street & ~dark & ~almost_white

        # --- branch 1
        if street.any():
            avg = s[street].astype(np.float32) / np.float32(3.0)
            f = C.mHardWinterStreetAverageFactor
            rf, ro = C.mHardWinterStreetAverageAdditionRandomFactor, C.mHardWinterStreetAverageAdditionRandomOffset
            n = int(street.sum())
            r[street] = _trunc(f * (avg + (_rand(n).astype(np.float32) * rf + ro)) + C.mHardWinterStreetAverageRedOffset)
            g[street] = _trunc(f * (avg + (_rand(n).astype(np.float32) * rf + ro)) + C.mHardWinterStreetAverageGreenOffset)
            b[street] = _trunc(f * (avg + (_rand(n).astype(np.float32) * rf + ro)) + C.mHardWinterStreetAverageBlueOffset)

        # --- branch 2
        if dark.any():
            sprinkle = (dark & snow_allowed
                        & (g0 > (r0 - C.mHardWinterDarkConditionRGDiffValue))
                        & (g0 > b0)
                        & (_rand(shape) < C.mHardWinterDarkConditionRandomLessThanValue))
            keep = dark & ~sprinkle
            n = int(sprinkle.sum())
            rf = C.mHardWinterDarkRandomFactor
            r[sprinkle] = C.mHardWinterDarkRedOffset + _trunc(_rand(n).astype(np.float32) * rf)
            g[sprinkle] = C.mHardWinterDarkGreenOffset + _trunc(_rand(n).astype(np.float32) * rf)
            b[sprinkle] = C.mHardWinterDarkBlueOffset + _trunc(_rand(n).astype(np.float32) * rf)
            factor = C.mHardWinterVeryDarkStreetFactor if v_streets else C.mHardWinterVeryDarkNormalFactor
            r[keep] = _trunc(factor * r0[keep].astype(np.float32))
            g[keep] = _trunc(factor * g0[keep].astype(np.float32))
            b[keep] = _trunc(factor * b0[keep].astype(np.float32))

        # --- branch 3
        aw = almost_white & (s <= C.mHardWinterAlmostWhiteConditionRGBSumLessEqualThanValue)
        if aw.any():
            r[aw] = _trunc(C.mHardWinterAlmostWhiteRedFactor * r0[aw].astype(np.float32))
            g[aw] = _trunc(C.mHardWinterAlmostWhiteGreenFactor * g0[aw].astype(np.float32))
            b[aw] = _trunc(C.mHardWinterAlmostWhiteBlueFactor * b0[aw].astype(np.float32))

        # --- branch 4
        if rest.any():
            reddish = (rest & snow_allowed
                       & ((r0 - C.mHardWinterRestConditionRGDiffValue) > g0)
                       & (r0 > b0))
            greenish = (rest & ~reddish
                        & (g0 >= (r0 - C.mHardWinterRestCondition2RGDiffValue))
                        & (g0 >= b0))
            bluish = rest & ~reddish & ~greenish

            rr = np.maximum(r0[reddish], C.mHardWinterRestRedMin)
            r[reddish] = rr
            g[reddish] = rr + C.mHardWinterRestGBOffsetToRed
            b[reddish] = rr + C.mHardWinterRestGBOffsetToRed

            forest = greenish & (~snow_allowed | (s < C.mHardWinterRestForestConditionRGBSumLessThan))
            nonforest = greenish & ~forest
            g[forest] = g0[forest] + C.mHardWinterRestForestGreenOffset
            gg = np.maximum(g0[nonforest], C.mHardWinterRestNonForestGreenLimit)
            g[nonforest] = gg
            r[nonforest] = gg + C.mHardWinterRestNonForestRedOffsetToGreen
            b[nonforest] = gg + C.mHardWinterRestNonForestBlueOffsetToGreen

            bl = bluish & snow_allowed
            bb = np.maximum(b0[bl], C.mHardWinterRestRestBlueMin)
            b[bl] = bb
            r[bl] = bb + C.mHardWinterRestRestRGToBlueOffset
            g[bl] = bb + C.mHardWinterRestRestRGToBlueOffset

        _save_bmp(np.stack([r, g, b], axis=-1), out_name)
    except Exception as e:
        print("Caught exception:", e)


def _seasonal_additive(img_name, out_name, mask_img_path, P):
    """Shared body of create_autumn / create_spring (dark / bright / greenish / rest)."""
    rgb = _load_rgb(img_name)
    r, g, b = rgb[..., 0].copy(), rgb[..., 1].copy(), rgb[..., 2].copy()
    water = _load_water_mask(mask_img_path, r.shape)
    s = r + g + b
    active = ~(C.mSpareOutWaterForSeasonsGeneration & water)

    dark = active & (s < P['dark_lt']) & (s > P['dark_gt'])
    bright = active & ~dark & (s >= P['bright_ge']) & (s < P['bright_lt'])
    greenish = active & ~dark & ~bright & ((P['green_bf'] * b) < (P['green_gf'] * g))
    rest = active & ~dark & ~bright & ~greenish

    _add(r, g, b, dark, *P['dark_add'])
    _add(r, g, b, bright, *P['bright_add'])
    _add(r, g, b, greenish, *P['green_add'])
    _add(r, g, b, rest, *P['rest_add'])
    _save_bmp(np.stack([r, g, b], axis=-1), out_name)


def create_autumn(img_name, out_name, mask_img_path):
    try:
        _seasonal_additive(img_name, out_name, mask_img_path, dict(
            dark_lt=C.mAutumnDarkConditionRGBSumLessThanValue, dark_gt=C.mAutumnDarkConditionRGBSumLargerThanValue,
            dark_add=(C.mAutumnDarkRedAddition, C.mAutumnDarkGreenAddition, C.mAutumnDarkBlueAddition),
            bright_ge=C.mAutumnBrightConditionRGBSumLargerEqualThanValue, bright_lt=C.mAutumnBrightConditionRGBSumLessThanValue,
            bright_add=(C.mAutumnBrightRedAddition, C.mAutumnBrightGreenAddition, C.mAutumnBrightBlueAddition),
            green_bf=C.mAutumnGreenishConditionBlueIntegerFactor, green_gf=C.mAutumnGreenishConditionGreenIntegerFactor,
            green_add=(C.mAutumnGreenishRedAddition, C.mAutumnGreenishGreenAddition, C.mAutumnGreenishBlueAddition),
            rest_add=(C.mAutumnRestRedAddition, C.mAutumnRestGreenAddition, C.mAutumnRestBlueAddition)))
    except Exception as e:
        print("Caught exception:", e)


def create_spring(img_name, out_name, mask_img_path):
    try:
        _seasonal_additive(img_name, out_name, mask_img_path, dict(
            dark_lt=C.mSpringDarkConditionRGBSumLessThanValue, dark_gt=C.mSpringDarkConditionRGBSumLargerThanValue,
            dark_add=(C.mSpringDarkRedAddition, C.mSpringDarkGreenAddition, C.mSpringDarkBlueAddition),
            bright_ge=C.mSpringBrightConditionRGBSumLargerEqualThanValue, bright_lt=C.mSpringBrightConditionRGBSumLessThanValue,
            bright_add=(C.mSpringBrightRedAddition, C.mSpringBrightGreenAddition, C.mSpringBrightBlueAddition),
            green_bf=C.mSpringGreenishConditionBlueIntegerFactor, green_gf=C.mSpringGreenishConditionGreenIntegerFactor,
            green_add=(C.mSpringGreenishRedAddition, C.mSpringGreenishGreenAddition, C.mSpringGreenishBlueAddition),
            rest_add=(C.mSpringRestRedAddition, C.mSpringRestGreenAddition, C.mSpringRestBlueAddition)))
    except Exception as e:
        print("Caught exception:", e)


def create_winter(img_name, out_name, mask_img_path):
    try:
        rgb = _load_rgb(img_name)
        r, g, b = rgb[..., 0].copy(), rgb[..., 1].copy(), rgb[..., 2].copy()
        r0, g0, b0 = r.copy(), g.copy(), b.copy()
        water = _load_water_mask(mask_img_path, r.shape)
        s = r0 + g0 + b0
        active = ~(C.mSpareOutWaterForSeasonsGeneration & water)

        street = (active
                  & _grey_condition(r0, g0, b0, C.mWinterStreetGreyConditionGreyToleranceValue)
                  & (s > C.mWinterStreetGreyConditionRGBSumLargerThanValue))
        dark = (active & ~street
                & (s < C.mWinterDarkConditionRGBSumLessThanValue) & (s > C.mWinterDarkConditionRGBSumLargerThanValue))
        bright = (active & ~street & ~dark
                  & (s >= C.mWinterBrightConditionRGBSumLargerEqualThanValue) & (s < C.mWinterBrightConditionRGBSumLessThanValue))
        greenish = (active & ~street & ~dark & ~bright
                    & ((C.mWinterGreenishConditionBlueIntegerFactor * b0) < (C.mWinterGreenishConditionGreenIntegerFactor * g0)))
        rest = active & ~street & ~dark & ~bright & ~greenish

        if street.any():
            vmax = np.maximum(np.maximum(r0[street], g0[street]), b0[street]).astype(np.float32)
            n = int(street.sum())
            rf = C.mWinterStreetGreyRandomFactor
            r[street] = _trunc(_rand(n).astype(np.float32) * rf + vmax)
            g[street] = _trunc(_rand(n).astype(np.float32) * rf + vmax)
            b[street] = _trunc(_rand(n).astype(np.float32) * rf + vmax)

        _add(r, g, b, dark, C.mWinterDarkRedAddition, C.mWinterDarkGreenAddition, C.mWinterDarkBlueAddition)
        _add(r, g, b, bright, C.mWinterBrightRedAddition, C.mWinterBrightGreenAddition, C.mWinterBrightBlueAddition)
        _add(r, g, b, greenish, C.mWinterGreenishRedAddition, C.mWinterGreenishGreenAddition, C.mWinterGreenishBlueAddition)
        _add(r, g, b, rest, C.mWinterRestRedAddition, C.mWinterRestGreenAddition, C.mWinterRestBlueAddition)

        _save_bmp(np.stack([r, g, b], axis=-1), out_name)
    except Exception as e:
        print("Caught exception:", e)

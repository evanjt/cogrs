//! Safe numeric casting utilities for geospatial image processing.
//!
//! This module provides safe conversion functions and documents our assumptions
//! about numeric ranges in the context of COG/GeoTIFF processing.
//!
//! # Design Decisions
//!
//! ## Image Dimensions (`usize` ↔ `f64`)
//! We allow `usize` to `f64` conversions without explicit checks because:
//! - Maximum practical image dimension: ~1 billion pixels per side
//! - `f64` mantissa: 52 bits, can exactly represent integers up to 2^53
//! - No real-world raster will exceed this limit
//!
//! ## Pixel Coordinates (`f64` → `usize`)
//! Float-to-integer conversions for pixel indices require bounds checking
//! because the float may be negative or exceed valid dimensions.
//!
//! ## File Offsets (`u64` → `usize`)
//! On 32-bit systems, file offsets > 4GB will fail. We use `TryFrom` for these.
//!
//! ## Pixel Values (integer → `f32`)
//! Converting integer pixel values to `f32` is intentional - we accept
//! precision loss for values > 2^24 as this is standard practice.

use std::convert::TryFrom;

/// Convert a `u64` file offset to `usize`, failing on 32-bit overflow.
///
/// # Errors
/// Returns an error string if the value exceeds `usize::MAX` (on 32-bit systems).
#[inline]
pub fn u64_to_usize(value: u64) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| {
        format!(
            "File offset {value} exceeds maximum addressable size on this platform"
        )
    })
}

/// Convert a `u32` to `u16`, failing on overflow.
///
/// # Errors
/// Returns an error string if the value exceeds `u16::MAX`.
#[inline]
pub fn u32_to_u16(value: u32) -> Result<u16, String> {
    u16::try_from(value).map_err(|_| format!("Value {value} exceeds u16 maximum (65535)"))
}

/// Convert a `usize` to `u32`, failing on 64-bit overflow.
///
/// # Errors
/// Returns an error string if the value exceeds `u32::MAX`.
#[inline]
pub fn usize_to_u32(value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("Value {value} exceeds u32 maximum"))
}

/// Convert a `u32` to `i32`, failing if it would wrap.
///
/// # Errors
/// Returns an error string if the value exceeds `i32::MAX`.
#[inline]
pub fn u32_to_i32(value: u32) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("Value {value} exceeds i32 maximum"))
}

/// Convert a float to a pixel index, returning `None` if out of bounds.
///
/// This function handles:
/// - Negative values (returns `None`)
/// - Values exceeding `max_value` (returns `None`)
/// - NaN values (returns `None`)
///
/// # Arguments
/// * `value` - The floating point coordinate
/// * `max_value` - The maximum valid index (exclusive)
#[inline]
#[must_use] 
pub fn f64_to_pixel_index(value: f64, max_value: usize) -> Option<usize> {
    if value.is_nan() || value < 0.0 {
        return None;
    }
    // Safety: we've already checked value >= 0 and is not NaN above
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = value as usize;
    if index >= max_value {
        None
    } else {
        Some(index)
    }
}

/// Convert a float to a clamped pixel index within valid bounds.
///
/// Unlike `f64_to_pixel_index`, this clamps to valid range instead of returning `None`.
///
/// # Arguments
/// * `value` - The floating point coordinate
/// * `max_value` - The maximum valid index (exclusive)
#[inline]
#[must_use] 
pub fn f64_to_clamped_pixel(value: f64, max_value: usize) -> usize {
    if value.is_nan() || value < 0.0 {
        return 0;
    }
    // Safety: we've already checked value >= 0 and is not NaN above
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = value as usize;
    if index >= max_value {
        max_value.saturating_sub(1)
    } else {
        index
    }
}

/// Convert a signed offset to unsigned, returning `None` if negative.
#[inline]
#[must_use] 
pub fn isize_to_usize(value: isize) -> Option<usize> {
    if value < 0 {
        None
    } else {
        // Safety: we've already checked value >= 0 above
        #[allow(clippy::cast_sign_loss)]
        Some(value as usize)
    }
}

/// Clamp an isize to valid usize range (0 to max-1).
#[inline]
#[must_use] 
pub fn isize_to_clamped_usize(value: isize, max_value: usize) -> usize {
    if value < 0 {
        0
    } else {
        // Safety: we've already checked value >= 0 above
        #[allow(clippy::cast_sign_loss)]
        let u = value as usize;
        if u >= max_value {
            max_value.saturating_sub(1)
        } else {
            u
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_to_usize() {
        assert!(u64_to_usize(0).is_ok());
        assert!(u64_to_usize(1000).is_ok());
        // On 64-bit, this should pass; on 32-bit it would fail
        #[cfg(target_pointer_width = "64")]
        assert!(u64_to_usize(u64::MAX).is_ok());
    }

    #[test]
    fn test_u32_to_u16() {
        assert_eq!(u32_to_u16(0), Ok(0));
        assert_eq!(u32_to_u16(65535), Ok(65535));
        assert!(u32_to_u16(65536).is_err());
    }

    #[test]
    fn test_f64_to_pixel_index() {
        assert_eq!(f64_to_pixel_index(0.0, 100), Some(0));
        assert_eq!(f64_to_pixel_index(50.5, 100), Some(50));
        assert_eq!(f64_to_pixel_index(99.9, 100), Some(99));
        assert_eq!(f64_to_pixel_index(100.0, 100), None);
        assert_eq!(f64_to_pixel_index(-1.0, 100), None);
        assert_eq!(f64_to_pixel_index(f64::NAN, 100), None);
    }

    #[test]
    fn test_f64_to_clamped_pixel() {
        assert_eq!(f64_to_clamped_pixel(0.0, 100), 0);
        assert_eq!(f64_to_clamped_pixel(50.5, 100), 50);
        assert_eq!(f64_to_clamped_pixel(150.0, 100), 99);
        assert_eq!(f64_to_clamped_pixel(-10.0, 100), 0);
        assert_eq!(f64_to_clamped_pixel(f64::NAN, 100), 0);
    }

    #[test]
    fn test_isize_to_usize() {
        assert_eq!(isize_to_usize(0), Some(0));
        assert_eq!(isize_to_usize(100), Some(100));
        assert_eq!(isize_to_usize(-1), None);
    }

    #[test]
    fn test_isize_to_clamped_usize() {
        assert_eq!(isize_to_clamped_usize(50, 100), 50);
        assert_eq!(isize_to_clamped_usize(-10, 100), 0);
        assert_eq!(isize_to_clamped_usize(150, 100), 99);
    }
}

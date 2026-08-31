/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use tap::{Pipe, Tap};

pub(crate) struct BCursor<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) cursor: usize,
}

macro_rules! impl_capture_generic {
    ($name:ident, $T:ty, $B:expr) => {
        pub fn $name(&mut self) -> $T {
            self.capture_exact::<$B>().pipe(<$T>::from_le_bytes)
        }
    };
}

macro_rules! impl_try_capture_generic {
    ($name:ident, $T:ty, $B:expr) => {
        pub fn $name(&mut self) -> anyhow::Result<$T> {
            self.try_capture_exact::<$B>().map(<$T>::from_le_bytes)
        }
    };
}

impl<'a> BCursor<'a> {
    /// Creates a new byte-cursor by wrapping borrowed bytes and setting the cursor to zero.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            data: bytes,
            cursor: 0,
        }
    }

    pub fn seek(&mut self, by: usize) -> &'a [u8] {
        self.capture_exact::<2>();
        &self.data[self.cursor..self.cursor + by]
    }

    pub fn capture(&mut self, by: usize) -> &'a [u8] {
        (&self.data[self.cursor..self.cursor + by]).tap(|_| self.cursor += by)
    }

    pub fn capture_exact<const BY: usize>(&mut self) -> [u8; BY] {
        [0_u8; BY].tap_mut(|bytes| bytes.copy_from_slice(self.capture(BY)))
    }

    impl_capture_generic!(capture_u8, u8, 1);
    impl_capture_generic!(capture_i8, i8, 1);
    impl_capture_generic!(capture_u16, u16, 2);
    impl_capture_generic!(capture_i16, i16, 2);
    impl_capture_generic!(capture_u32, u32, 4);
    impl_capture_generic!(capture_i32, i32, 4);
    impl_capture_generic!(capture_u64, u64, 8);
    impl_capture_generic!(capture_i64, i64, 8);
    impl_capture_generic!(capture_usize, usize, { (usize::BITS / 8) as usize });
    impl_capture_generic!(capture_isize, isize, { (isize::BITS / 8) as usize });
    impl_capture_generic!(capture_f32, f32, 4);
    impl_capture_generic!(capture_f64, f64, 8);

    /// Similar to [Self::try_capture], except the cursor isn't moved.
    pub fn try_seek(&mut self, by: usize) -> anyhow::Result<&'a [u8]> {
        self.data
            .get(self.cursor..self.cursor + by)
            .ok_or_else(|| anyhow::anyhow!("Failed to seek {by} bytes, out of bounds"))
    }

    /// Attempts to advance the cursor by a specified amount, capturing the bytes in between.
    /// Returns an error if and only if we try to cross out of bounds.
    pub fn try_capture(&mut self, by: usize) -> anyhow::Result<&'a [u8]> {
        self.data
            .get(self.cursor..self.cursor + by)
            .inspect(|_| self.cursor += by)
            .ok_or_else(|| anyhow::anyhow!("Failed to capture {by} bytes, out of bounds"))
    }

    /// Similar to [Self::try_capture], except we specify the amount to try advancing the
    /// cursor with using a compile-time constant generic, in order to get a known-size array in return.
    pub fn try_capture_exact<const BY: usize>(&mut self) -> anyhow::Result<[u8; BY]> {
        self.try_capture(BY)
            .map(|slice| [0_u8; BY].tap_mut(|bytes| bytes.copy_from_slice(slice)))
    }

    impl_try_capture_generic!(try_capture_u8, u8, 1);
    impl_try_capture_generic!(try_capture_i8, i8, 1);
    impl_try_capture_generic!(try_capture_u16, u16, 2);
    impl_try_capture_generic!(try_capture_i16, i16, 2);
    impl_try_capture_generic!(try_capture_u32, u32, 4);
    impl_try_capture_generic!(try_capture_i32, i32, 4);
    impl_try_capture_generic!(try_capture_u64, u64, 8);
    impl_try_capture_generic!(try_capture_i64, i64, 8);
    impl_try_capture_generic!(try_capture_usize, usize, { (usize::BITS / 8) as usize });
    impl_try_capture_generic!(try_capture_isize, isize, { (isize::BITS / 8) as usize });
    impl_try_capture_generic!(try_capture_f32, f32, 4);
    impl_try_capture_generic!(try_capture_f64, f64, 8);

    pub fn try_capture_f64_duo(&mut self) -> anyhow::Result<(f64, f64)> {
        Ok((self.try_capture_f64()?, self.try_capture_f64()?))
    }

    pub fn try_capture_f64_trio(&mut self) -> anyhow::Result<(f64, f64, f64)> {
        Ok((
            self.try_capture_f64()?,
            self.try_capture_f64()?,
            self.try_capture_f64()?,
        ))
    }

    #[cfg(target_endian = "little")]
    pub fn try_capture_f32_array(&mut self, length: usize) -> anyhow::Result<&'a [f32]> {
        self.try_capture(length * 4).map(bytemuck::cast_slice)
    }

    #[cfg(not(target_endian = "little"))]
    pub fn try_capture_f32_array(&mut self, length: usize) -> anyhow::Result<Vec<f32>> {
        self.try_capture(length * 4)
            .map(|slice| unsafe { slice.as_chunks_unchecked::<4>() })
            .map(|chunks| chunks.iter().copied().map(f32::from_le_bytes).collect())
    }
}

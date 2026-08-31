/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use std::mem::{MaybeUninit, transmute};
use tap::{Pipe, Tap};

pub(crate) struct BWriter {
    pub(crate) data: Vec<MaybeUninit<u8>>,
    pub(crate) cursor: usize,
}

macro_rules! impl_write_generic {
    ($name:ident, $T:ty, $B:expr) => {
        pub fn $name(&mut self, src: $T) {
            self.data[self.cursor..self.cursor + $B].write_copy_of_slice(&src.to_le_bytes());
            self.cursor += $B;
        }
    };
}

impl BWriter {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::<MaybeUninit<u8>>::with_capacity(capacity)
                .tap_mut(|vec| unsafe { vec.set_len(vec.capacity()) }),
            cursor: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        debug_assert_eq!(self.data.len(), self.data.capacity());
        self.data.capacity()
    }

    pub fn rem_capacity(&self) -> usize {
        self.data.capacity() - self.cursor
    }

    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
        unsafe { self.data.set_len(self.data.capacity()) };
    }

    pub fn expand(&mut self) {
        self.reserve(self.capacity().max(512));
    }

    // Does not modify/zeroize data, just sets the cursor to zero
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Zeroize data that is skipped and returns its range
    pub fn skip(&mut self, n: usize) -> std::range::Range<usize> {
        std::range::Range {
            start: self.cursor,
            end: self.cursor + n,
        }
        .tap(|range| {
            self.data[*range].fill(MaybeUninit::new(0));
            self.cursor = range.end;
        })
    }

    pub fn fill(&mut self, val: u8, n: usize) {
        self.data[self.cursor..self.cursor + n].fill(MaybeUninit::new(val));
        self.cursor += n;
    }

    pub fn write_slice<T: AsRef<[u8]>>(&mut self, src: T) {
        let src = src.as_ref();
        self.data[self.cursor..self.cursor + src.len()].write_copy_of_slice(src);
        self.cursor += src.len();
    }

    pub fn write_slice_at<T: AsRef<[u8]>>(&mut self, start_idx: usize, src: T) {
        let src = src.as_ref();
        if start_idx > self.cursor {
            panic!("Starting index cannot be larger than the current cursor position");
        }
        let end_idx = start_idx + src.len();
        self.data[start_idx..end_idx].write_copy_of_slice(src);
        self.cursor = usize::max(self.cursor, end_idx)
    }

    impl_write_generic!(write_u8, u8, 1);
    impl_write_generic!(write_i8, i8, 1);

    impl_write_generic!(write_u16, u16, 2);
    impl_write_generic!(write_i16, i16, 2);

    impl_write_generic!(write_u32, u32, 4);
    impl_write_generic!(write_i32, i32, 4);

    impl_write_generic!(write_u64, u64, 8);
    impl_write_generic!(write_i64, i64, 8);

    impl_write_generic!(write_usize, usize, (usize::BITS / 8) as usize);
    impl_write_generic!(write_isize, isize, (isize::BITS / 8) as usize);

    impl_write_generic!(write_f32, f32, 4);
    impl_write_generic!(write_f64, f64, 8);

    pub fn slice(&self, range: impl Into<std::range::Range<usize>>) -> &[u8] {
        let range = range.into();
        assert!(range.end <= self.cursor);
        unsafe { transmute(&self.data[range]) }
    }

    pub fn slice_from(&self, range: impl Into<std::range::RangeFrom<usize>>) -> &[u8] {
        let range = range
            .into()
            .pipe(|rf| std::range::Range::from(rf.start..self.cursor));
        unsafe { transmute(&self.data[range]) }
    }

    pub fn into_vec(mut self) -> Vec<u8> {
        unsafe {
            self.data.set_len(self.cursor);
            transmute(self.data)
        }
    }
}

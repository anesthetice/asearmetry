/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use crate::signal::SampleDisplay;

pub trait Sample:
    'static
    + Send
    + Sync
    + Clone
    + Copy
    + num_traits::Num
    + num_traits::NumAssign
    + num_traits::NumAssignRef
    + num_traits::FromPrimitive
    + core::fmt::Debug
    + core::fmt::Display
    + SampleDisplay
    + core::iter::Sum<Self>
where
    Self::Inner: Sample,
{
    type Inner;
    fn as_inner(&self) -> &Self::Inner;
    fn into_inner(self) -> Self::Inner;
}

impl<T> Sample for T
where
    T: 'static
        + Send
        + Sync
        + Clone
        + Copy
        + num_traits::Num
        + num_traits::NumAssign
        + num_traits::NumAssignRef
        + num_traits::FromPrimitive
        + core::fmt::Debug
        + core::fmt::Display
        + SampleDisplay
        + core::iter::Sum<Self>,
{
    type Inner = T;
    fn as_inner(&self) -> &Self::Inner {
        self
    }
    fn into_inner(self) -> Self::Inner {
        self
    }
}

pub trait IsKnownSampleType {
    const SAMPLE_TYPE: KnownSampleType;
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnownSampleType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    U128,
    I128,
    USIZE,
    ISIZE,
    F32,
    F64,
    CF32,
    CF64,
}

macro_rules! impl_is_known_sample_type {
    ($T:ty, $V:ident) => {
        impl IsKnownSampleType for $T {
            const SAMPLE_TYPE: KnownSampleType = KnownSampleType::$V;
        }
    };
}

impl_is_known_sample_type!(u8, U8);
impl_is_known_sample_type!(i8, I8);
impl_is_known_sample_type!(u16, U16);
impl_is_known_sample_type!(i16, I16);
impl_is_known_sample_type!(u32, U32);
impl_is_known_sample_type!(i32, I32);
impl_is_known_sample_type!(u64, U64);
impl_is_known_sample_type!(i64, I64);
impl_is_known_sample_type!(u128, U128);
impl_is_known_sample_type!(i128, I128);
impl_is_known_sample_type!(usize, USIZE);
impl_is_known_sample_type!(isize, ISIZE);
impl_is_known_sample_type!(f32, F32);
impl_is_known_sample_type!(f64, F64);
impl_is_known_sample_type!(num_complex::Complex32, CF32);
impl_is_known_sample_type!(num_complex::Complex64, CF64);

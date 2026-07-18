/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::math::Hertz;

// Modules
mod borrowed;
mod common;
mod display;
mod dsp;
mod owned;

// Exports
pub(crate) use dsp::DSPUtils;
pub(crate) use dsp::DefinedLtiConvolution;
pub use dsp::{_convolve_lti, _convolve_ltv, DSP, LtvFilter};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeDomain {}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FreqDomain {}

/// Represents a potentially multichannel discrete signal
///
/// Samples are stored in a channel-major layout, with one
/// owned buffer per channel.
///
/// The constant generic parameter `C` refers to the number
/// of channels, the generic `S` refers to the type of the sample,
/// and the generic `D` signifies the domain of the signal,
/// currently `TimeDomain` or `FreqDomain`.
#[derive(Clone, PartialEq, PartialOrd)]
pub struct Signal<const C: usize, S: Sample, D: Domain> {
    pub channels: [Vec<S>; C],
    pub sampling_rate: Option<Hertz>,
    pub _domain: D,
}

/// A borrowed, channel-major view into sliced sample data.
/// This is the non-owning counterpart to [`Signal`].
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct SignalSlice<'data, const C: usize, S: Sample, D: Domain> {
    pub channels: [&'data [S]; C],
    pub sampling_rate: Option<Hertz>,
    pub _domain: D,
}

pub trait Sample:
    num_traits::Num
    + num_traits::NumAssign
    + num_traits::NumAssignRef
    + core::fmt::Debug
    + core::fmt::Display
    + core::iter::Sum<Self>
    + Clone
    + Copy
    + Send
    + Sync
    + 'static
{
}

impl<T> Sample for T where
    T: num_traits::Num
        + num_traits::NumAssign
        + num_traits::NumAssignRef
        + core::fmt::Debug
        + core::fmt::Display
        + core::iter::Sum<Self>
        + Clone
        + Copy
        + Send
        + Sync
        + 'static
{
}

pub trait Domain:
    core::fmt::Debug + core::fmt::Display + Clone + Copy + Default + PartialEq
{
}
impl Domain for TimeDomain {}
impl Domain for FreqDomain {}

impl core::fmt::Display for TimeDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("time")
    }
}

impl core::fmt::Display for FreqDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("frequency")
    }
}

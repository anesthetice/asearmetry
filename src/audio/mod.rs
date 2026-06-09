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
mod owned;
mod special;
mod traits;

// Exports
pub use special::{MonoAudioBuf, MonoAudioBufSlice, StereoAudioBuf, StereoAudioBufSlice};
pub(crate) use traits::DiscreteSignalUtils;
pub use traits::{_convolve_lti, _convolve_ltv, AudioSignal, DiscreteSignal, LtvFilter};

/// Represents a multichannel audio signal.
///
/// Stores audio or signal samples in a channel-major layout, with
/// one owned buffer per channel. All buffers are expected to have
/// the same length (most times). Samples are represented by 32-bit
/// floating point numbers.
///
/// The constant generic parameter `C` refers to the number of channels.
#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
pub struct AudioBuffer<const C: usize> {
    pub(crate) channels: [Vec<f32>; C],
    /// Sampling rate in Hertz.
    pub(crate) sampling_rate: Option<Hertz>,
}

/// A borrowed, channel-major view into sliced audio sample data.
/// This is the non-owning counterpart to [`AudioBuffer`].
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct AudioBufferSlice<'data, const C: usize> {
    pub(crate) channels: [&'data [f32]; C],
    /// Sampling rate in Hertz.
    pub(crate) sampling_rate: Option<Hertz>,
}

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
pub use traits::{AudioSignal, DiscreteSignal};

/// Represents a multichannel audio signal.
///
/// Stores audio or signal samples in a channel-major layout, with
/// one owned buffer per channel. All buffers are expected to have
/// the same length at all times. Samples are represented by 32-bit
/// floating point numbers.
///
/// The constant generic parameter `C` refers to the number of channels.
#[derive(Clone)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct AudioBuffer<const C: usize> {
    pub(crate) channels: [Vec<f32>; C],
    /// Sampling rate in Hertz.
    pub(crate) sampling_rate: Option<u32>,
}

/// A borrowed, channel-major view into sliced audio sample data.
/// This is the non-owning counterpart to [`AudioBuffer`].
#[derive(Clone, Copy)]
pub struct AudioBufferSlice<'a, const C: usize> {
    pub(crate) channels: [&'a [f32]; C],
    /// Sampling rate in Hertz.
    pub(crate) sampling_rate: Option<u32>,
}

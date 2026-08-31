/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
pub mod audio;
mod borrowed;
mod common;
mod core;
mod display;
mod dsp;
mod owned;

// Exports
pub use core::{
    AnyDomain, Domain, FreqDomain, IsKnownSampleType, KnownSampleType, Sample, Signal, SignalSlice,
    StftDomain, StftInfo, TimeDomain,
};
pub use display::SampleDisplay;
pub use dsp::{
    _convolve_lti, _convolve_ltv, _dft, _dft_rayon, _fft_full, _fft_halved, _idft, _idft_rayon,
    _ifft, DSP, LtvFilter,
};
pub(crate) use dsp::{DSPUtils, DefinedLtiConvolution};
pub use opusic_c::Bandwidth;

pub type TimeSignal<const C: usize, S> = Signal<C, S, TimeDomain>;
pub type TimeSignalSlice<'data, const C: usize, S> = SignalSlice<'data, C, S, TimeDomain>;
pub type FreqSignal<const C: usize, S> = Signal<C, S, FreqDomain>;
pub type FreqSignalSlice<'data, const C: usize, S> = SignalSlice<'data, C, S, FreqDomain>;

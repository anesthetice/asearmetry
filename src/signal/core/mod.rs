/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
mod domain;
mod sample;

// Exports
pub use domain::{AnyDomain, Domain, FreqDomain, TimeDomain};
pub use sample::Sample;

/// Represents a potentially multichannel discrete signal
///
/// Samples are stored in a channel-major layout, i.e. with one
/// vector of samples per channel. All vectors are expected to
/// have identical length.
///
/// The constant generic parameter `C` refers to the number of channels,
/// the generic `S` refers to the underlying sample type used,
/// and the generic `D` represents the domain of the signal,
/// which can currently be either `TimeDomain` or `FreqDomain`.
#[derive(Clone, PartialEq, PartialOrd)]
pub struct Signal<const C: usize, S: Sample, D: Domain> {
    pub channels: [Vec<S>; C],
    pub sampling_rate: Option<crate::math::Hertz>,
    pub _domain: D,
}

/// A borrowed, channel-major view into sliced sample data.
/// This is the non-owning counterpart to [`Signal`].
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct SignalSlice<'data, const C: usize, S: Sample, D: Domain> {
    pub channels: [&'data [S]; C],
    pub sampling_rate: Option<crate::math::Hertz>,
    pub _domain: D,
}

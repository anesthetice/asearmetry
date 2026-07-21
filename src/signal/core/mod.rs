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

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
mod asp;
mod special;

// Exports
pub use asp::ASP;
pub use special::{MonoAudioBuf, MonoAudioBufSlice, StereoAudioBuf, StereoAudioBufSlice};

// Imports
use crate::signal::{Signal, SignalSlice, TimeDomain};

pub type AudioBuffer<const C: usize> = Signal<C, f32, TimeDomain>;
pub type AudioBufferSlice<'data, const C: usize> = SignalSlice<'data, C, f32, TimeDomain>;

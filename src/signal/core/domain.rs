/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeDomain {}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FreqDomain {}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StftDomain {}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StftInfo {
    pub hop_length: usize,
    pub frame_length: usize,
    pub padding: (usize, usize),
}

pub trait Domain:
    'static
    + Send
    + Sync
    + Clone
    + Copy
    + Default
    + PartialEq
    + core::fmt::Debug
    + core::fmt::Display
    + Into<AnyDomain>
where
    Self::Inner: Domain,
{
    type Inner;
}

impl Domain for TimeDomain {
    type Inner = TimeDomain;
}

impl Domain for FreqDomain {
    type Inner = FreqDomain;
}

impl Domain for StftDomain {
    type Inner = StftDomain;
}

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

impl core::fmt::Display for StftDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("frequency (short time)")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnyDomain {
    Time,
    Freq,
    Stft,
}

impl From<TimeDomain> for AnyDomain {
    fn from(_: TimeDomain) -> Self {
        Self::Time
    }
}

impl From<FreqDomain> for AnyDomain {
    fn from(_: FreqDomain) -> Self {
        Self::Freq
    }
}

impl From<StftDomain> for AnyDomain {
    fn from(_: StftDomain) -> Self {
        Self::Stft
    }
}

impl core::fmt::Display for AnyDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Time => TimeDomain {}.fmt(f),
            Self::Freq => FreqDomain {}.fmt(f),
            Self::Stft => StftDomain {}.fmt(f),
        }
    }
}

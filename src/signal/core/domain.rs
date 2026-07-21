/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnyDomain {
    Time(TimeDomain),
    Freq(FreqDomain),
}

impl From<TimeDomain> for AnyDomain {
    fn from(value: TimeDomain) -> Self {
        Self::Time(value)
    }
}

impl From<FreqDomain> for AnyDomain {
    fn from(value: FreqDomain) -> Self {
        Self::Freq(value)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeDomain {}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(non_snake_case)]
pub struct FreqDomain {
    /// Number of samples in the time-domain (required for inverse transform),
    /// characterized by: N_ts ∈ {2⋅N_fs + 1, 2⋅N_fs + 2} (where N_ts is the
    /// number of time samples, and N_fs is the number of frequency samples)
    pub N_ts: usize,
}

pub trait Domain:
    core::fmt::Debug + core::fmt::Display + Clone + Copy + Default + PartialEq + Into<AnyDomain>
{
    type Inner;
    fn as_inner(&self) -> &Self::Inner;
    fn into_inner(self) -> Self::Inner;
}

impl Domain for TimeDomain {
    type Inner = TimeDomain;
    fn as_inner(&self) -> &Self::Inner {
        self
    }
    fn into_inner(self) -> Self::Inner {
        self
    }
}

impl Domain for FreqDomain {
    type Inner = FreqDomain;
    fn as_inner(&self) -> &Self::Inner {
        self
    }
    fn into_inner(self) -> Self::Inner {
        self
    }
}

impl core::fmt::Display for AnyDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Time(inner) => inner.fmt(f),
            Self::Freq(inner) => inner.fmt(f),
        }
    }
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

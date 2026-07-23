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

impl core::fmt::Display for AnyDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Time(inner) => inner.fmt(f),
            Self::Freq(inner) => inner.fmt(f),
        }
    }
}

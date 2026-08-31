/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use crate::{
    math::Hertz,
    signal::{DSP, Domain, Sample, Signal, SignalSlice, TimeDomain},
};
use itertools::Itertools;
use std::marker::PhantomData;

impl<'data, const C: usize, S: Sample, D: Domain> SignalSlice<'data, C, S, D> {
    pub fn new(channels: [&'data [S]; C], sampling_rate: Option<Hertz>) -> Self {
        Self {
            channels,
            sampling_rate,
            _domain: PhantomData,
        }
    }
    pub fn new_empty() -> Self {
        Self {
            channels: std::array::repeat(&[]),
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
    pub fn with_sr(mut self, sampling_rate: Hertz) -> Self {
        self.sampling_rate = Some(sampling_rate);
        self
    }
    pub fn with_sr_opt(mut self, sampling_rate: Option<Hertz>) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }
    pub fn stack_ref<'a, 'b, const C1: usize, const C2: usize>(
        a: SignalSlice<'a, C1, S, D>,
        b: SignalSlice<'b, C2, S, D>,
    ) -> Self
    where
        'a: 'data, // 'a outlives 'data
        'b: 'data, // 'b outlives 'data
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(a.sr(), b.sr());
        let channels: [&'data [S]; C] =
            itertools::chain!(a.channels.into_iter(), b.channels.into_iter(),)
                .collect_array()
                .unwrap();

        Self::new(channels, sampling_rate)
    }
    pub fn with_transform<S2, F>(self, op: F) -> Signal<C, S2, D>
    where
        S2: Sample,
        F: FnOnce([&[S]; C]) -> [Vec<S2>; C],
    {
        Signal {
            channels: op(self.channels),
            sampling_rate: self.sampling_rate,
            _domain: self._domain,
        }
    }

    pub fn with_transform_cha<S2, F>(self, op: F) -> Signal<C, S2, D>
    where
        S2: Sample,
        F: FnMut(&[S]) -> Vec<S2>,
    {
        Signal {
            channels: self.channels.map(op),
            sampling_rate: self.sampling_rate,
            _domain: self._domain,
        }
    }

    pub fn with_domain<D2: Domain>(&'data self, _: D2) -> SignalSlice<'data, C, S, D2> {
        SignalSlice {
            channels: self.channels,
            sampling_rate: self.sampling_rate,
            _domain: PhantomData,
        }
    }
}

impl<'data, const C: usize, S: Sample, D: Domain> From<&'data Signal<C, S, D>>
    for SignalSlice<'data, C, S, D>
{
    fn from(value: &'data Signal<C, S, D>) -> Self {
        Self {
            channels: std::array::from_fn(|i| value.channels[i].as_slice()),
            sampling_rate: value.sampling_rate,
            _domain: value._domain,
        }
    }
}

impl<'data, const C: usize, S: Sample, D: Domain> From<[&'data [S]; C]>
    for SignalSlice<'data, C, S, D>
{
    fn from(value: [&'data [S]; C]) -> Self {
        Self {
            channels: value,
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
}

impl<'data, const C: usize, S: Sample, D: Domain> From<([&'data [S]; C], Option<Hertz>)>
    for SignalSlice<'data, C, S, D>
{
    fn from(value: ([&'data [S]; C], Option<Hertz>)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: value.1,
            _domain: PhantomData,
        }
    }
}

impl<'data, const C: usize, S: Sample, D: Domain> From<([&'data [S]; C], Hertz)>
    for SignalSlice<'data, C, S, D>
{
    fn from(value: ([&'data [S]; C], Hertz)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1),
            _domain: PhantomData,
        }
    }
}

impl<'data, S: Sample, D: Domain> From<&'data [S]> for SignalSlice<'data, 1, S, D> {
    fn from(value: &'data [S]) -> Self {
        Self {
            channels: [value],
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
}

/*
// Maybe switch to nightly for specialization if this is actually useful
impl<'data, const C: usize, U: num_traits::AsPrimitive<u32>> From<([&'data [f32]; C], U)>
    for AudioBufferSlice<'data, C>
{
    fn from(value: ([&'data [f32]; C], U)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1.as_()),
        }
    }
}
*/

impl<'data, const C: usize, S, D, T> PartialEq<T> for SignalSlice<'data, C, S, D>
where
    S: Sample,
    D: Domain,
    T: DSP<C, S, D>,
{
    fn eq(&self, other: &T) -> bool {
        self.chas().eq(&other._chas())
            && self.sampling_rate.eq(&other.sampling_rate())
            && self.domain().eq(&other.domain())
    }
}

impl<'data, const C: usize, S, D, T> approx::AbsDiffEq<T> for SignalSlice<'data, C, S, D>
where
    S: Sample + approx::AbsDiffEq,
    D: Domain,
    T: DSP<C, S, D>,
    SignalSlice<'data, C, S, D>: PartialEq<T>,
    <S as approx::AbsDiffEq>::Epsilon: Copy,
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }
    fn abs_diff_eq(&self, other: &T, epsilon: Self::Epsilon) -> bool {
        self.len() == other.len()
            && self
                .iter_cha()
                .zip(other._iter_cha())
                .all(|(cha_1, cha_2)| {
                    cha_1
                        .iter()
                        .zip(cha_2)
                        .all(|(s1, s2)| S::abs_diff_eq(s1, s2, epsilon))
                })
    }
}

#[cfg(feature = "serde")]
impl<'data, const C: usize, S: Sample, D: Domain> serde::Serialize for SignalSlice<'data, C, S, D>
where
    [&'data [S]; C]: serde::Serialize,
    S: serde::Serialize,
    D: serde::Serialize,
{
    fn serialize<SE>(&self, serializer: SE) -> Result<SE::Ok, SE::Error>
    where
        SE: serde::Serializer,
    {
        (self.channels, self.sampling_rate, self._domain).serialize(serializer)
    }
}

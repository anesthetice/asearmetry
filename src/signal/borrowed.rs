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

impl<'data, const C: usize, S: Sample, D: Domain> SignalSlice<'data, C, S, D> {
    pub fn new(channels: [&'data [S]; C], sampling_rate: Option<Hertz>) -> Self {
        Self {
            channels,
            sampling_rate,
            _domain: D::default(),
        }
    }
    pub fn new_empty() -> Self {
        Self {
            channels: std::array::repeat(&[]),
            sampling_rate: None,
            _domain: D::default(),
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

impl<'data, const C: usize, S: Sample> From<[&'data [S]; C]>
    for SignalSlice<'data, C, S, TimeDomain>
{
    fn from(value: [&'data [S]; C]) -> Self {
        Self {
            channels: value,
            sampling_rate: None,
            _domain: TimeDomain::default(),
        }
    }
}

impl<'data, const C: usize, S: Sample> From<([&'data [S]; C], Option<Hertz>)>
    for SignalSlice<'data, C, S, TimeDomain>
{
    fn from(value: ([&'data [S]; C], Option<Hertz>)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: value.1,
            _domain: TimeDomain::default(),
        }
    }
}

impl<'data, const C: usize, S: Sample> From<([&'data [S]; C], Hertz)>
    for SignalSlice<'data, C, S, TimeDomain>
{
    fn from(value: ([&'data [S]; C], Hertz)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1),
            _domain: TimeDomain::default(),
        }
    }
}

impl<'data, S: Sample> From<&'data [S]> for SignalSlice<'data, 1, S, TimeDomain> {
    fn from(value: &'data [S]) -> Self {
        Self {
            channels: [value],
            sampling_rate: None,
            _domain: TimeDomain::default(),
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

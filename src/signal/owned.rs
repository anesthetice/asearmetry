/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use crate::{
    math::Hertz,
    signal::{DSP, Domain, FreqDomain, Sample, Signal, TimeDomain},
};
use std::marker::PhantomData;

impl<const C: usize, S: Sample, D: Domain> Signal<C, S, D> {
    pub fn new(channels: [Vec<S>; C], sampling_rate: Option<Hertz>) -> Self {
        Self {
            channels,
            sampling_rate,
            _domain: PhantomData,
        }
    }
    pub fn new_empty() -> Self {
        Self {
            channels: std::array::repeat(Vec::new()),
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
    pub fn new_zeros(nb_samples: usize) -> Self {
        Self {
            channels: std::array::repeat(vec![S::zero(); nb_samples]),
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
    pub fn with_capacity(per_channel_capacity: usize, sampling_rate: Option<Hertz>) -> Self {
        Self {
            // Do not use `std::array::repeat` as cloning a vector will not preserve capacity
            channels: std::array::from_fn(|_| Vec::with_capacity(per_channel_capacity)),
            sampling_rate,
            _domain: PhantomData,
        }
    }
    pub fn with_sr(mut self, sampling_rate: Hertz) -> Self {
        self.sampling_rate = Some(sampling_rate);
        self
    }
    pub fn set_sr(&mut self, sampling_rate: Hertz) -> &mut Self {
        self.sampling_rate = Some(sampling_rate);
        self
    }
    pub fn with_sr_opt(mut self, sampling_rate: Option<Hertz>) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }
    pub fn set_sr_opt(&mut self, sampling_rate: Option<Hertz>) -> &mut Self {
        self.sampling_rate = sampling_rate;
        self
    }

    pub fn cha_mut(&mut self, c: usize) -> &mut Vec<S> {
        &mut self.channels[c]
    }

    pub(crate) fn cha_mut_uc(&mut self, c: usize) -> &mut Vec<S> {
        unsafe { self.channels.get_unchecked_mut(c) }
    }

    pub fn set<I>(&mut self, index: I, value: [S; C])
    where
        I: std::slice::SliceIndex<[S], Output = S> + Copy,
    {
        self.channels
            .iter_mut()
            .zip(value)
            .for_each(|(cha, val)| cha[index] = val);
    }

    pub fn iter_cha_mut(&mut self) -> impl Iterator<Item = &mut Vec<S>> {
        self.channels.iter_mut()
    }

    pub fn for_each_cha_mut<F: FnMut(&mut Vec<S>)>(&mut self, op: F) {
        self.channels.iter_mut().for_each(op);
    }

    pub fn with_transform<S2, F>(self, op: F) -> Signal<C, S2, D>
    where
        S2: Sample,
        F: FnOnce([Vec<S>; C]) -> [Vec<S2>; C],
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
        F: FnMut(Vec<S>) -> Vec<S2>,
    {
        Signal {
            channels: self.channels.map(op),
            sampling_rate: self.sampling_rate,
            _domain: self._domain,
        }
    }

    pub fn with_domain<D2: Domain>(self, _domain: D2) -> Signal<C, S, D2> {
        Signal {
            channels: self.channels,
            sampling_rate: self.sampling_rate,
            _domain: PhantomData,
        }
    }
}

impl<S: Sample, D: Domain> Signal<1, S, D> {
    pub fn repeat_cha<const C: usize>(self) -> Signal<C, S, D> {
        let [channel] = self.channels;
        Signal::new(std::array::repeat(channel), self.sampling_rate)
    }
}

impl<const C: usize, S: Sample, D: Domain> From<[Vec<S>; C]> for Signal<C, S, D> {
    fn from(value: [Vec<S>; C]) -> Self {
        Self {
            channels: value,
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
}

impl<const C: usize, S: Sample, D: Domain> From<([Vec<S>; C], Option<Hertz>)> for Signal<C, S, D> {
    fn from(value: ([Vec<S>; C], Option<Hertz>)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: value.1,
            _domain: PhantomData,
        }
    }
}

impl<const C: usize, S: Sample, D: Domain> From<([Vec<S>; C], Hertz)> for Signal<C, S, D> {
    fn from(value: ([Vec<S>; C], Hertz)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1),
            _domain: PhantomData,
        }
    }
}

impl<S: Sample, D: Domain> From<Vec<S>> for Signal<1, S, D> {
    fn from(value: Vec<S>) -> Self {
        Self {
            channels: [value],
            sampling_rate: None,
            _domain: PhantomData,
        }
    }
}

/*
// Maybe switch to nightly for specialization if this is actually useful
impl<const C: usize, U: num_traits::AsPrimitive<u32>> From<([Vec<f32>; C], U)> for Signal<C, S, D> {
    fn from(value: ([Vec<f32>; C], U)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1.as_()),
        }
    }
}
*/

impl<const C: usize, S, D, T> PartialEq<T> for Signal<C, S, D>
where
    S: Sample,
    D: Domain,
    T: DSP<C, S, D>,
{
    fn eq(&self, other: &T) -> bool {
        self.view().eq(other)
    }
}

impl<const C: usize, S, D, T> approx::AbsDiffEq<T> for Signal<C, S, D>
where
    S: Sample + approx::AbsDiffEq,
    D: Domain,
    T: DSP<C, S, D>,
    Signal<C, S, D>: PartialEq<T>,
    <S as approx::AbsDiffEq>::Epsilon: Copy,
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }
    fn abs_diff_eq(&self, other: &T, epsilon: Self::Epsilon) -> bool {
        self.view().abs_diff_eq(other, epsilon)
    }
}

#[cfg(feature = "serde")]
impl<const C: usize, S: Sample, D: Domain> serde::Serialize for Signal<C, S, D>
where
    [Vec<S>; C]: serde::Serialize,
    S: serde::Serialize,
    D: serde::Serialize,
{
    fn serialize<SE>(&self, serializer: SE) -> Result<SE::Ok, SE::Error>
    where
        SE: serde::Serializer,
    {
        (&self.channels, self.sampling_rate, self._domain).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, const C: usize, S: Sample, D: Domain> serde::Deserialize<'de> for Signal<C, S, D>
where
    [Vec<S>; C]: serde::Deserialize<'de>,
    S: serde::Deserialize<'de>,
    D: serde::Deserialize<'de>,
{
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        let tuple: ([Vec<S>; C], Option<Hertz>, PhantomData<D>) =
            serde::Deserialize::deserialize(deserializer)?;
        Ok(Self {
            channels: tuple.0,
            sampling_rate: tuple.1,
            _domain: tuple.2,
        })
    }
}

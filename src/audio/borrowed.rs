use crate::{
    audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal},
    math::Hertz,
};
use itertools::Itertools;

impl<'data, const C: usize> AudioBufferSlice<'data, C> {
    pub fn new(channels: [&'data [f32]; C], sampling_rate: Option<Hertz>) -> Self {
        Self {
            channels,
            sampling_rate,
        }
    }
    pub fn new_empty() -> Self {
        Self {
            channels: std::array::repeat(&[]),
            sampling_rate: None,
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
        a: AudioBufferSlice<'a, C1>,
        b: AudioBufferSlice<'b, C2>,
    ) -> Self
    where
        'a: 'data, // 'a outlives 'data
        'b: 'data, // 'b outlives 'data
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(a.sr(), b.sr());
        let channels: [&'data [f32]; C] =
            itertools::chain!(a.channels.into_iter(), b.channels.into_iter(),)
                .collect_array()
                .unwrap();

        Self::new(channels, sampling_rate)
    }
}

impl<'data, const C: usize> From<&'data AudioBuffer<C>> for AudioBufferSlice<'data, C> {
    fn from(value: &'data AudioBuffer<C>) -> Self {
        Self {
            channels: std::array::from_fn(|i| value.channels[i].as_slice()),
            sampling_rate: value.sampling_rate,
        }
    }
}

impl<'data, const C: usize> From<[&'data [f32]; C]> for AudioBufferSlice<'data, C> {
    fn from(value: [&'data [f32]; C]) -> Self {
        Self {
            channels: value,
            sampling_rate: None,
        }
    }
}

impl<'data, const C: usize> From<([&'data [f32]; C], Option<Hertz>)>
    for AudioBufferSlice<'data, C>
{
    fn from(value: ([&'data [f32]; C], Option<Hertz>)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: value.1,
        }
    }
}

impl<'data, const C: usize> From<([&'data [f32]; C], Hertz)> for AudioBufferSlice<'data, C> {
    fn from(value: ([&'data [f32]; C], Hertz)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1),
        }
    }
}

impl<'data> From<&'data [f32]> for AudioBufferSlice<'data, 1> {
    fn from(value: &'data [f32]) -> Self {
        Self {
            channels: [value],
            sampling_rate: None,
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
impl<'data, const C: usize> serde::Serialize for AudioBufferSlice<'data, C>
where
    [&'data [f32]; C]: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.channels, self.sampling_rate).serialize(serializer)
    }
}

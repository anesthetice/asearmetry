use crate::audio::{AudioBuffer, AudioBufferSlice};

impl<'a, const C: usize> AudioBufferSlice<'a, C> {
    pub fn new(channels: [&'a [f32]; C], sampling_rate: Option<u32>) -> Self {
        Self {
            channels,
            sampling_rate,
        }
    }
    pub fn with_sampling_rate(mut self, sampling_rate: Option<u32>) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }
}

impl<'a, const C: usize> From<&'a AudioBuffer<C>> for AudioBufferSlice<'a, C> {
    fn from(value: &'a AudioBuffer<C>) -> Self {
        Self {
            channels: std::array::from_fn(|i| value.channels[i].as_slice()),
            sampling_rate: value.sampling_rate,
        }
    }
}

impl<'a, const C: usize> From<[&'a [f32]; C]> for AudioBufferSlice<'a, C> {
    fn from(value: [&'a [f32]; C]) -> Self {
        Self {
            channels: value,
            sampling_rate: None,
        }
    }
}

impl<'a, const C: usize> From<([&'a [f32]; C], Option<u32>)> for AudioBufferSlice<'a, C> {
    fn from(value: ([&'a [f32]; C], Option<u32>)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: value.1,
        }
    }
}

impl<'a, const C: usize> From<([&'a [f32]; C], u32)> for AudioBufferSlice<'a, C> {
    fn from(value: ([&'a [f32]; C], u32)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1),
        }
    }
}

impl<'a> From<&'a [f32]> for AudioBufferSlice<'a, 1> {
    fn from(value: &'a [f32]) -> Self {
        Self {
            channels: [value],
            sampling_rate: None,
        }
    }
}

/*
// Maybe switch to nightly for specialization if this is actually useful
impl<'a, const C: usize, U: num_traits::AsPrimitive<u32>> From<([&'a [f32]; C], U)>
    for AudioBufferSlice<'a, C>
{
    fn from(value: ([&'a [f32]; C], U)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1.as_()),
        }
    }
}
*/

#[cfg(feature = "serde")]
impl<'a, const C: usize> serde::Serialize for AudioBufferSlice<'a, C>
where
    [&'a [f32]; C]: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.channels, self.sampling_rate).serialize(serializer)
    }
}

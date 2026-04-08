// Struct declarations

/// Represents a multichannel audio signal.
///
/// Stores audio or signal samples in a channel-major layout, with
/// one owned buffer per channel. All buffers are expected to have
/// the same length at all times. Samples are represented by 32-bit
/// floating point numbers.
///
/// The constant generic parameter `C` refers to the number of channels.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct AudioBuffer<const C: usize> {
    pub(crate) channels: [Vec<f32>; C],
    /// Sampling rate in Hertz.
    pub(crate) sampling_rate: Option<u32>,
}

/// A borrowed, channel-major view into sliced audio sample data.
/// This is the non-owning counterpart to [`AudioBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct AudioBufferSlice<'a, const C: usize> {
    pub(crate) channels: [&'a [f32]; C],
    /// Sampling rate in Hertz.
    pub(crate) sampling_rate: Option<u32>,
}

// Start of [`AudioBuffer`] related code
//
impl<const C: usize> AudioBuffer<C> {
    pub fn new(channels: [Vec<f32>; C], sampling_rate: Option<u32>) -> Self {
        Self {
            channels,
            sampling_rate,
        }
    }

    pub fn with_capacity(per_channel_capacity: usize, sampling_rate: Option<u32>) -> Self {
        Self {
            // Do not use `std::array::repeat` as cloning a vector will not preserve capacity
            channels: std::array::from_fn(|_| Vec::with_capacity(per_channel_capacity)),
            sampling_rate,
        }
    }
    pub fn new_zeros(nb_samples: usize) -> Self {
        Self {
            channels: std::array::repeat(vec![0.0; nb_samples]),
            sampling_rate: None,
        }
    }
    pub fn new_empty() -> Self {
        Self {
            channels: std::array::repeat(Vec::new()),
            sampling_rate: None,
        }
    }
    pub fn with_sampling_rate(mut self, sampling_rate: Option<u32>) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }
    pub fn iter_cha_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        self.channels.iter_mut().map(Vec::as_mut_slice)
    }
}

impl<const C: usize> From<[Vec<f32>; C]> for AudioBuffer<C> {
    fn from(value: [Vec<f32>; C]) -> Self {
        Self {
            channels: value,
            sampling_rate: None,
        }
    }
}

impl<const C: usize> From<([Vec<f32>; C], Option<u32>)> for AudioBuffer<C> {
    fn from(value: ([Vec<f32>; C], Option<u32>)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: value.1,
        }
    }
}

impl<const C: usize> From<([Vec<f32>; C], u32)> for AudioBuffer<C> {
    fn from(value: ([Vec<f32>; C], u32)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1),
        }
    }
}

/*
// Maybe switch to nightly for specialization if this is actually useful
impl<const C: usize, U: num_traits::AsPrimitive<u32>> From<([Vec<f32>; C], U)> for AudioBuffer<C> {
    fn from(value: ([Vec<f32>; C], U)) -> Self {
        Self {
            channels: value.0,
            sampling_rate: Some(value.1.as_()),
        }
    }
}
*/

#[cfg(feature = "serde")]
impl<'de, const C: usize> serde::Deserialize<'de> for AudioBuffer<C>
where
    [Vec<f32>; C]: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let tuple: ([Vec<f32>; C], Option<u32>) = serde::Deserialize::deserialize(deserializer)?;
        Ok(AudioBuffer::from(tuple))
    }
}

#[cfg(feature = "serde")]
impl<const C: usize> serde::Serialize for AudioBuffer<C>
where
    [Vec<f32>; C]: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.channels, self.sampling_rate).serialize(serializer)
    }
}

// Start of [`AudioBufferSlice`] related code
//
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

use crate::audio::AudioBuffer;

impl<const C: usize> AudioBuffer<C> {
    pub fn new(channels: [Vec<f32>; C], sampling_rate: Option<u32>) -> Self {
        Self {
            channels,
            sampling_rate,
        }
    }
    pub fn new_empty() -> Self {
        Self {
            channels: std::array::repeat(Vec::new()),
            sampling_rate: None,
        }
    }
    pub fn new_zeros(nb_samples: usize) -> Self {
        Self {
            channels: std::array::repeat(vec![0.0; nb_samples]),
            sampling_rate: None,
        }
    }
    pub fn with_capacity(per_channel_capacity: usize, sampling_rate: Option<u32>) -> Self {
        Self {
            // Do not use `std::array::repeat` as cloning a vector will not preserve capacity
            channels: std::array::from_fn(|_| Vec::with_capacity(per_channel_capacity)),
            sampling_rate,
        }
    }
    pub fn with_sr(mut self, sampling_rate: u32) -> Self {
        self.sampling_rate = Some(sampling_rate);
        self
    }
    pub fn with_sr_opt(mut self, sampling_rate: Option<u32>) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }
    pub fn cha_mut(&mut self, c: usize) -> &mut Vec<f32> {
        &mut self.channels[c]
    }
    pub fn iter_cha_mut(&mut self) -> impl Iterator<Item = &mut Vec<f32>> {
        self.channels.iter_mut()
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

impl From<Vec<f32>> for AudioBuffer<1> {
    fn from(value: Vec<f32>) -> Self {
        Self {
            channels: [value],
            sampling_rate: None,
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

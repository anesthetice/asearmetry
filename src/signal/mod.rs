// Modules
mod traits;

// Exports
pub use traits::{AudioSignalConvolution, AudioSignalCore};

// Imports
use crate::Seconds;
use core::f32::consts::{PI, TAU};
use hound::WavWriter;

/// A collection of per-channel sample buffers.
///
/// Stores audio or signal samples in a channel-major layout, with
/// one owned buffer per channel. All buffers are expected to have
/// the same length at all times. Samples are represented by 32-bit
/// floating point numbers.
///
/// The constant generic parameter `C` refers to the number of channels.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct ChannelBuffers<const C: usize>(pub(crate) [Vec<f32>; C]);

/// A borrowed, channel-major view into sliced sample data.
/// This is the non-owning counterpart to [`ChannelBuffers`].
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct ChannelBuffersSlice<'a, const C: usize>(pub(crate) [&'a [f32]; C]);

/// Represents a multichannel audio signal sampled at a fixed rate.
/// It is composed of a [`ChannelBuffers`] structure paired with a
/// sample rate ([`f32`]), measured in Hertz.
#[derive(Debug, Clone)]
pub struct AudioBuffer<const C: usize> {
    pub(crate) inner: ChannelBuffers<C>,
    pub(crate) sample_rate: f32,
}

// [`ChannelBuffers`] specific implementations
impl<const C: usize> ChannelBuffers<C> {
    pub fn new_empty() -> Self {
        Self(std::array::repeat(Vec::new()))
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(std::array::repeat(Vec::with_capacity(capacity)))
    }
    fn cha_mut(&mut self, c: usize) -> &mut [f32] {
        self.0[c].as_mut_slice()
    }
    pub fn into_audio_buf(self, sample_rate: f32) -> AudioBuffer<C> {
        AudioBuffer {
            inner: self,
            sample_rate,
        }
    }
    /// Applied collectively across all channels
    pub fn normalize(&mut self) {
        let max_array: [f32; C] = std::array::from_fn(|i| {
            self.cha(i)
                .iter()
                .copied()
                .map(f32::abs)
                .reduce(f32::max)
                .unwrap()
        });
        let max = max_array.into_iter().reduce(f32::max).unwrap();
        for c in 0..C {
            self.cha_mut(c)
                .iter_mut()
                .for_each(|x| *x = (*x / max).clamp(-1.0, 1.0));
        }
    }
}
impl<const C: usize> From<[Vec<f32>; C]> for ChannelBuffers<C> {
    fn from(value: [Vec<f32>; C]) -> Self {
        Self(value)
    }
}
impl<const C: usize> From<AudioBuffer<C>> for ChannelBuffers<C> {
    fn from(value: AudioBuffer<C>) -> Self {
        value.inner
    }
}

// [`ChannelBuffersSlice`] specific implementations
impl<'a, const C: usize> From<&'a ChannelBuffers<C>> for ChannelBuffersSlice<'a, C> {
    fn from(value: &'a ChannelBuffers<C>) -> Self {
        Self(std::array::from_fn(|i| value.0[i].as_slice()))
    }
}
impl<'a, const C: usize> From<&'a AudioBuffer<C>> for ChannelBuffersSlice<'a, C> {
    fn from(value: &'a AudioBuffer<C>) -> Self {
        (&value.inner).into()
    }
}
impl<'a, const C: usize> From<[&'a [f32]; C]> for ChannelBuffersSlice<'a, C> {
    fn from(value: [&'a [f32]; C]) -> Self {
        Self(value)
    }
}

// [`AudioBuffer`] specific implementations
pub type MonoAudioBuf = AudioBuffer<1>;
pub type StereoAudioBuf = AudioBuffer<2>;

impl MonoAudioBuf {
    pub fn new_mono(samples: Vec<f32>, sample_rate: f32) -> AudioBuffer<1> {
        AudioBuffer {
            inner: ChannelBuffers([samples]),
            sample_rate,
        }
    }
}
impl StereoAudioBuf {
    pub fn new_stereo(
        left_samples: Vec<f32>,
        right_samples: Vec<f32>,
        sample_rate: f32,
    ) -> AudioBuffer<2> {
        assert_eq!(left_samples.len(), right_samples.len());
        AudioBuffer {
            inner: ChannelBuffers([left_samples, right_samples]),
            sample_rate,
        }
    }
}
impl<const C: usize> AudioBuffer<C> {
    pub fn new<T: Into<ChannelBuffers<C>>>(channel_bufs: T, sample_rate: f32) -> Self {
        Self {
            inner: channel_bufs.into(),
            sample_rate,
        }
    }
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    pub fn duration(&self) -> Seconds {
        self.len() as f32 / self.sample_rate
    }
    pub fn low_pass(&self, cutoff_freq: f32) -> Self {
        let a = TAU * cutoff_freq / (TAU * cutoff_freq + self.sample_rate);

        let mut out = ChannelBuffers::<C>::with_capacity(self.len());

        for c in 0..C {
            let y_vec: &mut Vec<f32> = &mut out.0[c];
            let mut y_prev: f32 = 0.0;
            for x in self.cha(c) {
                let y = (1.0 - a) * y_prev + a * x;
                y_prev = y;
                y_vec.push(y);
            }
        }

        Self {
            inner: out,
            sample_rate: self.sample_rate,
        }
    }
    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        writer: &mut WavWriter<W>,
    ) -> anyhow::Result<()> {
        debug_assert_eq!(writer.spec().sample_rate, self.sample_rate as u32);

        for i in 0..self.len() {
            for cha in 0..C {
                let _ = writer.write_sample(self.cha(cha)[i]).inspect_err(|err| {
                    eprintln!("An error occured while writing wav samples, {err}")
                });
            }
        }
        writer.flush()?;
        Ok(())
    }
}

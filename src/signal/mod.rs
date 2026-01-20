// Modules
mod borrow;
mod convolution;

// Exports
pub use borrow::AudioFrameRef;
use itertools::Itertools;

// Imports
use crate::Seconds;
use hound::WavWriter;

pub trait AudioBufferCore<'a, const CHANNELS: usize> {
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn cha(&'a self, c: usize) -> &'a [f32];
    fn get<I>(&'a self, index: I) -> AudioFrameRef<'a, CHANNELS>
    where
        I: std::slice::SliceIndex<[f32], Output = &'a [f32]> + Copy;
}

/// A time-synchronized frame of audio samples.
#[derive(Debug)]
#[repr(transparent)]
pub struct AudioFrame<const CHANNELS: usize>(pub(crate) [Vec<f32>; CHANNELS]);

impl<'a, const CHANNELS: usize> AudioBufferCore<'a, CHANNELS> for AudioFrame<CHANNELS> {
    fn is_empty(&self) -> bool {
        self.0.first().map(Vec::is_empty).unwrap_or(true)
    }
    fn len(&self) -> usize {
        self.0.first().map(Vec::len).unwrap_or(0)
    }
    fn cha(&'a self, c: usize) -> &'a [f32] {
        self.0[c].as_slice()
    }
    fn get<I>(&'a self, index: I) -> AudioFrameRef<'a, CHANNELS>
    where
        I: std::slice::SliceIndex<[f32], Output = &'a [f32]> + Copy,
    {
        std::array::from_fn(|i| self.cha(i)[index]).into()
    }
}

impl<const CHANNELS: usize> From<[Vec<f32>; CHANNELS]> for AudioFrame<CHANNELS> {
    fn from(value: [Vec<f32>; CHANNELS]) -> Self {
        assert!(value.iter().map(Vec::len).all_equal());
        Self(value)
    }
}

impl<const CHANNELS: usize> AudioFrame<CHANNELS> {
    pub fn into_signal(self, sampling_rate: f32) -> Signal<CHANNELS> {
        Signal {
            samples: self,
            sampling_rate,
        }
    }

    pub(crate) fn from_unchecked(value: [Vec<f32>; CHANNELS]) -> Self {
        Self(value)
    }
}

/// Audio signal, potentially multichannel, represented with 32-bit floating point numbers.
pub struct Signal<const CHANNELS: usize> {
    samples: AudioFrame<CHANNELS>,
    pub sampling_rate: f32,
}

pub type MonoSig = Signal<1>;
pub type StereoSig = Signal<2>;

impl MonoSig {
    pub fn new_mono(samples: Vec<f32>, sampling_rate: f32) -> Signal<1> {
        Signal {
            samples: AudioFrame([samples]),
            sampling_rate,
        }
    }
}

impl StereoSig {
    pub fn new_stereo(
        left_samples: Vec<f32>,
        right_samples: Vec<f32>,
        sampling_rate: f32,
    ) -> Signal<2> {
        assert_eq!(left_samples.len(), right_samples.len());
        Signal {
            samples: AudioFrame([left_samples, right_samples]),
            sampling_rate,
        }
    }
}

impl<'a, const CHANNELS: usize> AudioBufferCore<'a, CHANNELS> for Signal<CHANNELS> {
    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    fn len(&self) -> usize {
        self.samples.len()
    }
    fn cha(&'a self, c: usize) -> &'a [f32] {
        self.samples.cha(c)
    }
    fn get<I>(&'a self, index: I) -> AudioFrameRef<'a, CHANNELS>
    where
        I: std::slice::SliceIndex<[f32], Output = &'a [f32]> + Copy,
    {
        self.samples.get(index)
    }
}

impl<const CHANNELS: usize> Signal<CHANNELS> {
    pub fn new<T: Into<AudioFrame<CHANNELS>>>(samples: T, sampling_rate: f32) -> Self {
        Self {
            samples: samples.into(),
            sampling_rate,
        }
    }

    pub fn duration(&self) -> Seconds {
        self.len() as f32 / self.sampling_rate
    }

    pub fn to_borrowed(&self) -> AudioFrameRef<'_, CHANNELS> {
        self.into()
    }

    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        writer: &mut WavWriter<W>,
    ) -> anyhow::Result<()> {
        debug_assert_eq!(writer.spec().sample_rate, self.sampling_rate as u32);

        for i in 0..self.len() {
            for cha in 0..CHANNELS {
                let _ = writer.write_sample(self.cha(cha)[i]).inspect_err(|err| {
                    eprintln!("An error occured while writing wav samples, {err}")
                });
            }
        }
        writer.flush()?;
        Ok(())
    }
}

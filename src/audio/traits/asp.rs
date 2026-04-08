// Imports
use crate::{
    Seconds,
    audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal},
};
use std::f32::consts::TAU;

pub trait AudioSignal<const C: usize>: DiscreteSignal<C> {
    fn duration(&self) -> Option<Seconds> {
        self.sampling_rate_f32().map(|sr| self.len() as f32 / sr)
    }

    fn delay(&self, by: Seconds) -> AudioBuffer<C> {
        let sampling_rate = self
            .sampling_rate_f32()
            .expect("Sampling rate is not defined");
        self.pad_left((by * sampling_rate).ceil() as usize)
    }

    fn low_pass(&self, cutoff_freq: f32) -> AudioBuffer<C> {
        let sampling_rate = self
            .sampling_rate_f32()
            .expect("Sampling rate is not defined");

        let a = TAU * cutoff_freq / (TAU * cutoff_freq + sampling_rate);

        let mut out = self.into_owned();

        for cha in out.iter_cha_mut() {
            let mut y_prev: f32 = 0.0;
            cha.iter_mut().for_each(|x| {
                let y = (1.0 - a) * y_prev + a * *x;
                y_prev = y;
                *x = y
            });
        }

        out
    }

    fn write_to<W: std::io::Write + std::io::Seek>(&self, writer: &mut W) -> anyhow::Result<()> {
        debug_assert!(
            self.get_abs_max() <= 1.0,
            "Some samples are out of bounds (∉ [0, 1])"
        );

        let sampling_rate = self
            .sampling_rate()
            .ok_or_else(|| anyhow::anyhow!("The sampling rate must be defined to write audio"))?;

        let spec = hound::WavSpec {
            channels: C as u16,
            sample_rate: sampling_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let samples_i16 = self.interleaved_samples_i16();
        let mut writer = hound::WavWriter::new(writer, spec)?;
        let mut efficient_writer = writer.get_i16_writer(samples_i16.len() as u32);

        samples_i16.into_iter().for_each(|sample| {
            unsafe { efficient_writer.write_sample_unchecked(sample) };
        });
        efficient_writer.flush()?;

        Ok(())
    }

    fn write_to_file<Q: AsRef<std::path::Path>>(&self, filepath: Q) -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(filepath)?;

        self.write_to(&mut file)
    }
}

impl<const C: usize> AudioSignal<C> for AudioBuffer<C> {}
impl<const C: usize> AudioSignal<C> for AudioBufferSlice<'_, C> {}
impl<const C: usize, T> AudioSignal<C> for &T where T: AudioSignal<C> {}

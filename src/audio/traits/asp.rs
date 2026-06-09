/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::{
    audio::{AudioBuffer, DiscreteSignal, MonoAudioBuf},
    math::{Seconds, sinc},
};
use itertools::Itertools;
use std::f32::consts::PI;

pub trait AudioSignal<const C: usize>: DiscreteSignal<C> {
    fn duration(&self) -> Option<Seconds> {
        self.sampling_rate().map(|sr| self.len() as f64 / sr)
    }

    fn delay(&self, by: Seconds) -> AudioBuffer<C> {
        let sampling_rate = self.sr_or_panic();
        self.pad_left((by * sampling_rate).ceil() as usize)
    }

    /// The parameter M is such that the filter's number of taps (i.e. its length) is equal to 2M+1.
    /// This is to ensure that the filter's length is odd.
    #[allow(non_snake_case)]
    fn apply_low_pass_filter(&self, cutoff_freq: f32, M: usize) -> AudioBuffer<C>
    where
        Self: super::DefinedLtiConvolution<C, 1, C>,
    {
        let num_taps = (M * 2) + 1;
        let f_norm = cutoff_freq / self.sr_f32_or_panic();

        if f_norm > 0.5 {
            eprintln!(
                "Warning, the provided cutoff frequency is larger than the nyquist frequency"
            );
        }

        let filter: MonoAudioBuf = (0..num_taps)
            .map(|n| {
                // We must offset the center of the filter by M to the right as we are in the discrete case, h[n<0]=0
                let ideal = 2.0 * f_norm * sinc(2.0 * f_norm * (n as f32 - M as f32));

                // If we returned the ideal here above we would be actually using: "h[n] = h_ideal[n] ⋅ w_rect[n]"
                // our function would be discontinuous which is not good especially in our discrete case,
                // see "Gibbs Phenomenon". Therefore, we use the "Hann" window function to smooth things out.
                let hann = f32::sin(PI * n as f32 / (2 * M) as f32).powi(2);

                ideal * hann
            })
            .collect_vec()
            .into();

        self.convolve_then_crop(filter)
    }

    /// The parameter M is such that the filter's number of taps (i.e. its length) is equal to 2M+1.
    /// This is to ensure that the filter's length is odd.
    #[allow(non_snake_case)]
    fn apply_median_filter(&self, M: usize) -> AudioBuffer<C> {
        assert!(M > 0);
        self.apply_centered_window(M, |x| {
            let mut vec = x.to_vec();
            vec.sort_unstable_by(f32::total_cmp);
            if vec.len() % 2 == 1 {
                vec[vec.len() / 2]
            } else {
                0.5 * (vec[vec.len() / 2 - 1] + vec[vec.len() / 2])
            }
        })
    }

    /// The parameter M is such that the filter's number of taps (i.e. its length) is equal to 2M+1.
    /// This is to ensure that the filter's length is odd.
    #[allow(non_snake_case)]
    fn apply_gaussian_filter(&self, M: usize, a: f32) -> AudioBuffer<C>
    where
        Self: super::DefinedLtiConvolution<C, 1, C>,
    {
        assert!(M > 0);
        let M = M as i64;
        let num_taps = (M * 2) + 1;
        let filter: MonoAudioBuf = (0..num_taps)
            .map(|n| {
                // We must offset the center of the filter by M to the right as we are in the discrete case, h[n<0]=0
                let x = (n - M) as f32;
                let ideal = f32::sqrt(a / PI) * f32::exp(-a * x.powi(2));

                // Once again we use the Hann window function to avoid the "Gibbs Phenomenon".
                let hann = f32::sin(PI * n as f32 / (2 * M) as f32).powi(2);

                ideal * hann
            })
            .collect_vec()
            .into();

        self.convolve_then_crop(filter)
    }

    fn write_to<W: std::io::Write + std::io::Seek>(&self, writer: &mut W) -> anyhow::Result<()> {
        debug_assert!(
            self.get_abs_max() <= 1.0,
            "Some samples are out of bounds (∉ [0, 1]), got absmax of {}",
            self.get_abs_max()
        );

        let sampling_rate = self
            .sampling_rate()
            .ok_or_else(|| anyhow::anyhow!("The sampling rate must be defined to write audio"))?;

        let spec = hound::WavSpec {
            channels: C as u16,
            sample_rate: sampling_rate.round() as u32,
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

        self.write_to(&mut file)?;
        file.sync_all()?;
        Ok(())
    }
}

impl<const C: usize, T> AudioSignal<C> for T where T: DiscreteSignal<C> {}

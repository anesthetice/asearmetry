/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::{
    math::{Hertz, Seconds, hann_window_iter, sinc},
    signal::{
        DSP, DefinedLtiConvolution, TimeDomain,
        audio::{AudioBuffer, MonoAudioBuf},
    },
    utils::BWriter,
};
use anyhow::anyhow;
use itertools::Itertools;
use num_complex::Complex32;
use std::{
    cmp::Ordering,
    f32::consts::PI as PI_F32,
    f64::consts::{PI as PI_F64, TAU as TAU_F64},
    ops::MulAssign,
};
use tap::{Pipe, Tap};

pub trait ASP<const C: usize>: DSP<C, f32, TimeDomain> {
    fn duration(&self) -> Option<Seconds> {
        self.sampling_rate().map(|sr| self.len() as f64 / sr)
    }

    fn seconds_to_index(&self, s: Seconds) -> usize {
        (self.sr_or_panic() * s).round() as usize
    }

    fn delay(&self, by: Seconds) -> AudioBuffer<C> {
        let sampling_rate = self.sr_or_panic();
        self.pad_left((by * sampling_rate).ceil() as usize)
    }

    fn fade_in_linear(self, n_samples: usize) -> AudioBuffer<C> {
        assert!(n_samples > 1);
        let mut out = self.into_owned();

        let slope = 1.0_f32 / (n_samples - 1) as f32;
        out.iter_cha_mut().for_each(|cha_mut| {
            cha_mut
                .iter_mut()
                .take(n_samples)
                .enumerate()
                .for_each(|(idx, s)| s.mul_assign(idx as f32 * slope))
        });

        out
    }

    fn fade_out_linear(self, n_samples: usize) -> AudioBuffer<C> {
        assert!(n_samples > 1);
        let mut out = self.into_owned();

        let slope = 1.0_f32 / (n_samples - 1) as f32;
        out.iter_cha_mut().for_each(|cha_mut| {
            cha_mut
                .iter_mut()
                .rev()
                .take(n_samples)
                .enumerate()
                .for_each(|(idx, s)| s.mul_assign(idx as f32 * slope))
        });

        out
    }

    fn scale_time(self, factor: f64) -> AudioBuffer<C> {
        assert!(factor > 0.0 && factor.is_finite());
        let original_sr = self.sr_or_panic();
        self.into_owned()
            .with_sr(factor * original_sr)
            .resample(original_sr, 21)
    }

    #[allow(non_snake_case)]
    fn shift_pitch(self, by: Hertz) -> AudioBuffer<C> {
        let N_t = self.len();
        let N_f = N_t / 2 + 1;
        let f_s = self.sr_or_panic();

        let Δf = f_s / N_t as f64;
        let by_n_samples = (by / Δf).round() as isize;
        log::info!("{by_n_samples}");

        /* old and incorrect
        self.dft()
            .to_halved()
            .pipe(|mut fsig| {
                let value_at_zero = fsig.index(0);
                fsig = fsig.shift([by_n_samples; C], [Complex32::ZERO; C]);
                fsig.set(0, value_at_zero);
                fsig
            })
            .idft_from_halved(n_samples)
        */

        match by_n_samples.cmp(&0) {
            // Pitch up, we have to "stretch" the frequencies
            Ordering::Greater => {
                todo!()
            }
            // We do nothing except return the input, and throw a warning
            Ordering::Equal => {
                log::warn!(
                    "The `shift_pitch` method changed nothing, the `by` parameter might need a higher absolute value"
                );
                self.into_owned()
            }
            // Pitch down, we have to squeeze the frequencies
            Ordering::Less => {
                let f1 = self.sr_or_panic();
                let f2 = f1 * (N_f - by_n_samples.unsigned_abs()) as f64 / N_f as f64;

                let mut dc = [Complex32::ZERO; C];
                let mut nyquist = [Complex32::ZERO; C];

                self.dft()
                    .into_halved()
                    .tap(|fsig| {
                        dc = fsig.index(0);
                        nyquist = fsig.index(N_f - 1);
                    })
                    .resample(f2, 21)
                    .with_sr(f1)
                    .pad_right_to_len(N_f)
                    .tap_mut(|fsig| {
                        fsig.set(0, dc);
                        fsig.set(N_f - 1, nyquist);
                    })
                    .idft_halved(N_t)
            }
        }
    }

    /// The parameter M is such that the filter's number of taps (i.e. its length) is equal to 2M+1.
    /// This is to ensure that the filter's length is odd.
    #[allow(non_snake_case)]
    fn apply_lowpass_filter(&self, cutoff_freq: Hertz, M: usize) -> AudioBuffer<C>
    where
        Self: DefinedLtiConvolution<C, 1, C, f32, TimeDomain>,
    {
        assert!(M > 0);
        let num_taps = (M * 2) + 1;
        let M = M as i64;
        let f_norm = cutoff_freq / self.sr_or_panic();
        if f_norm > 0.5 {
            log::warn!(
                "The provided cutoff frequency of {cutoff_freq} Hz is larger than the nyquist frequency"
            );
        }

        // We must offset the center of the filter by M to the right as we are in the discrete case, h[n<0]=0
        let filter: MonoAudioBuf = (-M..=M)
            .zip(hann_window_iter::<f64>(num_taps))
            .map(|(x, hann)| {
                // I could make this slightly more efficient (some terms cancel),
                // but I am too lazy, it would also required handling the x=0 case.
                let ideal = 2.0 * f_norm * sinc(2.0 * f_norm * x as f64);

                // If we returned the ideal here above we would be actually using: "h[n] = h_ideal[n] ⋅ w_rect[n]"
                // our function would be discontinuous which is not good especially in our discrete case,
                // see "Gibbs Phenomenon". Therefore, we use the Hann window to smooth things out.
                (ideal * hann) as f32
            })
            .collect_vec()
            .into();

        self.convolve_then_crop(filter)
    }

    /// The parameter M is such that the filter's number of taps (i.e. its length) is equal to 2M+1.
    /// This is to ensure that the filter's length is odd.
    #[allow(non_snake_case)]
    fn apply_highpass_filter(&self, cutoff_freq: Hertz, M: usize) -> AudioBuffer<C>
    where
        Self: DefinedLtiConvolution<C, 1, C, f32, TimeDomain>,
    {
        assert!(M > 0);
        let num_taps = (M * 2) + 1;
        let M = M as i64;
        let f_norm = cutoff_freq / self.sr_or_panic();
        if f_norm > 0.5 {
            log::warn!(
                "The provided cutoff frequency of {cutoff_freq} Hz is larger than the nyquist frequency"
            );
        }

        // We must offset the center of the filter by M to the right as we are in the discrete case, h[n<0]=0
        let filter: MonoAudioBuf = (-M..=M)
            .zip(hann_window_iter::<f64>(num_taps))
            .map(|(n, hann)| {
                // I could make this slightly more efficient (some terms cancel),
                // but I am too lazy, it would also required handling the x=0 case.
                let δ_n = if n != 0 { 0.0 } else { 1.0 };
                let ideal = δ_n - 2.0 * f_norm * sinc(2.0 * f_norm * n as f64);

                // If we returned the ideal here above we would be actually using: "h[n] = h_ideal[n] ⋅ w_rect[n]"
                // our function would be discontinuous which is not good especially in our discrete case,
                // see "Gibbs Phenomenon". Therefore, we use the Hann window to smooth things out.
                (ideal * hann) as f32
            })
            .collect_vec()
            .into();

        self.convolve_then_crop(filter)
    }

    /// The parameter M is such that the filter's number of taps (i.e. its length) is equal to 2M+1.
    /// This is to ensure that the filter's length is odd.
    #[allow(non_snake_case)]
    fn apply_bandpass_filter(&self, cutoff_freqs: (Hertz, Hertz), M: usize) -> AudioBuffer<C>
    where
        Self: DefinedLtiConvolution<C, 1, C, f32, TimeDomain>,
    {
        assert!(M > 0);
        assert!(cutoff_freqs.0 >= 0.0 && cutoff_freqs.0 <= cutoff_freqs.1);

        let num_taps = (M * 2) + 1;
        let M = M as i64;
        let f_s = self.sr_or_panic();

        let (f_low, f_high) = cutoff_freqs;
        let d_f = f_high - f_low;
        let f_m = 0.5 * (f_high + f_low);

        // We must offset the center of the filter by M to the right as we are in the discrete case, h[n<0]=0
        let filter: MonoAudioBuf = (-M..=M)
            .zip(hann_window_iter::<f64>(num_taps))
            .map(|(n, hann)| {
                let ideal = 2.0
                    * d_f
                    * sinc(d_f * n as f64 / f_s)
                    * f64::cos(TAU_F64 * f_m * n as f64 / f_s);

                // If we returned the ideal here above we would be actually using: "h[n] = h_ideal[n] ⋅ w_rect[n]"
                // our function would be discontinuous which is not good especially in our discrete case,
                // see "Gibbs Phenomenon". Therefore, we use the Hann window to smooth things out.
                (ideal * hann) as f32
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
        self.apply_into_centered_window(M, |x| {
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
    fn apply_gaussian_filter(&self, M: usize, a: f64) -> AudioBuffer<C>
    where
        Self: DefinedLtiConvolution<C, 1, C, f32, TimeDomain>,
    {
        assert!(M > 0);
        let num_taps = (M * 2) + 1;
        let M = M as i64;
        // We must offset the center of the filter by M to the right as we are in the discrete case, h[n<0]=0.
        // Also using a Hann window just to be safe, so that the filter has a smooth cutoff near the boundaries.
        let filter: MonoAudioBuf = (-M..=M)
            .zip(hann_window_iter::<f64>(num_taps))
            .map(|(x, hann)| {
                let ideal = f64::sqrt(a / PI_F64) * f64::exp(-a * x.pow(2) as f64);
                (ideal * hann) as f32
            })
            .collect_vec()
            .into();

        self.convolve_then_crop(filter)
    }

    // The resulting signal will have a length of `first.len() + second.len() - n_overlap`.
    fn crossfade<T1, T2>(first: T1, second: T2, n_overlap: usize) -> AudioBuffer<C>
    where
        T1: ASP<C>,
        T2: ASP<C>,
    {
        #[allow(non_snake_case)]
        let N = n_overlap;

        let (start, mid_1) = first._view().last_n_split(N);
        let (mid_2, end) = second._view().first_n_split(N);

        let transition = Self::merge(
            mid_1.apply_enumerate(&mut |(n, x_n)| {
                (PI_F32 * n as f32 / (2.0 * N as f32)).cos().powi(2) * x_n
            }),
            mid_2.apply_enumerate(&mut |(n, x_n)| {
                (PI_F32 * n as f32 / (2.0 * N as f32)).sin().powi(2) * x_n
            }),
        );

        Self::concatenate([start, transition.view(), end])
    }

    fn crossfade_concatenate<T, I>(input: I, n_overlap: usize) -> AudioBuffer<C>
    where
        T: ASP<C>,
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter();
        let acc = input.next().expect("Input is empty").into_owned();
        input.fold(acc, |acc, other| Self::crossfade(acc, other, n_overlap))
    }

    #[rustfmt::skip]
    fn encode_to_wav(&self) -> anyhow::Result<Vec<u8>> {
        let sampling_rate = self.sr()
            .ok_or_else(|| anyhow!("The sampling rate of the signal must be defined"))?
            .pipe(|sr_f64| {
                let sr_u32 = sr_f64 as u32;
                if approx::ulps_eq!(sr_f64, sr_u32 as f64) {
                    Ok(sr_u32)
                } else {
                    Err(anyhow!("The sampling rate of the signal ({}) must be properly convertible to an integer", sr_f64))
                }
            })?;

        let interleaved_samples = self.interleave_samples();

        let header_size = 16 + 26 + 16 + 8;
        let body_size = self.len() * C * 4;

        let mut bw = BWriter::new(header_size + body_size);

        // Refer to:
        // https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
        // https://learn.microsoft.com/en-us/windows/win32/medfound/tutorial--decoding-audio#write-the-wave-file-header
        // https://www.recordingblogs.com/wiki/fact-chunk-of-a-wave-file

        // Master `RIFF` chunk
        bw.write_slice(b"RIFF");                             // `ckID`
        bw.write_u32((header_size + body_size - 8) as u32);  // `cksize`
        bw.write_slice(b"WAVE");                             // `WAVEID`

        // The `fmt` chunk (counts as one of the `WAVE` chunks of the master `RIFF` chunk)
        bw.write_slice(b"fmt ");                             // `ckID`
        bw.write_u32(18);                                    // `cksize`
        bw.write_u16(0x0003);                                // `wFormatTag`, 3 => WAVE_FORMAT_IEEE_FLOAT
        bw.write_u16(C as u16);                              // `nChannels`
        bw.write_u32(sampling_rate);                         // `nSamplesPerSec`
        bw.write_u32(sampling_rate * 4 * C as u32);          // `nAvgBytesPerSec`
        bw.write_u16((C * 4) as u16);                        // `nBlockAlign`
        bw.write_u16(32);                                    // `wBitsPerSample`
        bw.write_u16(0);                                     // `cbSize`

        // The `fact` chunk
        bw.write_slice(b"fact");                             // `ckID`
        bw.write_u32(4);                                     // `cksize`
        bw.write_u32((self.len() * C) as u32);               // `dwSampleLength`

        // The `data` chunk
        bw.write_slice(b"data");                             // `ckID`
        bw.write_u32(body_size as u32);                      // `cksize`
        #[cfg(target_endian = "little")]
        bw.write_slice(bytemuck::cast_slice(&interleaved_samples));
        #[cfg(not(target_endian = "little"))]
        bw.write_slice(interleaved_samples.into_iter().flat_map(f32::to_le_bytes).collect_vec());
        // no need to pad by a single 0x00 byte, as f32 = 4 bytes, and 4 times anything is even

        Ok(bw.into_vec())
    }

    fn write_to_wav_file<Q: AsRef<std::path::Path>>(&self, filepath: Q) -> anyhow::Result<()> {
        crate::utils::write_to_file(filepath, self.encode_to_wav()?)
    }

    #[cfg(feature = "opus")]
    fn encode_to_opus(&self) -> anyhow::Result<Vec<u8>> {
        opus_impl::encode_to_opus_impl(self.clarify_ref(), None, None, None)
    }

    #[cfg(feature = "opus")]
    fn encode_to_opus_with(
        &self,
        bitrate: Option<u32>,
        complexity: Option<u8>,
        bandwidth: Option<opusic_c::Bandwidth>,
    ) -> anyhow::Result<Vec<u8>> {
        opus_impl::encode_to_opus_impl(self.clarify_ref(), bitrate, complexity, bandwidth)
    }

    #[cfg(feature = "opus")]
    fn write_to_opus_file<Q: AsRef<std::path::Path>>(&self, filepath: Q) -> anyhow::Result<()> {
        crate::utils::write_to_file(filepath, self.encode_to_opus()?)
    }

    #[cfg(feature = "opus")]
    fn write_to_opus_file_with<Q: AsRef<std::path::Path>>(
        &self,
        filepath: Q,
        bitrate: Option<u32>,
        complexity: Option<u8>,
        bandwidth: Option<opusic_c::Bandwidth>,
    ) -> anyhow::Result<()> {
        crate::utils::write_to_file(
            filepath,
            self.encode_to_opus_with(bitrate, complexity, bandwidth)?,
        )
    }
}

impl<const C: usize, T> ASP<C> for T where T: DSP<C, f32, TimeDomain> {}

#[cfg(feature = "opus")]
mod opus_impl {
    use crate::{
        signal::{DSP, SignalSlice, TimeDomain},
        utils::BWriter,
    };
    use anyhow::{Context, anyhow, bail};
    use itertools::Itertools;
    use tap::Pipe;

    pub fn encode_to_opus_impl<const C: usize>(
        input: SignalSlice<C, f32, TimeDomain>,
        bitrate: Option<u32>,
        complexity: Option<u8>,
        bandwidth: Option<opusic_c::Bandwidth>,
    ) -> anyhow::Result<Vec<u8>> {
        let sampling_rate = input.sr()
            .ok_or_else(|| anyhow!("The sampling rate of the signal must be defined"))?
            .pipe(|sr_f64| {
                let sr_u32 = sr_f64 as u32;
                if approx::ulps_eq!(sr_f64, sr_u32 as f64) {
                    Ok(sr_u32)
                } else {
                    Err(anyhow!("The sampling rate of the signal ({}) must be properly convertible to an integer", sr_f64))
                }
            })?;

        // From the `libopus` docs: https://opus-codec.org/docs/opus_api-1.5/group__opus__encoder.html
        // > To encode a frame, opus_encode() or opus_encode_float() must be called
        // > with exactly one frame (2.5, 5, 10, 20, 40 or 60 ms) of audio data
        let frame_length = (sampling_rate / 25) as usize; // currently using 40 ms
        let frame_total_byte_size = frame_length * C * 4;

        let mut encoder = opusic_c::Encoder::new(
            match C {
                1 => opusic_c::Channels::Mono,
                2 => opusic_c::Channels::Stereo,
                _ => anyhow::bail!("Only mono- and stereo-channel signals are supported"),
            },
            match sampling_rate {
                8000 => opusic_c::SampleRate::Hz8000,
                12_000 => opusic_c::SampleRate::Hz12000,
                16_000 => opusic_c::SampleRate::Hz16000,
                24_000 => opusic_c::SampleRate::Hz24000,
                48_000 => opusic_c::SampleRate::Hz48000,
                _ => bail!("A sampling rate of {sampling_rate} Hz is not allowed, it must be one of the following: 8000, 12'000, 16'000, 24'000, or 48'000 Hz"),
            },
            opusic_c::Application::Audio,
        ).map_err(|ec| anyhow!("Failed to create the OPUS encoder: {ec:?}"))?;

        encoder
            .set_bitrate(opusic_c::Bitrate::Value(bitrate.unwrap_or(192_000)))
            .map_err(|ec| anyhow!("Failed to set the bitrate of the OPUS encoder: {ec:?}"))?;

        encoder
            .set_bandwidth(bandwidth.unwrap_or(opusic_c::Bandwidth::Full))
            .map_err(|ec| anyhow!("Failed to set the bandwidth of the OPUS encoder: {ec:?}"))?;

        if let Some(value) = complexity {
            encoder.set_complexity(value).map_err(|ec| {
                anyhow!("Failed to set the complexity of the OPUS encoder: {ec:?}")
            })?;
        }

        let mut encoder_buf = Vec::<u8>::with_capacity(4 * frame_total_byte_size);

        let mut opus_writer = OggOpusWriter::new(
            input.len() * C * 4 / 8,
            sampling_rate.into(),
            frame_length as u64,
        );

        let pre_skip: u16 = encoder
            .get_look_ahead()
            .map_err(|ec| anyhow!("Failed to get the lookahead of the OPUS encoder: {ec:?}"))?
            .try_into()
            .context("The lookahead of the OPUS encoder could not be converted from u32 to u16")?;

        opus_writer.write_header_page(C as u8, pre_skip);
        opus_writer.write_tags_page();

        let (frames_iter, leftover) = input.as_blocks_strict(frame_length);

        let leftover = leftover.map(|a| a.pad_right_to_len(frame_length));

        let frames = frames_iter
            .chain(leftover.as_ref().map(|s| s.view()))
            .collect_vec();

        frames[..frames.len() - 1].iter().for_each(|frame| {
            encoder
                .encode_float_to_vec(&frame.interleave_samples(), &mut encoder_buf)
                .unwrap();
            opus_writer.write_audio_page(&encoder_buf);
            encoder_buf.clear();
        });

        frames[frames.len() - 1].pipe(|frame| {
            encoder
                .encode_float_to_vec(&frame.interleave_samples(), &mut encoder_buf)
                .unwrap();
            let mut raw_length = input.len() % frame_length;
            if raw_length == 0 {
                raw_length = 255
            };
            opus_writer.write_final_audio_page(raw_length, &encoder_buf);
        });

        Ok(opus_writer.bw.into_vec())
    }

    struct OggOpusWriter {
        bw: BWriter,

        sampling_rate: u64,
        frame_length: u64,

        page_sidx: usize,
        page_n: u32,
        granule_pos: u64,

        crc_hasher: crc_any::CRCu32,
    }

    impl OggOpusWriter {
        const CHECKSUM_OFFSET: usize = 22;

        /* Not sure if this is worth it to be fair, will leave as a comment for now
        thread_local! {
            static CRC_HASHER: RefCell<crc_any::CRCu32> =
                RefCell::new(crc_any::CRCu32::create_crc(0x04c11db7, 32, 0x0000, 0x0000, false));
        }
        */

        fn new(capacity: usize, sampling_rate: u64, frame_length: u64) -> Self {
            Self {
                bw: BWriter::new(capacity),
                sampling_rate,
                frame_length,

                page_sidx: 0,
                page_n: 0,
                granule_pos: 0,

                // From the "page checksum" section at: https://xiph.org/vorbis/doc/framing.html
                // > 32 bit CRC value (direct algorithm, initial val and final XOR = 0, generator polynomial=0x04c11db7).
                // > The value is computed over the entire header (with the CRC field in the header set to zero) and
                // > then continued over the page. The CRC field is then filled with the computed value.
                crc_hasher: crc_any::CRCu32::create_crc(0x04c11db7, 32, 0x0000, 0x0000, false),
            }
        }

        fn finish_page(&mut self) {
            self.crc_hasher.update(self.bw.slice_from(self.page_sidx..));
            let crc = self.crc_hasher.get_crc();
            self.crc_hasher.reset();

            self.bw
                .write_slice_at(self.page_sidx + Self::CHECKSUM_OFFSET, crc.to_le_bytes());
            self.page_sidx = self.bw.cursor;
            self.page_n += 1;
        }

        fn write_header_page(&mut self, n_channels: u8, pre_skip: u16) {
            static REQ_CAPACITY: usize = 28 + 19;
            if self.bw.rem_capacity() < REQ_CAPACITY {
                self.bw.expand();
            }

            self.bw.write_slice(b"OggS"); // Capture pattern
            self.bw.write_u8(0); // Ogg bitstream format version
            self.bw.write_u8(0x02); // Header type - BOS
            self.bw.write_u64(0); // Granule position
            self.bw.write_u32(0); // Bitstream serial number
            self.bw.write_u32(self.page_n); // Page sequence number
            self.bw.skip(4); // Checksum - zeroized for now
            self.bw.write_u8(1); // Page segments
            self.bw.write_u8(19); // Segment table - length of the opus header that follows

            // Ogg Opus Identification Header, refer to https://www.rfc-editor.org/info/rfc7845/#section-5.1
            self.bw.write_slice("OpusHead"); // Magic signature
            self.bw.write_u8(1); // Version
            self.bw.write_u8(n_channels); // Output channel count
            self.bw.write_u16(pre_skip); // Pre-skip
            self.bw.write_u32(self.sampling_rate as u32); // Input sample rate
            self.bw.write_i16(0); // Output gain
            self.bw.write_u8(0); // Channel Mapping Family

            debug_assert_eq!(self.bw.cursor - self.page_sidx, REQ_CAPACITY);
            self.finish_page();
        }

        fn write_tags_page(&mut self) {
            static REQ_CAPACITY: usize = 28 + 16;
            if self.bw.rem_capacity() < REQ_CAPACITY {
                self.bw.expand();
            }

            self.bw.write_slice(b"OggS"); // Capture pattern
            self.bw.write_u8(0); // Ogg bitstream format version
            self.bw.write_u8(0x00); // Header type
            self.bw.write_u64(0); // Granule position
            self.bw.write_u32(0); // Bitstream serial number
            self.bw.write_u32(self.page_n); // Page sequence number
            self.bw.skip(4); // Checksum - zeroized for now
            self.bw.write_u8(1); // Page segments
            self.bw.write_u8(16); // Segment table - length of the opus comment header

            // Ogg Opus Comment Header, refer to https://www.rfc-editor.org/info/rfc7845/#section-5.2
            self.bw.write_slice("OpusTags"); // Magic signature
            self.bw.write_u32(0); // Vendor string length
            self.bw.write_u32(0); // User comment list length

            debug_assert_eq!(self.bw.cursor - self.page_sidx, REQ_CAPACITY);
            self.finish_page();
        }

        fn write_audio_page(&mut self, opus_frame: &[u8]) {
            let n_segments = opus_frame.len() / 255 + 1;
            assert!(n_segments <= 255);

            let req_capacity = 27 + n_segments + opus_frame.len();
            if self.bw.rem_capacity() < req_capacity {
                self.bw.expand();
            }

            // From: https://www.rfc-editor.org/info/rfc7845/#section-4
            // > The granule position of an audio data page encodes the total number
            // > of PCM samples in the stream up to and including the last fully
            // > decodable sample from the last packet completed on that page.
            // > …
            // > Therefore, the value in the granule position
            // > field always counts samples assuming a 48 kHz decoding rate, and the
            // > rest of this specification makes the same assumption.
            self.granule_pos += (48_000 / self.sampling_rate) * self.frame_length;

            self.bw.write_slice(b"OggS"); // Capture pattern
            self.bw.write_u8(0); // Ogg bitstream format version
            self.bw.write_u8(0x00); // Header type - note: not using 0x01 as we expect 1 page = 1 opus packet (to simplify)
            self.bw.write_u64(self.granule_pos); // Granule position (might be i64 for opus actually?)
            self.bw.write_u32(0); // Bitstream serial number
            self.bw.write_u32(self.page_n); // Page sequence number
            self.bw.skip(4); // Checksum - zeroized for now
            self.bw.write_u8(n_segments as u8); // Page segments
            self.bw.fill(u8::MAX, n_segments - 1); // Won't underflow as n_segments >= 1
            self.bw.write_u8((opus_frame.len() % 255) as u8);

            self.bw.write_slice(opus_frame);

            debug_assert_eq!(self.bw.cursor - self.page_sidx, req_capacity);
            self.finish_page();
        }

        fn write_final_audio_page(&mut self, raw_length: usize, opus_frame: &[u8]) {
            let n_segments = opus_frame.len() / 255 + 1;
            assert!(n_segments <= 255);

            let req_capacity = 27 + n_segments + opus_frame.len();
            if self.bw.rem_capacity() < req_capacity {
                self.bw.expand();
            }

            self.granule_pos += (48_000 / self.sampling_rate) * raw_length as u64;

            self.bw.write_slice(b"OggS"); // Capture pattern
            self.bw.write_u8(0); // Ogg bitstream format version
            self.bw.write_u8(0x04); // Header type - EOS
            self.bw.write_u64(self.granule_pos); // Granule position (might be i64 for opus actually?)
            self.bw.write_u32(0); // Bitstream serial number
            self.bw.write_u32(self.page_n); // Page sequence number
            self.bw.skip(4); // Checksum - zeroized for now
            self.bw.write_u8(n_segments as u8); // Page segments
            self.bw.fill(u8::MAX, n_segments.saturating_sub(1));
            self.bw.write_u8((opus_frame.len() % 255) as u8);

            self.bw.write_slice(opus_frame);

            debug_assert_eq!(self.bw.cursor - self.page_sidx, req_capacity);
            self.finish_page();
        }
    }
}

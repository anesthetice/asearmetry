/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::{
    math::{Hertz, Radians, Seconds, apply_tukey_window},
    signal::{
        DSP, FreqDomain, Signal,
        audio::{AudioBuffer, AudioBufferSlice, TimeDomain},
    },
};
use itertools::Itertools;
use num_complex::Complex32;
use std::{
    f32::consts::TAU as TAU_F32, f64::consts::TAU as TAU_F64, marker::PhantomData, ops::MulAssign,
};
use std::{fs::File, path::Path};
use symphonia::{
    core::{
        codecs::{audio::AudioDecoderOptions, registry::CodecRegistry},
        errors::Error as SymphError,
        formats::{FormatOptions, TrackType, probe::Hint},
        io::MediaSourceStream,
        meta::MetadataOptions,
    },
    default::register_enabled_codecs,
};

pub type MonoAudioBuf = AudioBuffer<1>;
pub type MonoAudioBufSlice<'a> = AudioBufferSlice<'a, 1>;

pub type StereoAudioBuf = AudioBuffer<2>;
pub type StereoAudioBufSlice<'a> = AudioBufferSlice<'a, 2>;

// Start of [`MonoAudioBuf`] related code
//
impl MonoAudioBuf {
    pub fn new_mono(samples: Vec<f32>, sampling_rate: Option<Hertz>) -> Self {
        AudioBuffer {
            channels: [samples],
            sampling_rate,
            _domain: PhantomData,
        }
    }

    pub fn into_stereo(self) -> StereoAudioBuf {
        let [samples] = self.channels;
        StereoAudioBuf::new([samples.clone(), samples], self.sampling_rate)
    }

    pub fn sinusoidal(
        duration: Seconds,
        sampling_rate: Hertz,
        amplitude: f64,
        frequency: Hertz,
        phase_offset: Radians,
    ) -> Self {
        let nb_samples = (sampling_rate * duration).ceil() as usize;
        let δt = 1.0 / sampling_rate;
        let ω = TAU_F64 * frequency;

        AudioBuffer::new_zeros(nb_samples)
            .with_sr(sampling_rate)
            .apply_enumerate(&mut |(n, _)| {
                (amplitude * f64::sin((n as f64) * δt * ω + phase_offset)) as f32
            })
    }

    #[allow(non_snake_case)]
    pub fn noise(
        duration: Seconds,
        sampling_rate: Hertz,
        freq_bounds: (Hertz, Hertz),
        alpha_tukey_window: f64,
        rng: &mut impl rand::RngExt,
    ) -> Self {
        let (f_min, f_max) = freq_bounds;
        println!("{f_min}, {f_max}, {sampling_rate}");

        assert!(duration > 0.0);
        assert!(sampling_rate > 0.0);
        assert!(f_min >= 0.0 && f_min < f_max && f_min < sampling_rate / 2.0);

        let N_t: usize = (sampling_rate * duration) as usize;
        let N_f: usize = (N_t / 2) + 1;
        let Δf: Hertz = sampling_rate / N_t as f64;

        let N_f_nonzero_start = ((f_min / Δf).ceil() as usize).max(1);
        let N_f_nonzero_end = ((f_max / Δf).ceil() as usize).min(N_f - 2);

        // Note that spectrum_halved[0] is already 0 (→ mean of zero which is what we want).
        let mut spectrum_halved = vec![Complex32::ZERO; N_f];

        for z in &mut spectrum_halved[N_f_nonzero_start..=N_f_nonzero_end] {
            let phase = rng.random_range(0.0..TAU_F32);
            *z = Complex32::from_polar(1.0, phase);
        }

        apply_tukey_window(
            &mut spectrum_halved[N_f_nonzero_start..=N_f_nonzero_end],
            alpha_tukey_window,
        );

        Signal::<1, Complex32, FreqDomain>::new([spectrum_halved], Some(sampling_rate))
            .idft_halved(N_t)
            .normalize()
    }

    /// Load data from an audio file, works for audio files whose codec is supported
    /// by symphonia: https://docs.rs/symphonia/latest/symphonia/, channels are averaged
    /// to form the mono one returned here.
    pub fn load_from_file<Q: AsRef<Path>>(filepath: Q) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();

        let mss = MediaSourceStream::new(Box::new(File::open(filepath)?), Default::default());

        // Let Symphonia guess the format.
        let hint = Hint::new();
        let mut format = symphonia::default::get_probe().probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )?;

        // Select the first audio track.
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| anyhow::anyhow!("No default audio track found"))?;

        let track_id = track.id;

        let track_audio_codec_params = track
            .codec_params
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Audio track is missing codec parameters"))?
            .audio()
            .unwrap();

        let track_n_channels = track_audio_codec_params
            .channels
            .as_ref()
            .map(|chas| chas.count())
            .ok_or_else(|| anyhow::anyhow!("Audio track does not contain any channels"))?;

        let sampling_rate = track_audio_codec_params // safe as `TrackType::Audio` is set above
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("Unknown sampling rate"))?;

        log::debug!(
            "Starting to decode audio file located at '{}', track id: {track_id}, codec parameters: {track_audio_codec_params:?}, sampling rate: {sampling_rate} [Hz]",
            filepath.display()
        );

        let mut codec_registry = CodecRegistry::new();
        register_enabled_codecs(&mut codec_registry);
        #[cfg(feature = "opus")]
        codec_registry.register_audio_decoder::<symphonia_adapter_libopus::OpusDecoder>();

        // Create decoder.
        let mut decoder = codec_registry
            .make_audio_decoder(track_audio_codec_params, &AudioDecoderOptions::default())?;

        let mut raw_samples: Vec<f32> = Vec::new();
        let mut raw_samples_holder: Vec<f32> = Vec::new();

        // Decode packets.
        loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break, // Reached the end of the stream.
                Err(err) => {
                    anyhow::bail!(err)
                }
            };

            if packet.track_id != track_id {
                log::warn!("Packet ID does not match track ID");
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    decoded.copy_to_vec_interleaved(&mut raw_samples_holder);
                    raw_samples.extend_from_slice(&raw_samples_holder);
                    raw_samples_holder.clear();
                }
                Err(err @ SymphError::IoError(_)) => {
                    log::warn!("Failed to decode a packet due to an IO issue, {err}");
                    continue;
                }
                Err(err @ SymphError::DecodeError(_)) => {
                    log::warn!("Failed to decode a packet due to a decoder issue, {err}");
                    continue;
                }
                Err(err) => {
                    anyhow::bail!(err);
                }
            }
        }

        log::debug!(
            "Finished decoding the audio file at '{}', got {} samples at {sampling_rate} for a total duration of ~{:.1} [s]",
            filepath.display(),
            raw_samples.len(),
            raw_samples.len() as f64 / sampling_rate as f64
        );

        let mono_samples = raw_samples
            .chunks_exact(track_n_channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / track_n_channels as f32)
            .collect_vec();

        Ok(Self::new_mono(mono_samples, Some(sampling_rate as f64)))
    }
}

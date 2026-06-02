// Imports
use crate::{
    audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal},
    math::{Hertz, Meters, Radians, Seconds},
};
use itertools::Itertools;
use std::f64::consts::TAU;
use std::{fs::File, path::Path};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error as SymphError,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
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
        assert!(amplitude.abs() <= 1.0);

        let nb_samples = (sampling_rate * duration).ceil() as usize;
        let δt = 1.0 / sampling_rate;
        let ω = TAU * frequency;

        AudioBuffer::new_zeros(nb_samples)
            .with_sr(sampling_rate)
            .apply_enumerate(&mut |(n, _)| {
                (amplitude * f64::sin((n as f64) * δt * ω + phase_offset)) as f32
            })
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

        println!("{track_audio_codec_params:?}");

        let track_n_channels = track_audio_codec_params
            .channels
            .as_ref()
            .map(|chas| chas.count())
            .ok_or_else(|| anyhow::anyhow!("Audio track does not contain any channels"))?;

        let sampling_rate = track_audio_codec_params // safe as `TrackType::Audio` is set above
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("Unknown sampling rate"))?;
        println!("Sampling rate: {} Hz", sampling_rate);

        // Create decoder.
        let mut decoder = symphonia::default::get_codecs()
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
                eprintln!("AA: {}", packet.track_id);
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    decoded.copy_to_vec_interleaved(&mut raw_samples_holder);
                    raw_samples.extend_from_slice(&raw_samples_holder);
                    raw_samples_holder.clear();
                }
                Err(err @ SymphError::IoError(_)) => {
                    eprintln!("Failed to decode a packet due to an IO issue, {err}");
                    continue;
                }
                Err(err @ SymphError::DecodeError(_)) => {
                    eprintln!("Failed to decode a packet due to a decoder issue, {err}");
                    continue;
                }
                Err(err) => {
                    anyhow::bail!(err);
                }
            }
        }

        println!("Decoded {} f32 samples", raw_samples.len());

        let mono_samples = raw_samples
            .chunks_exact(track_n_channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / track_n_channels as f32)
            .collect_vec();

        Ok(Self::new_mono(mono_samples, Some(sampling_rate as f64)))
    }
}

impl StereoAudioBuf {
    pub fn from_left_right_mono<T1, T2>(left: T1, right: T2) -> Self
    where
        T1: DiscreteSignal<1>,
        T2: DiscreteSignal<1>,
    {
        let sampling_rate =
            Self::resolve_sampling_rate_pair(left.sampling_rate(), right.sampling_rate());
        let [left] = left.into_owned().channels;
        let [right] = right.into_owned().channels;
        Self::new([left, right], sampling_rate)
    }
}

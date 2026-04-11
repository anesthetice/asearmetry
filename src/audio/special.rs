// Imports
use crate::{
    Hertz, Radians, Seconds,
    audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal},
};
use std::f32::consts::TAU;
use std::{fs::File, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error as SymphError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

pub type MonoAudioBuf = AudioBuffer<1>;
pub type MonoAudioBufSlice<'a> = AudioBufferSlice<'a, 1>;

pub type StereoAudioBuf = AudioBuffer<2>;
pub type StereoAudioBufSlice<'a> = AudioBufferSlice<'a, 2>;

// Start of [`MonoAudioBuf`] related code
//
impl MonoAudioBuf {
    pub fn new_mono(samples: Vec<f32>, sampling_rate: Option<u32>) -> Self {
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
        sampling_rate: u32,
        amplitude: f32,
        frequency: Hertz,
        phase_offset: Radians,
    ) -> Self {
        assert!(amplitude.abs() <= 1.0);

        let nb_samples = (sampling_rate as f32 * duration).ceil() as usize;
        let δt = 1.0 / sampling_rate as f32;
        let ω = TAU * frequency;

        AudioBuffer::new_zeros(nb_samples)
            .with_sr(sampling_rate)
            .apply_with_context(|(n, _)| amplitude * f32::sin((n as f32) * δt * ω + phase_offset))
    }

    /// Load data from an audio file, works for audio files whose codec is supported
    /// by symphonia: https://docs.rs/symphonia/latest/symphonia/, channels are averaged
    /// to form the mono one returned here.
    pub fn load_from_file<Q: AsRef<Path>>(filepath: Q) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();

        let mss = MediaSourceStream::new(Box::new(File::open(filepath)?), Default::default());

        // Let Symphonia guess the format.
        let hint = Hint::new();
        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;
        let mut format = probed.format;

        // Select the first audio track.
        let track = format
            .default_track()
            .ok_or_else(|| anyhow::anyhow!("No default track found"))?;

        let track_id = track.id;

        let sampling_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("Unknown sampling rate"))?;
        println!("Sampling rate: {} Hz", sampling_rate);

        // Create decoder.
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())?;

        let mut samples: Vec<f32> = Vec::new();

        // Decode packets.
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphError::IoError(io_error)) => {
                    if io_error.kind() == std::io::ErrorKind::UnexpectedEof {
                        break;
                    } else {
                        anyhow::bail!(io_error);
                    }
                }
                Err(err) => {
                    anyhow::bail!(err)
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
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
            };

            let spec = *decoded.spec();
            let mut buf_f32 = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buf_f32.copy_interleaved_ref(decoded);

            let channels = spec.channels.count();
            let mixed_samples = buf_f32.samples();

            for frame in mixed_samples.chunks_exact(channels) {
                let sum: f32 = frame.iter().copied().sum();
                samples.push(sum / channels as f32);
            }
        }

        println!("Decoded {} f32 samples", samples.len());
        Ok(Self::new_mono(samples, Some(sampling_rate)))
    }
}

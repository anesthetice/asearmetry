use crate::signal::MonoSig;
use std::{fs::File, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error as SymphError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

pub fn read_audio_file<Q: AsRef<Path>>(filepath: Q) -> anyhow::Result<MonoSig> {
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

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("Unknown sample rate"))?;
    println!("Sample rate: {} Hz", sample_rate);

    // Create decoder.
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

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
    Ok(MonoSig::new_mono(samples, sample_rate as f32))
}

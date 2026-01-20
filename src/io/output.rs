use crate::signal::{Signal, StereoSig};
use std::path::Path;

pub fn write_audio_file<Q: AsRef<Path>>(filepath: Q, signal: &StereoSig) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: signal.sampling_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(filepath, spec)?;
    signal.write_to(&mut writer)?;

    writer.finalize()?;
    Ok(())
}

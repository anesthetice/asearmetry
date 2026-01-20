use crate::signal::{AudioBuffer, StereoAudioBuf};
use std::path::Path;

pub fn write_audio_file<Q: AsRef<Path>>(
    filepath: Q,
    signal: &StereoAudioBuf,
) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: signal.sample_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(filepath, spec)?;
    signal.write_to(&mut writer)?;

    writer.finalize()?;
    Ok(())
}

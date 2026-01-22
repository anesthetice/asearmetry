// Modules
mod input;
mod output;

// Exports
pub use input::read_audio_file;
pub use output::write_audio_file;

use crate::{
    coordinates::Sphere3D,
    signal::{AudioBuffer, StereoAudioBuf},
};
use anyhow::anyhow;
use itertools::Itertools;
use polars::prelude::*;
use std::{collections::HashMap, path::Path};

pub fn load_hrir_data<Q: AsRef<Path>>(
    filepath: Q,
) -> anyhow::Result<Vec<(Sphere3D, StereoAudioBuf)>> {
    let mut reader = ParquetReader::new(std::fs::File::open(filepath)?);

    let key_value_metadata = reader
        .get_metadata()?
        .key_value_metadata
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Missing key-value metadata in parquet file"))?
        .into_iter()
        .filter_map(|kv| Some((kv.key, kv.value?)))
        .collect::<HashMap<String, String>>();

    let sample_rate = key_value_metadata
        .get("sample_rate")
        .ok_or_else(|| anyhow!("Missing `sampling_rate` key-value pair"))?
        .parse::<f32>()?;

    let df = reader.finish().unwrap();

    let process_float_col = |name: &str| -> anyhow::Result<Vec<f32>> {
        let col = df.column(name)?.f32()?;
        Ok(col.into_no_null_iter().collect_vec())
    };

    let process_list_float_col = |name: &str| -> anyhow::Result<Vec<Vec<f32>>> {
        let col = df.column(name)?.list()?;
        Ok(col
            .into_iter()
            .map(|s| s.unwrap().f32().unwrap().into_no_null_iter().collect_vec())
            .collect_vec())
    };

    let src_radius_vec = process_float_col("src_radius")?;
    let src_azimuth_vec = process_float_col("src_azimuth")?;
    let src_zenith_vec = process_float_col("src_zenith")?;
    let hrir_left_vec = process_list_float_col("hrir_left")?;
    let hrir_right_vec = process_list_float_col("hrir_right")?;

    Ok(itertools::izip!(
        src_radius_vec,
        src_azimuth_vec,
        src_zenith_vec,
        hrir_left_vec,
        hrir_right_vec
    )
    .map(|(radius, azimuth, zenith, hrir_left, hrir_right)| {
        let position = Sphere3D::new(radius, azimuth, zenith).and_clamp_angles();
        let hrir = AudioBuffer::new([hrir_left, hrir_right], sample_rate);
        (position, hrir)
    })
    .collect_vec())
}

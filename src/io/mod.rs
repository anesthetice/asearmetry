// Modules
mod input;
mod output;

use std::collections::HashMap;

use anyhow::anyhow;
// Exports
pub use input::read_audio_file;
use itertools::Itertools;
pub use output::write_audio_file;

use polars::prelude::*;

use crate::signal::{Signal, StereoSig};

pub fn load_hrir() -> anyhow::Result<StereoSig> {
    let mut reader = ParquetReader::new(std::fs::File::open("sofa_conversion/hrtf.parquet")?);

    let key_value_metadata = reader
        .get_metadata()?
        .key_value_metadata
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Missing key-value metadata in parquet file"))?
        .into_iter()
        .filter_map(|kv| Some((kv.key, kv.value?)))
        .collect::<HashMap<String, String>>();

    let sampling_rate = key_value_metadata
        .get("sampling_rate")
        .ok_or_else(|| anyhow!("Missing `sampling_rate` key-value pair"))?
        .parse::<f32>()?;

    let df = reader.finish().unwrap();

    let AnyValue::List(ref list) = df.get(200).unwrap()[3] else {
        panic!();
    };
    let left = list
        .cast(&DataType::Float32)?
        .f32()
        .unwrap()
        .into_iter()
        .flatten()
        .collect_vec();

    let AnyValue::List(ref list) = df.get(200).unwrap()[4] else {
        panic!();
    };
    let right = list
        .cast(&DataType::Float32)?
        .f32()
        .unwrap()
        .into_iter()
        .flatten()
        .collect_vec();

    Ok(StereoSig::new_stereo(left, right, sampling_rate))
}

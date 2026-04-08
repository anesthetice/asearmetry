// Imports
use crate::{
    audio::StereoAudioBuf,
    brp::{Binauralizer, HrirProjection},
    coordinates::{Cart3D, Shell2D, Sphere3D},
};
use anyhow::anyhow;
use itertools::Itertools;
use polars::prelude::*;
use rstar::RTree;
use std::{collections::HashMap, path::Path};

impl Binauralizer {
    pub fn load_from_file<Q: AsRef<Path>>(filepath: Q) -> anyhow::Result<Self> {
        let mut reader = ParquetReader::new(std::fs::File::open(filepath)?);

        let key_value_metadata = reader
            .get_metadata()?
            .key_value_metadata
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Missing key-value metadata in parquet file"))?
            .into_iter()
            .filter_map(|kv| Some((kv.key, kv.value?)))
            .collect::<HashMap<String, String>>();

        let hrir_sampling_rate = key_value_metadata
            .get("sampling_rate")
            .ok_or_else(|| anyhow!("Missing `sampling_rate` key-value pair"))?
            .parse::<u32>()?;

        let left_ear_pos: Cart3D = key_value_metadata
            .get("left_ear_position_cartesian")
            .ok_or_else(|| anyhow!("Missing `left_ear_position_cartesian` key-value pair"))?
            .split(",")
            .map(|s| s.trim().parse::<f32>().unwrap())
            .collect_tuple::<(f32, f32, f32)>()
            .unwrap()
            .into();

        let right_ear_pos: Cart3D = key_value_metadata
            .get("right_ear_position_cartesian")
            .ok_or_else(|| anyhow!("Missing `right_ear_position_cartesian` key-value pair"))?
            .split(",")
            .map(|s| s.trim().parse::<f32>().unwrap())
            .collect_tuple::<(f32, f32, f32)>()
            .unwrap()
            .into();

        println!(
            "left ear: {:?}\nright ear: {:?}",
            left_ear_pos, right_ear_pos
        );

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

        assert!(
            src_radius_vec.iter().all_equal(),
            "Currently only handles cases where the radius of the source is constant."
        );

        assert!(hrir_left_vec.iter().map(Vec::len).all_equal());
        assert!(hrir_right_vec.iter().map(Vec::len).all_equal());

        let hrir_size = hrir_left_vec.first().unwrap().len();

        let data = itertools::izip!(
            src_radius_vec,
            src_azimuth_vec,
            src_zenith_vec,
            hrir_left_vec,
            hrir_right_vec
        )
        .map(|(radius, azimuth, zenith, hrir_left, hrir_right)| {
            let position: Shell2D = Sphere3D::new(radius, azimuth, zenith).clamp_angles().into();
            let hrir = StereoAudioBuf::from([hrir_left, hrir_right]);
            /*
            let a = hrir.cha(1).iter().copied().reduce(f32::max).unwrap();
            println!("{a}");
            println!(
                "{:?}",
                hrir.cha(0).iter().map(|x| format!("{x:.2E}")).join(", ")
            );
            println!(
                "\n{:?}",
                hrir.cha(1).iter().map(|x| format!("{x:.2E}")).join(", ")
            );
            //panic!("");
            */
            HrirProjection::new(position, hrir)
        })
        .collect_vec();

        let hrir_tree = RTree::bulk_load(data);

        let binaur = Binauralizer {
            hrir_tree,
            hrir_radius: 1.0,
            hrir_size,
            hrir_sampling_rate,
            left_ear_pos,
            right_ear_pos,
        };

        Ok(binaur)
    }
}

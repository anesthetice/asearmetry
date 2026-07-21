/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::{
    audio::{ASP, AudioBuffer, StereoAudioBuf},
    binaur::{Binauralizer, HrirProjection},
    coordinates::{Cart3D, Shell2D},
    math::{Hertz, Meters},
    signal::DSP,
};
use anyhow::{Context, anyhow};
use itertools::Itertools;
use polars::prelude::*;
use std::{collections::HashMap, path::Path};
use tap::Pipe;

#[derive(Debug, Clone)]
pub struct BinauralizerPrecursor {
    pub hrir_vec: Vec<StereoAudioBuf>,
    pub hrir_pos_vec: Vec<Shell2D>,
    pub hrir_radius: Meters,
    pub hrir_size: usize,
    pub hrir_sampling_rate: Hertz,
    pub left_ear_pos: Cart3D,
    pub right_ear_pos: Cart3D,
    pub brir_opt: Option<StereoAudioBuf>,
}

impl BinauralizerPrecursor {
    pub fn load_from_file<Q: AsRef<Path>>(filepath: Q) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();

        log::debug!(
            "Attempting to load a BinauralizerPrecursor from the file at: {}",
            filepath.display()
        );

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
            .parse::<Hertz>()?;

        let left_ear_pos: Cart3D = key_value_metadata
            .get("left_ear_position_cartesian")
            .ok_or_else(|| anyhow!("Missing `left_ear_position_cartesian` key-value pair"))?
            .split(",")
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect_tuple::<(f64, f64, f64)>()
            .ok_or_else(|| {
                anyhow!(
                    "Invalid `left_ear_position_cartesian` value, could not extract (f32; 3) tuple"
                )
            })?
            .into();

        let right_ear_pos: Cart3D = key_value_metadata
            .get("right_ear_position_cartesian")
            .ok_or_else(|| anyhow!("Missing `right_ear_position_cartesian` key-value pair"))?
            .split(",")
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect_tuple::<(f64, f64, f64)>()
            .ok_or_else(|| {
                anyhow!(
                    "Invalid `right_ear_position_cartesian` value, could not extract (f32; 3) tuple"
                )
            })?
            .into();

        log::debug!(
            "Finished parsing dataframe metadata, got: sampling rate: {hrir_sampling_rate:.1} Hz, left ear position: {left_ear_pos}, right ear position: {right_ear_pos}"
        );

        let df = reader.finish()?;

        let process_float_col = |name: &str| -> anyhow::Result<Vec<f64>> {
            df.column(name)
                .with_context(|| format!("The desired column `{name}` does not exist"))?
                .f64()
                .with_context(|| {
                    format!("Expected the column `{name}` to have a `Float64` datatype")
                })?
                .into_no_null_iter()
                .collect_vec()
                .pipe(Ok)
        };

        let process_list_float_col = |name: &str| -> anyhow::Result<Vec<Vec<f32>>> {
            df.column(name)
                .with_context(|| format!("The desired column `{name}` does not exist"))?
                .list()
                .with_context(|| format!("Expected the column `{name}` to have a `List` datatype"))?
                .into_iter()
                .map(|s| s.unwrap().f32().unwrap().into_no_null_iter().collect_vec())
                .collect_vec()
                .pipe(Ok)
        };

        let src_radius_vec = process_float_col("src_radius")?;
        let src_azimuth_vec = process_float_col("src_azimuth")?;
        let src_zenith_vec = process_float_col("src_zenith")?;
        let hrir_left_vec = process_list_float_col("hrir_left")?;
        let hrir_right_vec = process_list_float_col("hrir_right")?;

        {
            let comparator = src_radius_vec.first().unwrap();
            assert!(
                src_radius_vec
                    .iter()
                    .all(|r| approx::abs_diff_eq!(r, comparator, epsilon = 1E-2)),
                "Currently only handles cases where the radius of the source is constant."
            );
        }

        let hrir_radius = src_radius_vec.first().unwrap().round();

        assert!(hrir_left_vec.iter().map(Vec::len).all_equal());
        assert!(hrir_right_vec.iter().map(Vec::len).all_equal());
        assert_eq!(
            hrir_left_vec.first().map(|v| v.len()),
            hrir_right_vec.first().map(|v| v.len())
        );

        let hrir_size = hrir_left_vec.first().unwrap().len();

        let hrir_vec = itertools::izip!(hrir_left_vec, hrir_right_vec)
            .map(|(hrir_left, hrir_right)| {
                StereoAudioBuf::from([hrir_left, hrir_right]).with_sr(hrir_sampling_rate)
            })
            .collect_vec();

        let hrir_pos_vec = itertools::izip!(src_azimuth_vec, src_zenith_vec,)
            .map(|(azimuth, zenith)| Shell2D::new(azimuth, zenith).clamp_angles())
            .collect_vec();

        log::debug!(
            "Finished loading `BinauralizerPrecursor`, with {} HRIRs",
            hrir_vec.len()
        );

        Ok(BinauralizerPrecursor {
            hrir_vec,
            hrir_pos_vec,
            hrir_radius,
            hrir_size,
            hrir_sampling_rate,
            left_ear_pos,
            right_ear_pos,
            brir_opt: None,
        })
    }

    pub fn into_binauralizer(mut self) -> Binauralizer {
        // The first important step is to remove the delay in the HRIRs
        self.hrir_vec.iter_mut().for_each(|hrir| {
            let hrir_spike_smooth = hrir
                .clone()
                .abs()
                .apply_gaussian_filter(10, 0.05)
                .normalize_to(hrir.get_abs_max());

            const CENTER_INDEX: usize = 30;

            // We align both peaks to the currently arbitrary position of 30
            let mut adjust_peak = |c: usize| {
                let peak_idx = hrir_spike_smooth.index_cha(c).get_abs_max_index();

                #[allow(non_snake_case)]
                let Δ_abs = peak_idx.abs_diff(CENTER_INDEX);

                // peak is to the right, we remove Δ_abs elements from the start to shift it to the left
                if peak_idx > CENTER_INDEX {
                    hrir.cha_mut_uc(c).drain(0..Δ_abs);
                    hrir.cha_mut_uc(c).resize(self.hrir_size, 0.0);
                }
                // peak is to the left, we add Δ_abs elements to the start to shift it to the right
                else if peak_idx < CENTER_INDEX {
                    let mut vec = vec![0.0; Δ_abs];
                    vec.extend_from_slice(hrir.cha_uc(c));
                    vec.resize(self.hrir_size, 0.0);
                    *hrir.cha_mut_uc(c) = vec;
                }

                // TODO: maybe apply window function here as well to avoid any discontinuities?
            };

            adjust_peak(0);
            adjust_peak(1);
        });

        /*
        let hrir_rtree = rstar::RTree::bulk_load(
            self.hrir_vec
                .into_iter()
                .zip(self.hrir_pos_vec)
                .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
                .collect(),
        );
        */

        // The second step is to interpolate more HRIRs
        let hrir_rtree = {
            // Original non-interpolated rtree
            let hrir_rtree_ni = rstar::RTree::bulk_load(
                self.hrir_vec
                    .into_iter()
                    .zip(self.hrir_pos_vec)
                    .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
                    .collect(),
            );

            const N_NEAREST: usize = 3;

            let rtree_elements = Shell2D::generate_fib_lattice(4096)
                .into_iter()
                .map(|new_pos| {
                    //println!("desired position: {}", new_pos);
                    let mut total_dist = 0.0;
                    let mut dists = Vec::with_capacity(N_NEAREST);
                    let mut hrirs = Vec::with_capacity(N_NEAREST);

                    hrir_rtree_ni
                        .nearest_neighbor_iter(new_pos.into())
                        .take(N_NEAREST)
                        .for_each(|gwd| {
                            let dist = gwd.geom().dist(new_pos);
                            //println!("• close to: {}, with a dist of: {}", gwd.geom(), dist);
                            total_dist += dist;
                            dists.push(dist);
                            hrirs.push(gwd.data.view());
                        });

                    //println!();

                    assert!(total_dist > 0.0);

                    let new_hrir = hrirs.into_iter().zip(dists).fold(
                        AudioBuffer::<2>::with_capacity(self.hrir_size, None),
                        |new_hrir, (hrir, dist)| {
                            let weight = (dist / total_dist) as f32;
                            new_hrir.merge_with(hrir.apply(|x| x * weight))
                        },
                    );

                    HrirProjection::new(new_pos, new_hrir)
                })
                .collect_vec();

            rstar::RTree::bulk_load(rtree_elements)
        };

        Binauralizer {
            hrir_rtree,
            hrir_radius: self.hrir_radius,
            hrir_size: self.hrir_size,
            hrir_sampling_rate: self.hrir_sampling_rate,
            left_ear_pos: self.left_ear_pos,
            right_ear_pos: self.right_ear_pos,
        }
    }

    pub fn into_binauralizer_old(mut self) -> Binauralizer {
        // The first important step is to remove the delay in the HRIRs
        let mut max_len: usize = 0;
        self.hrir_vec.iter_mut().for_each(|hrir| {
            let hrir_spike_smooth = hrir
                .clone()
                .abs()
                .apply_median_filter(4)
                //.apply_gaussian_filter(4, 0.33)
                .normalize_to(hrir.get_abs_max());

            let find_start = |c: usize| {
                let (μ, σ): (f32, f32) = {
                    let slice = &hrir_spike_smooth.cha_uc(c)[0..10];
                    (crate::math::mean(slice), crate::math::std(slice, 1))
                };
                let threshold = μ + 10.0 * σ;
                hrir_spike_smooth
                    .cha_uc(c)
                    .iter()
                    .find_position(|x| **x > threshold)
                    .unwrap()
                    .0
            };

            let left_start = find_start(0);
            let right_start = find_start(1);

            max_len = max_len
                .max(self.hrir_size - left_start)
                .max(self.hrir_size - right_start);

            hrir.cha_mut_uc(0).drain(0..left_start);
            hrir.cha_mut_uc(1).drain(0..right_start);
        });

        let hrir_rtree = rstar::RTree::bulk_load(
            self.hrir_vec
                .into_iter()
                .zip(self.hrir_pos_vec)
                .map(|(mut hrir, pos)| {
                    hrir = hrir.pad_right_with_last_to_len(max_len);
                    HrirProjection::new(pos, hrir)
                })
                .collect(),
        );

        Binauralizer {
            hrir_rtree,
            hrir_radius: self.hrir_radius,
            hrir_size: max_len,
            hrir_sampling_rate: self.hrir_sampling_rate,
            left_ear_pos: self.left_ear_pos,
            right_ear_pos: self.right_ear_pos,
        }
    }

    pub fn into_binauralizer_no_processing(self) -> Binauralizer {
        let hrir_rtree = rstar::RTree::bulk_load(
            self.hrir_vec
                .into_iter()
                .zip(self.hrir_pos_vec)
                .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
                .collect(),
        );

        Binauralizer {
            hrir_rtree,
            hrir_radius: self.hrir_radius,
            hrir_size: self.hrir_size,
            hrir_sampling_rate: self.hrir_sampling_rate,
            left_ear_pos: self.left_ear_pos,
            right_ear_pos: self.right_ear_pos,
        }
    }
}

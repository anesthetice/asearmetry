/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::{
    binaur::{Binauralizer, HrirProjection},
    coordinates::{Cart3D, Shell2D},
    math::{Hertz, Meters},
    signal::{
        DSP,
        audio::{ASP, AudioBuffer, StereoAudioBuf},
    },
    utils::{BCursor, read_from_file},
};
use itertools::Itertools;
use std::{io::Read, path::Path};

#[derive(Debug, Clone)]
pub struct BinauralizerPrecursor {
    pub hrir_vec: Vec<StereoAudioBuf>,
    pub hrir_pos_vec: Vec<Shell2D>,
    pub hrir_radius: Meters,
    pub hrir_length: usize,
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

        let buf = read_from_file(filepath, Some(6291456))?; // 6 Mebibytes
        let mut bcursor = BCursor::new(buf.as_slice());

        let hrir_sampling_rate = bcursor.try_capture_f64()?;
        let hrir_length = bcursor.try_capture_u64()? as usize;
        let left_ear_pos = Cart3D::from(bcursor.try_capture_f64_trio()?);
        let right_ear_pos = Cart3D::from(bcursor.try_capture_f64_trio()?);
        let hrir_radius = bcursor.try_capture_f64()?;

        let n_elements = bcursor.try_capture_u64()? as usize;
        let mut hrir_vec = Vec::with_capacity(n_elements);
        let mut hrir_pos_vec = Vec::with_capacity(n_elements);
        for _ in 0..n_elements {
            hrir_pos_vec.push(Shell2D::from(bcursor.try_capture_f64_duo()?).clamp_angles());
            hrir_vec.push(StereoAudioBuf::new(
                [
                    bcursor.try_capture_f32_array(hrir_length)?.to_vec(),
                    bcursor.try_capture_f32_array(hrir_length)?.to_vec(),
                ],
                Some(hrir_sampling_rate),
            ));
        }

        log::debug!(
            "Finished parsing dataframe metadata, got: sampling rate: {hrir_sampling_rate:.1} Hz, left ear position: {left_ear_pos}, right ear position: {right_ear_pos}"
        );

        log::debug!(
            "Finished loading `BinauralizerPrecursor`, with {} HRIRs",
            n_elements
        );

        Ok(BinauralizerPrecursor {
            hrir_vec,
            hrir_pos_vec,
            hrir_radius,
            hrir_length,
            hrir_sampling_rate,
            left_ear_pos,
            right_ear_pos,
            brir_opt: None,
        })
    }

    pub fn into_binauralizer(mut self) -> Binauralizer {
        // The first important step is to remove the delay in the HRIRs (or rather make it uniform)
        self.hrir_vec = self
            .hrir_vec
            .into_iter()
            .map(|hrir| {
                let hrir_spike_smooth = hrir
                    .view()
                    .abs()
                    .apply_gaussian_filter(10, 0.05)
                    .normalize_to(hrir.get_abs_max());

                const CENTER_INDEX: isize = 30;

                // We align both peaks to the currently arbitrary position of `CENTER_INDEX`
                let left_peak_idx = hrir_spike_smooth.index_cha(0).get_abs_max_index() as isize;
                let right_peak_idx = hrir_spike_smooth.index_cha(1).get_abs_max_index() as isize;

                hrir.shift(
                    [CENTER_INDEX - left_peak_idx, CENTER_INDEX - right_peak_idx],
                    [0.0; 2],
                )
                // TODO: maybe apply window function here as well to avoid any discontinuities?
            })
            .collect();

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
        const N_NEAREST: usize = 4;
        const N_INTERP_POINTS: usize = 16_384 * 8;
        let hrir_rtree = {
            // Original non-interpolated rtree
            let hrir_rtree_ni = rstar::RTree::bulk_load(
                self.hrir_vec
                    .into_iter()
                    .zip(self.hrir_pos_vec)
                    .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
                    .collect(),
            );

            let rtree_elements = Shell2D::generate_fib_lattice(N_INTERP_POINTS)
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
                            let dist = gwd.geom().dist_angular(new_pos);
                            //println!("• close to: {}, with a dist of: {}", gwd.geom(), dist);
                            total_dist += dist;
                            dists.push(dist);
                            hrirs.push(gwd.data.view());
                        });

                    //println!();

                    assert!(total_dist > 0.0);

                    // Higher is better this time, unlike the angular distance.
                    let scores = dists
                        .into_iter()
                        .map(|dist| total_dist - dist)
                        .collect_vec();
                    let total_score = scores.iter().sum::<f64>();

                    let hrir_interp = hrirs.into_iter().zip(scores).fold(
                        AudioBuffer::<2>::with_capacity(self.hrir_length, None),
                        |hrir_interp, (hrir, score)| {
                            let weight = (score / total_score) as f32;
                            hrir_interp.merge_with(hrir.apply(|x| x * weight))
                        },
                    );

                    HrirProjection::new(new_pos, hrir_interp)
                })
                .collect_vec();

            rstar::RTree::bulk_load(rtree_elements)
        };

        Binauralizer {
            hrir_rtree,
            hrir_radius: self.hrir_radius,
            hrir_length: self.hrir_length,
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
                .max(self.hrir_length - left_start)
                .max(self.hrir_length - right_start);

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
            hrir_length: max_len,
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
            hrir_length: self.hrir_length,
            hrir_sampling_rate: self.hrir_sampling_rate,
            left_ear_pos: self.left_ear_pos,
            right_ear_pos: self.right_ear_pos,
        }
    }
}

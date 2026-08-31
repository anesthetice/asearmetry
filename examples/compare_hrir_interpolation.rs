/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

use std::f64::consts::TAU;

use asearmetry::prelude::*;
use indoc::printdoc;
use itertools::Itertools;
use tap::{Pipe, Tap};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    const HRIR_FILEPATH_1: &str = "sofa_conversion/output/HATS051123_1_processed.hrir.asear";
    const HRIR_FILEPATH_2: &str = "sofa_conversion/output/KU100051023_1_processed.hrir.asear";
    const HRIR_FILEPATH_3: &str = "sofa_conversion/output/ZTV406081722_1_processed.hrir.asear";

    let binauralizer = pre_processing(
        BinauralizerPrecursor::load_from_file(HRIR_FILEPATH_1)?,
        true,
    );

    let binauralizer_unal = pre_processing(
        BinauralizerPrecursor::load_from_file(HRIR_FILEPATH_1)?,
        false,
    );

    [2_usize, 3, 4, 5, 6, 7, 8, 9]
        .into_iter()
        .for_each(|n_nearest| {
            interpolation_method_1(binauralizer.clone(), n_nearest)
                .pipe(|input| compute_score(input, &format!("Method 1: n_nearest = {n_nearest}")));
        });

    [2_usize, 3, 4, 5, 6, 7, 8, 9]
        .into_iter()
        .for_each(|n_nearest| {
            interpolation_method_1(binauralizer_unal.clone(), n_nearest).pipe(|input| {
                compute_score(
                    input,
                    &format!("Method 1 (no alignment): n_nearest = {n_nearest}"),
                )
            });
        });

    Ok(())
}

fn interpolation_method_1(
    binauralizer: Binauralizer,
    n_nearest: usize,
) -> Vec<(Shell2D, StereoAudioBuf, StereoAudioBuf)> {
    let Binauralizer {
        hrir_rtree,
        hrir_radius,
        hrir_length,
        hrir_sampling_rate,
        left_ear_pos,
        right_ear_pos,
    } = binauralizer;

    hrir_rtree
        .iter()
        .map(|gwd| (*gwd.geom(), gwd.data.clone()))
        .map(|(pos, original_hrir)| {
            let mut total_dist = 0.0;
            let mut dists = Vec::with_capacity(n_nearest);
            let mut hrirs = Vec::with_capacity(n_nearest);

            assert!(
                *hrir_rtree
                    .nearest_neighbor_iter(pos.into())
                    .next()
                    .unwrap()
                    .geom()
                    == pos
            );

            //println!("for pos: {pos:.2}");

            hrir_rtree
                .nearest_neighbor_iter(pos.into())
                .take(n_nearest + 1)
                .skip(1)
                .for_each(|gwd| {
                    assert!(*gwd.geom() != pos);
                    let dist = gwd.geom().dist_angular(pos);
                    /*
                    println!(
                        "• close to: {:.2}, with an angular dist of: {:.5}°",
                        gwd.geom(),
                        360.0 * dist / TAU
                    );
                    */
                    total_dist += dist;
                    dists.push(dist);
                    hrirs.push(gwd.data.view());
                });
            assert!(total_dist > 0.0);

            // Higher is better this time, unlike the angular distance.
            let scores = dists
                .into_iter()
                .map(|dist| total_dist - dist)
                .collect_vec();
            let total_score = scores.iter().sum::<f64>();

            let interp_hrir = hrirs.into_iter().zip(scores).fold(
                AudioBuffer::<2>::with_capacity(hrir_length, None),
                |new_hrir, (hrir, score)| {
                    let weight = (score / total_score) as f32;
                    new_hrir.merge_with(hrir.apply(|x| x * weight))
                },
            );

            (pos, original_hrir, interp_hrir)
        })
        .collect_vec()
}

fn compute_score(input: Vec<(Shell2D, StereoAudioBuf, StereoAudioBuf)>, title: &str) {
    let (scores, positions) = input
        .into_iter()
        .map(|(pos, hrir, hrir_interp)| {
            let score = hrir
                .apply_with(hrir_interp, |a, b| (a - b).powi(2))
                .channels
                .into_iter()
                .map(|cha| cha.into_iter().sum::<f32>())
                .sum::<f32>();
            (score, pos)
        })
        .multiunzip::<(Vec<f32>, Vec<Shell2D>)>();

    let total_score = scores.iter().sum::<f32>();
    let mean_score: f32 = mean(&scores);
    let std_score: f32 = std(&scores, 1);

    let (best_score, best_pos) = scores
        .iter()
        .zip_eq(&positions)
        .reduce(|a, b| if a.0 > b.0 { b } else { a })
        .unwrap();

    let (worst_score, worst_pos) = scores
        .iter()
        .zip_eq(&positions)
        .reduce(|a, b| if a.0 > b.0 { a } else { b })
        .unwrap();

    printdoc! {"
        {title}
        ├ total  : {total_score:.2}
        ├ mean   : {mean_score:.2}
        ├ std    : {std_score:.2}
        ├ best   : {best_score:.2} @ {best_pos}
        └ worst  : {worst_score:.2} @ {worst_pos}
    "};
}

fn pre_processing(mut bp: BinauralizerPrecursor, alignment_flag: bool) -> Binauralizer {
    if alignment_flag {
        bp.hrir_vec = bp
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
    }

    let hrir_rtree = rstar::RTree::bulk_load(
        bp.hrir_vec
            .into_iter()
            .zip(bp.hrir_pos_vec)
            .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
            .collect(),
    );

    Binauralizer {
        hrir_rtree,
        hrir_radius: bp.hrir_radius,
        hrir_length: bp.hrir_length,
        hrir_sampling_rate: bp.hrir_sampling_rate,
        left_ear_pos: bp.left_ear_pos,
        right_ear_pos: bp.right_ear_pos,
    }
}

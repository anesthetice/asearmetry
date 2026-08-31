/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(mixed_script_confusables)]
#![allow(unused)]

mod common;
use common::get_example_filepath;

use asearmetry::prelude::*;
use itertools::Itertools;
use num_complex::Complex32;
use petgraph::graph::NodeIndex;
use rand::RngExt;
use std::{f32::consts::PI as PI_F32, f64::consts::PI, ops::Add};
use tap::Tap;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let input = MonoAudioBuf::load_from_file("audio/sample_03.wav")?
        .slice_by_time(9.00, 29.00)
        .into_owned();

    //let input = MonoAudioBuf::sinusoidal(15.0, 48_000, 0.9, 300.0, 0.0);
    let input_sr = input.sampling_rate().unwrap();

    let binaur = BinauralizerPrecursor::load_from_file(
        "sofa_conversion/output/ZTV406081722_1_processed.hrir.asear",
    )?
    .into_binauralizer();

    println!(
        "input freq: {}; HRIR freq: {}",
        input_sr, binaur.hrir_sampling_rate
    );

    let duration = input.duration().unwrap();
    let trajectory = simple_circle_trajectory(duration);
    //let trajectory = change_direction_trajectory(duration);

    let out = binaur.run(&input, trajectory).normalize();

    println!("Writing to file");
    out.write_to_opus_file("audio/out.opus")?;

    Ok(())
}

fn simple_circle_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    Trajectory::from_equations(
        |t| Sphere3D::new(5.0, -t * PI / 8.0, PI / 2.0),
        duration,
        3.0 / 48_000.0,
    )
}

fn static_front_trajectory(duration: Seconds) -> Trajectory<Cart3D> {
    Trajectory::from_equations(|_| Cart3D::new(1.0, 0.0, 0.0), duration, 0.05)
}

fn change_direction_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    let δt: Seconds = 0.001;
    let mut start_time = 0.0;
    let mut pos = Sphere3D::new(0.9, 0.0, PI / 2.0);

    let mut rng = rand::rng();
    let distr = rand::distr::Uniform::new_inclusive(-1.0_f64, 1.0_f64).unwrap();
    let mut θ_av: Radians = 0.0;
    let mut φ_av: Radians = 0.0;

    Trajectory::from_equations(
        |t| {
            if start_time == 0.0 || t - start_time > 4.0 {
                start_time = t;
                θ_av = rng.sample(distr) * 3.0;
                φ_av = rng.sample(distr) * 0.6;
            }
            pos.θ += δt * θ_av;
            pos.φ += δt * φ_av;
            pos.clamp_angles_in_place();
            pos
        },
        duration,
        δt,
    )
}

fn change_direction_trajectory_cart(duration: Seconds) -> Trajectory<Cart3D> {
    let δt: f64 = 0.001;
    let mut start_time = 0.0;
    let mut pos = Cart3D::new(1.0, 0.0, 0.0);

    let mut rng = rand::rng();
    let distr = rand::distr::Uniform::new_inclusive(-1.0_f64, 1.0_f64).unwrap();

    let mut x_av: f64 = 0.0;
    let mut y_av: f64 = 0.0;
    let mut z_av: f64 = 0.0;

    Trajectory::from_equations(
        |t| {
            if start_time == 0.0 || t - start_time > 5.0 {
                start_time = t;
                x_av = rng.sample(distr) * 1.2;
                y_av = rng.sample(distr) * 1.2;
                z_av = rng.sample(distr) * 1.2;
            }
            pos.x += δt * x_av;
            pos.y += δt * y_av;
            pos.z += δt * z_av;
            pos
        },
        duration,
        δt,
    )
}

fn bespoke_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    Trajectory::from_equations(
        |t| {
            if t < 10.0 {
                let θ = if ((t * 2.0) as u32).is_multiple_of(2) {
                    -PI / 2.0
                } else {
                    PI / 2.0
                };
                Sphere3D::new(1.0, θ, PI / 2.0)
            } else {
                Sphere3D::new(1.0, 0.0, PI / 2.0)
            }
        },
        duration,
        1E-4,
    )
}

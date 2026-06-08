#![allow(mixed_script_confusables)]
#![allow(unused)]

// Modules
mod misc;

// Imports
use asearmetry::{
    audio::{
        AudioBuffer, AudioBufferSlice, AudioSignal, DiscreteSignal, MonoAudioBuf, MonoAudioBufSlice,
    },
    binaur::{Binauralizer, BinauralizerPrecursor},
    coordinates::{Cart3D, Shell2D, Sphere3D},
    math::{Radians, Seconds},
    trajectory::{self, Trajectory},
};
use itertools::Itertools;
use rand::RngExt;
use std::{f64::consts::PI, ops::Add};
use tap::Tap;

fn main() -> anyhow::Result<()> {
    asearmetry::scene::build()
        .trajectory(
            simple_circle_trajectory(10.0)
                .with_basis::<Cart3D>()
                .downsample(1.0 / 30.0),
        )
        .call();

    return Ok(());

    //return binaur_plot();

    let input = MonoAudioBuf::load_from_file("audio/sample_03.wav")?
        //.apply_low_pass_filter(14_000.0, 256)
        .slice_by_time(9.00, 30.00)
        .into_owned();

    //let input = MonoAudioBuf::sinusoidal(15.0, 48_000, 0.9, 300.0, 0.0);
    let input_sr = input.sampling_rate().unwrap();

    let binaur = BinauralizerPrecursor::load_from_file(
        "sofa_conversion/output/AKO536081622_1_processed.asear.hrtf.parquet",
    )?
    .into_binauralizer();

    println!(
        "input freq: {}; HRIR freq: {}",
        input_sr, binaur.hrir_sampling_rate
    );

    let duration = input.duration().unwrap();
    let trajectory = simple_circle_trajectory(duration);
    //let trajectory = change_direction_trajectory(duration);

    println!("a");
    //trajectory.plot(Some(1.5));
    println!("b");

    let out = binaur
        .run(&input, trajectory)
        .apply_low_pass_filter(12_000.0, 256)
        .normalize();

    println!("Writing to file");
    out.write_to_file("audio/out.wav")?;

    Ok(())
}

fn simple_circle_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    Trajectory::from_equations(
        |t| Sphere3D::new(0.5, -t * PI / 4.0, PI / 2.0),
        duration,
        1.0 / 1024.0,
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

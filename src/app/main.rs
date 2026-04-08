#![allow(mixed_script_confusables)]
#![allow(unused)]

use std::f32::consts::PI;

use asearmetry::{
    Seconds,
    audio::{AudioSignal, DiscreteSignal, MonoAudioBuf},
    brp::Binauralizer,
    coordinates::{Cart3D, Sphere3D},
    trajectory::Trajectory,
};
use rand::RngExt;

fn simple_circle_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    Trajectory::from_equations(
        |t| Sphere3D::new(1.0, -t * PI / 4.0, PI / 2.0),
        duration,
        0.005,
    )
}

fn static_front_trajectory(duration: Seconds) -> Trajectory<Cart3D> {
    Trajectory::from_equations(|_| Cart3D::new(1.0, 0.0, 0.0), duration, 0.05)
}

fn change_direction_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    let δt: f32 = 0.001;
    let mut start_time = 0.0;
    let mut pos = Sphere3D::new(0.7, 0.0, PI / 2.0);

    let mut rng = rand::rng();
    let distr = rand::distr::Uniform::new_inclusive(-1.0_f32, 1.0_f32).unwrap();
    let mut θ_av: f32 = 0.0;
    let mut φ_av: f32 = 0.0;

    Trajectory::from_equations(
        |t| {
            if start_time == 0.0 || t - start_time > 4.0 {
                start_time = t;
                θ_av = rng.sample(distr) * 3.0;
                φ_av = rng.sample(distr) * 0.6;
            }
            pos.θ += δt * θ_av;
            pos.φ = (pos.φ + δt * φ_av).clamp(PI / 6.0, 5.0 * PI / 6.0);
            pos.clamp_angles_in_place();
            pos
        },
        duration,
        δt,
    )
}

fn main() -> anyhow::Result<()> {
    let input = MonoAudioBuf::load_from_file("audio/sample_04.wav")?;
    //let input = MonoAudioBuf::sinusoidal(15.0, 48_000, 0.9, 300.0, 0.0);
    let input_sr = input.sampling_rate().unwrap();

    let binaur = Binauralizer::load_from_file("sofa_conversion/output/hrtf.parquet")?;

    println!(
        "input freq: {}; HRIR freq: {}",
        input_sr, binaur.hrir_sampling_rate
    );

    let duration = input.duration().unwrap();
    let trajectory = simple_circle_trajectory(duration);
    //let trajectory = change_direction_trajectory(duration);

    let out = binaur
        .run(&input, trajectory)
        //.merge_with(binaur.run(&input.delay(0.3), trajectory_2))
        .low_pass(4000.0)
        .normalize();

    println!("Writing to file");
    out.write_to_file("audio/out.wav")?;

    Ok(())
}

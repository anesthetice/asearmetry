#![allow(mixed_script_confusables)]
#![allow(unused)]

use std::f32::consts::PI;

use asearmetry::{
    Seconds,
    audio::{
        AudioBuffer, AudioBufferSlice, AudioSignal, DiscreteSignal, MonoAudioBuf, MonoAudioBufSlice,
    },
    brp::{Binauralizer, BinauralizerPrecursor},
    coordinates::{Cart3D, Sphere3D},
    trajectory::Trajectory,
};
use rand::RngExt;

fn simple_circle_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    Trajectory::from_equations(
        |t| Sphere3D::new(1.0 + (t / 4.0), -t * PI / 4.0, PI / 2.0),
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

fn binaur_plot() -> anyhow::Result<()> {
    let binaur_precursor =
        BinauralizerPrecursor::load_from_file("sofa_conversion/output/hrtf.parquet")?;

    let binaur_raw = binaur_precursor.clone().into_binauralizer_no_processing();
    let binaur_pro = binaur_precursor.clone().into_binauralizer();

    let pos_array = [
        Sphere3D::new(1.0, 0.0, PI / 2.0),
        Sphere3D::new(1.0, PI / 2.0, PI / 2.0),
        Sphere3D::new(1.0, -PI / 2.0, PI / 2.0),
    ];

    use kuva::prelude::*;

    let mut plots: Vec<Vec<Plot>> = Vec::new();
    let mut layouts: Vec<Layout> = Vec::new();
    for pos in pos_array {
        let raw_plot = binaur_raw.get_hrir(pos).slice(0..80).plot();
        let raw_layout = Layout::auto_from_plots(&raw_plot).with_title(format!("Raw @ {pos}"));
        plots.push(raw_plot);
        layouts.push(raw_layout);

        let pro_plot = binaur_pro.get_hrir(pos).slice(0..80).plot();
        let pro_layout = Layout::auto_from_plots(&pro_plot).with_title(format!("Pro @ {pos}"));
        plots.push(pro_plot);
        layouts.push(pro_layout);

        let hrir = binaur_raw.get_hrir(pos);
        let smooth_plot = hrir
            .abs()
            .apply_median_filter(4)
            .apply_gaussian_filter(4, 0.33)
            .normalize_to(hrir.get_abs_max())
            .slice(0..80)
            .plot();
        let smooth_layout =
            Layout::auto_from_plots(&smooth_plot).with_title(format!("Smooth @ {pos}"));
        plots.push(smooth_plot);
        layouts.push(smooth_layout);
    }

    let scene = Figure::new(pos_array.len(), 3)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    let svg = SvgBackend.render_scene(&scene);
    std::fs::write("figure.svg", svg).unwrap();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    return binaur_plot();

    let input = MonoAudioBuf::load_from_file("audio/sample_bespoke.wav")?
        .apply_low_pass_filter(14_000.0, 256);
    //let input = MonoAudioBuf::sinusoidal(15.0, 48_000, 0.9, 300.0, 0.0);
    let input_sr = input.sampling_rate().unwrap();

    let binaur = BinauralizerPrecursor::load_from_file("sofa_conversion/output/hrtf.parquet")?
        .into_binauralizer();

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
        .normalize();

    println!("Writing to file");
    out.write_to_file("audio/out.wav")?;

    Ok(())
}

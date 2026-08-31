/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

mod common;
use common::{get_example_filepath, save_scene};

use asearmetry::prelude::*;
use itertools::Itertools;
use std::f64::consts::PI;

pub fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    #[cfg(feature = "plot")]
    inner()?;

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");

    Ok(())
}

#[cfg(feature = "plot")]
fn inner() -> anyhow::Result<()> {
    use kuva::prelude::*;

    let input = MonoAudioBuf::merge_many(
        [
            MonoAudioBuf::sinusoidal(10.0, 48_000.0, 1.0, 400.0, 0.0),
            //MonoAudioBuf::sinusoidal(10.0, 48_000.0, 1.0, 500.0, 0.0),
            //MonoAudioBuf::sinusoidal(10.0, 48_000.0, 1.0, 600.0, 0.0),
        ],
        None,
    );

    //let input = MonoAudioBuf::sinusoidal(15.0, 48_000, 0.9, 300.0, 0.0);
    let input_sr = input.sampling_rate().unwrap();

    let binaur = BinauralizerPrecursor::load_from_file(
        "sofa_conversion/output/AKO536081622_1_processed.asear.hrtf.parquet",
    )?
    .into_binauralizer();

    let duration = input.duration().unwrap();
    let trajectory = simple_circle_trajectory(duration);

    let output = binaur
        .run(&input, trajectory)
        //.apply_low_pass_filter(12_000.0, 256)
        .normalize();

    indoc::printdoc! {"
        Start of the input audio signal
        {}

        Start of the processed audio signal
        {}

        DFT of the input audio signal
        {}

        DFT of the processed audio signal
        {}
    ", input.slice(0..300), output.slice(0..300), "a", "b"
    };

    #[rustfmt::skip]
    let (plots, layouts): (Vec<Vec<Plot>>, Vec<Layout>) = [
        input.slice(0..300).plot_builder()
            .layout_title("Start of the input audio signal").plot_stroke_width(1.2)
            .build().plot_and_layout(),

        input.view().dft().complex_norm().plot_builder()
            .layout_title("DFT of the input audio signal").plot_stroke_width(1.2)
            .build().plot_and_layout(),

        output.slice(0..300).plot_builder()
            .layout_title("Start of the processed audio signal").plot_stroke_width(1.2)
            .build().plot_and_layout(),

        output.view().dft().complex_norm().plot_builder()
            .layout_title("DFT of the processed audio signal").plot_stroke_width(1.2)
            .build().plot_and_layout(),
    ]
    .into_iter()
    .multiunzip();

    let scene = Figure::new(2, 2)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    let svg = SvgBackend.render_scene(&scene);
    std::fs::write("check_binaurlizer.svg", svg)?;
    println!("Wrote plot to 'check_binaurlizer.svg'");

    input
        .normalize()
        .into_stereo()
        .write_to_wav_file("binaur_in.wav")?;
    println!("Wrote binauralization input to 'binaur_in.wav'");

    output.write_to_wav_file("binaur_out.wav")?;
    println!("Wrote binauralization output to 'binaur_out.wav'");

    Ok(())
}

fn simple_circle_trajectory(duration: Seconds) -> Trajectory<Sphere3D> {
    Trajectory::from_equations(
        |t| Sphere3D::new(0.5, -t * PI / 4.0, PI / 2.0),
        duration,
        1.0 / 1024.0,
    )
}

/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

mod common;
use common::get_example_filepath;

use asearmetry::prelude::*;
use itertools::Itertools;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    #[cfg(feature = "plot")]
    check_resampling();

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");
}

#[cfg(feature = "plot")]
pub fn check_resampling() {
    use asearmetry::signal::DSP;
    use kuva::prelude::*;

    let s1 = MonoAudioBuf::merge_many(
        [
            MonoAudioBuf::sinusoidal(5.0, 2000.0, 1.0, 2.0, 0.0),
            MonoAudioBuf::sinusoidal(5.0, 2000.0, 0.5, 4.1, 1.0),
            MonoAudioBuf::sinusoidal(5.0, 2000.0, 0.1, 9.8, 1.0),
            MonoAudioBuf::sinusoidal(5.0, 2000.0, 4.0, 0.2, 0.5),
        ],
        None,
    )
    .normalize();

    let sr1_3 = s1.resample(210.0, 3);
    let sr1_7 = s1.resample(210.0, 7);
    let sr1_17 = s1.resample(210.0, 17);
    let sr1_pure = s1.resample_pure(210.0);

    let s2 = MonoAudioBuf::new_mono(
        [
            vec![0.0_f32; 200].as_slice(),
            vec![1.0_f32; 100].as_slice(),
            vec![0.0_f32; 200].as_slice(),
        ]
        .concat(),
        Some(100.0),
    );

    let sr2_3 = s2.resample(250.0, 3);
    let sr2_7 = s2.resample(250.0, 7);
    let sr2_17 = s2.resample(250.0, 17);
    let sr2_pure = s2.resample_pure(250.0);

    #[rustfmt::skip]
    let plots = vec![
        s1.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr1_3.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr1_7.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr1_17.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr1_pure.plot_builder().plot_stroke_width(2.5).build().plot(),
        s2.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr2_3.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr2_7.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr2_17.plot_builder().plot_stroke_width(2.5).build().plot(),
        sr2_pure.plot_builder().plot_stroke_width(2.5).build().plot(),
    ];

    let layouts = plots
        .iter()
        .enumerate()
        .map(|(i, plot)| {
            let title = match i {
                0 => "original signal: f₁ = 2000 Hz",
                1 => "resampled signal: f₂ = 210 Hz, M=3",
                2 => "resampled signal: f₂ = 210 Hz, M=7",
                3 => "resampled signal: f₂ = 210 Hz, M=17",
                4 => "resampled signal: f₂ = 210 Hz, M→+∞",
                //
                5 => "original signal: f₁ = 100 Hz",
                6 => "resampled signal: f₂ = 250 Hz, M=3",
                7 => "resampled signal: f₂ = 250 Hz, M=7",
                8 => "resampled signal: f₂ = 250 Hz, M=17",
                9 => "resampled signal: f₂ = 250 Hz, M→+∞",
                //
                _ => unreachable!(),
            };
            let (y_axis_min, y_axis_max) = if i < 5 { (-1.0, 1.0) } else { (0.0, 1.5) };
            Layout::auto_from_plots(plot)
                .with_x_axis_min(0.0)
                .with_x_axis_max(5.0)
                .with_y_axis_min(y_axis_min)
                .with_y_axis_max(y_axis_max)
                .with_title(title)
        })
        .collect_vec();

    let scene = Figure::new(2, 5)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    let svg = SvgBackend.render_scene(&scene);
    std::fs::write("resampling.svg", svg).unwrap();
    println!("Wrote output to 'resampling.svg'");
}

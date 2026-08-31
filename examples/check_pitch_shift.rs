/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

mod common;
use common::get_example_filepath;

use asearmetry::prelude::*;
use itertools::Itertools;
use std::f64::consts::PI;

#[allow(non_snake_case)]
pub fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    //let x = MonoAudioBuf::sinusoidal(10.0, 48_000.0, 1.0, 500.0, 0.0);

    let x = MonoAudioBuf::load_from_file("./audio/sample_03.wav")?
        .slice_by_time(0.0, 30.0)
        .into_owned();

    return Ok(());
    let H: usize = 51;
    let W: usize = 1001;
    let W_f: usize = W / 2 + 1;

    let (𝒳, mut 𝒳_info) = x.stft(H, W);

    let K = 𝒳.len() / W_f;
    let N_padded = W + (K - 1) * H;

    𝒳_info.hop_length = 52;

    let y = 𝒳.istft(𝒳_info);

    y.view()
        .dft()
        .complex_norm()
        .resample(5_000.0, 51)
        .plot_builder()
        .build()
        .render_to_file(get_example_filepath("guh.svg")?)?;

    y //.modify_speed(1.3)
        .apply_highpass_filter(250.0, 60)
        .normalize()
        .into_stereo()
        .write_to_wav_file("./audio/goofy_stft.wav")?;

    /*
    𝒳.as_blocks(W_f)
        .map(|frame| {
            let [s] = frame.index(W_f - 2);
            s.arg()
        })
        .tuple_windows()
        .for_each(|(s_abs_1, s_abs_2)| {
            println!("{}", s_abs_2 - s_abs_1);
        });
    */

    Ok(())
}

pub fn main2() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    #[cfg(feature = "plot")]
    inner()?;

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");

    Ok(())
}

#[cfg(feature = "plot")]
pub fn inner() -> anyhow::Result<()> {
    use asearmetry::prelude::*;
    use kuva::prelude::*;

    const DURATION: Seconds = 0.02;
    const SAMPLING_RATE: Hertz = 48_000.0;
    const AMPLITUDE: f64 = 1.0;

    let mut plots: Vec<Vec<Plot>> = Vec::new();
    let mut layouts: Vec<Layout> = Vec::new();

    [
        (
            "before shift-pitch",
            MonoAudioBuf::sinusoidal(DURATION, SAMPLING_RATE, AMPLITUDE, 100.0, 0.0),
        ),
        (
            "after shift-pitch",
            MonoAudioBuf::sinusoidal(DURATION, SAMPLING_RATE, AMPLITUDE, 100.0, 0.0)
                .shift_pitch(-50.0),
        ),
        (
            "desired",
            MonoAudioBuf::sinusoidal(DURATION, SAMPLING_RATE, AMPLITUDE, 50.0, 0.0),
        ),
    ]
    .into_iter()
    .for_each(|(s, sig)| {
        let (t_plot, t_layout) = sig
            .plot_builder()
            .x_axis_use_samples(true)
            .plot_stroke_width(1.5)
            .layout_title(s)
            .build()
            .plot_and_layout();
        plots.push(t_plot);
        layouts.push(t_layout);

        let (f_plot, f_layout) = sig
            .dft()
            .complex_norm()
            .clamp_to(5.0)
            .plot_builder()
            .plot_stroke_width(1.5)
            .layout_extra(|la| la.with_x_axis_min(0.0).with_x_axis_max(1000.0))
            .build()
            .plot_and_layout();
        plots.push(f_plot);
        layouts.push(f_layout);
    });

    let scene = Figure::new(3, 2)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    let svg = SvgBackend.render_scene(&scene);
    std::fs::write(get_example_filepath("thing.svg")?, svg)?;

    Ok(())
}

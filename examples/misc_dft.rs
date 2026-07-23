/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

use asearmetry::signal::{DSP, FreqDomain, Signal};
use num_complex::Complex32;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "plot")]
    inner()?;

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");

    Ok(())
}

#[allow(non_snake_case, non_upper_case_globals)]
#[cfg(feature = "plot")]
fn inner() -> anyhow::Result<()> {
    use asearmetry::{
        audio::ASP,
        math::{Hertz, Seconds},
    };
    use kuva::prelude::*;
    use rand::{RngExt, SeedableRng};
    use std::f32::consts::TAU;
    use tap::Pipe;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Let's try to create 5 seconds of white noise (audible frequencies),
    // before applying a low pass filter to see the effects on a plot.
    const duration: Seconds = 5.0;
    const sampling_rate: Hertz = 44_100.0;
    const N_t: usize = (sampling_rate * duration) as usize;
    const N_f: usize = (N_t / 2) + 1;
    const Δf: Hertz = sampling_rate / N_t as f64;
    const f_cutoff: Hertz = 5_000.0;
    const RNG_seed: u64 = 2000;

    let freq_samples_halved = {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(RNG_seed);
        let N_f_nonzero_start = (20.0 / Δf).ceil() as usize;
        let N_f_nonzero_end = (20_000.0 / Δf).ceil() as usize;

        let mut spectrum = vec![Complex32::ZERO; N_f];
        // Note that spectrum[0] is already 0 (→ mean of zero which is what we want).

        for z in &mut spectrum[N_f_nonzero_start..=N_f_nonzero_end] {
            let phase = rng.random_range(0.0..TAU);
            *z = Complex32::from_polar(1.0, phase);
        }

        spectrum
    };

    let white_noise_fs = Signal::new_from_halved([freq_samples_halved], Some(sampling_rate), N_t);

    let mut plots: Vec<Vec<Plot>> = Vec::new();
    let mut layouts: Vec<Layout> = Vec::new();

    let white_noise_ts = white_noise_fs.idft();

    white_noise_fs
        .view()
        .complex_norm()
        .plot_builder()
        .build()
        .plot_and_layout()
        .pipe(|(p, l)| {
            plots.push(p);
            layouts.push(l);
        });

    white_noise_ts
        .plot_builder()
        .build()
        .plot_and_layout()
        .pipe(|(p, l)| {
            plots.push(p);
            layouts.push(l);
        });

    let lowpass_noise_ts = white_noise_ts.apply_low_pass_filter(f_cutoff as f32, 25);
    let lowpass_noise_fs = lowpass_noise_ts.dft();

    lowpass_noise_fs
        .view()
        .complex_norm()
        .plot_builder()
        .build()
        .plot_and_layout()
        .pipe(|(p, l)| {
            plots.push(p);
            layouts.push(l);
        });

    lowpass_noise_ts
        .plot_builder()
        .build()
        .plot_and_layout()
        .pipe(|(p, l)| {
            plots.push(p);
            layouts.push(l);
        });

    let scene = Figure::new(2, 2)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    let svg = SvgBackend.render_scene(&scene);
    std::fs::write("misc_dft.svg", svg)?;
    println!("Wrote output to 'misc_dft.svg'");

    white_noise_ts
        .normalize()
        .into_stereo()
        .write_to_file("white_noise.wav")?;

    lowpass_noise_ts
        .normalize()
        .into_stereo()
        .write_to_file("white_noise_lowpassed.wav")?;

    println!("Wrote output to 'white_noise.wav' and 'white_noise_lowpassed.wav'");

    Ok(())
}

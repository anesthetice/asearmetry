/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

mod common;
use common::{get_example_filepath, save_scene};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    #[cfg(feature = "plot")]
    inner()?;

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");

    Ok(())
}

#[allow(non_snake_case, non_upper_case_globals)]
#[cfg(feature = "plot")]
fn inner() -> anyhow::Result<()> {
    use asearmetry::prelude::*;
    use kuva::prelude::*;
    use rand::SeedableRng;
    use tap::Pipe;

    const f_cutoff_for_lowpass_example: Hertz = 2000.0;
    const f_cutoff_for_highpass_example: Hertz = 4000.0;
    const f_cutoffs_for_bandpass_example: (Hertz, Hertz) = (2000.0, 4000.0);

    // Let's create 3 seconds of white noise
    const duration: Seconds = 3.0;
    const sampling_rate: Hertz = 12_000.0;
    const RNG_SEED: u64 = 2000;
    const M: usize = 51;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(RNG_SEED);

    let whitenoise_td =
        MonoAudioBuf::noise(duration, sampling_rate, (0.0, 12_000.0), 0.1, &mut rng);
    let whitenoise_fd = whitenoise_td.clone().dft();

    let (whitenoise_lp_td, whitenoise_lp_fd) = whitenoise_td
        .apply_lowpass_filter(f_cutoff_for_lowpass_example, M)
        .pipe(|filtered_td| (filtered_td.clone(), filtered_td.dft()));

    let (whitenoise_hp_td, whitenoise_hp_fd) = whitenoise_td
        .apply_highpass_filter(f_cutoff_for_highpass_example, M)
        .pipe(|filtered_td| (filtered_td.clone(), filtered_td.dft()));

    let (whitenoise_bp_td, whitenoise_bp_fd) = whitenoise_td
        .apply_bandpass_filter(f_cutoffs_for_bandpass_example, M)
        .pipe(|filtered_td| (filtered_td.clone(), filtered_td.dft()));

    let mut plots: Vec<Vec<Plot>> = Vec::new();
    let mut layouts: Vec<Layout> = Vec::new();

    #[rustfmt::skip]
    [
        (whitenoise_fd, whitenoise_td, "White noise at audible frequencies".to_string()),
        (whitenoise_lp_fd, whitenoise_lp_td, format!("Lowpass with f <= {f_cutoff_for_lowpass_example} [Hz]")),
        (whitenoise_hp_fd, whitenoise_hp_td, format!("Highpass with f >= {f_cutoff_for_highpass_example} [Hz]")),
        (whitenoise_bp_fd, whitenoise_bp_td, format!("Bandpass with f ∈ [{}, {}] [Hz]", f_cutoffs_for_bandpass_example.0, f_cutoffs_for_bandpass_example.1)),
    ]
        .into_iter()
        .for_each(|(noise_fd, noise_td, title)| {
            noise_fd
                .complex_norm()
                .plot_builder()
                .layout_title(title)
                .build()
                .plot_and_layout_to(&mut plots, &mut layouts);

            noise_td
                .plot_builder()
                .layout_title("Equivalent signal in the time domain")
                .build()
                .plot_and_layout_to(&mut plots, &mut layouts);
        });

    let scene = Figure::new(4, 2)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    save_scene(scene, get_example_filepath("check_filters.svg")?)?;

    /*
    white_noise_ts
        .normalize()
        .into_stereo()
        .write_to_file(get_example_filepath("white_noise.wav")?)?;

    lowpass_noise_ts
        .normalize()
        .into_stereo()
        .write_to_file(get_example_filepath("white_noise_lowpassed.wav")?)?;

    println!("Wrote output to 'white_noise.wav' and 'white_noise_lowpassed.wav'");
    */

    Ok(())
}

use std::f64::consts::PI;

use asearmetry::{
    audio::{AudioSignal, DiscreteSignal, MonoAudioBuf},
    binaur::BinauralizerPrecursor,
    coordinates::Sphere3D,
};
use itertools::Itertools;

#[cfg(feature = "plot")]
pub fn check_resampling() {
    use kuva::prelude::*;

    let s1 = MonoAudioBuf::merge_many([
        MonoAudioBuf::sinusoidal(5.0, 2000.0, 1.0, 2.0, 0.0),
        MonoAudioBuf::sinusoidal(5.0, 2000.0, 0.5, 4.1, 1.0),
        MonoAudioBuf::sinusoidal(5.0, 2000.0, 0.1, 9.8, 1.0),
        MonoAudioBuf::sinusoidal(5.0, 2000.0, 4.0, 0.2, 0.5),
    ])
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

    let plots = vec![
        s1.plot(Some(2.5)),
        sr1_3.plot(Some(2.5)),
        sr1_7.plot(Some(2.5)),
        sr1_17.plot(Some(2.5)),
        sr1_pure.plot(Some(2.5)),
        s2.plot(Some(2.5)),
        sr2_3.plot(Some(2.5)),
        sr2_7.plot(Some(2.5)),
        sr2_17.plot(Some(2.5)),
        sr2_pure.plot(Some(2.5)),
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
                4 => "resampled signal: f₂ = 210 Hz, M=+∞",
                //
                5 => "original signal: f₁ = 100 Hz",
                6 => "resampled signal: f₂ = 250 Hz, M=3",
                7 => "resampled signal: f₂ = 250 Hz, M=7",
                8 => "resampled signal: f₂ = 250 Hz, M=17",
                9 => "resampled signal: f₂ = 250 Hz, M=+∞",
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
    println!("Wrote to 'resampling.svg'");
}

pub fn binaur_plot() -> anyhow::Result<()> {
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
        let raw_plot = binaur_raw.get_hrir_both(pos).slice(0..80).plot(Some(1.5));
        let raw_layout = Layout::auto_from_plots(&raw_plot).with_title(format!("Raw @ {pos}"));
        plots.push(raw_plot);
        layouts.push(raw_layout);

        let pro_plot = binaur_pro.get_hrir_both(pos).slice(0..80).plot(Some(1.5));
        let pro_layout = Layout::auto_from_plots(&pro_plot).with_title(format!("Pro @ {pos}"));
        plots.push(pro_plot);
        layouts.push(pro_layout);

        let hrir = binaur_raw.get_hrir_both(pos);
        let smooth_plot = hrir
            .abs()
            .apply_gaussian_filter(10, 0.05)
            .normalize_to(hrir.get_abs_max())
            .slice(0..80)
            .plot(Some(1.5));
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

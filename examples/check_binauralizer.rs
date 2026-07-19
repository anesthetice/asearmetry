/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

use asearmetry::{audio::ASP, coordinates::Sphere3D, signal::DSP};
use std::f64::consts::PI;

pub fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "plot", feature = "polars"))]
    binaur_plot()?;

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");

    Ok(())
}

#[cfg(all(feature = "plot", feature = "polars"))]
pub fn binaur_plot() -> anyhow::Result<()> {
    use asearmetry::binaur::BinauralizerPrecursor;
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
        let (raw_plot, raw_layout) = binaur_raw
            .get_hrir_both(pos)
            .slice(0..80)
            .plot_builder()
            .plot_line_stroke_width(1.5)
            .layout_title(format!("Raw @ {pos}"))
            .build()
            .plot_and_layout();
        plots.push(raw_plot);
        layouts.push(raw_layout);

        let (pro_plot, pro_layout) = binaur_pro
            .get_hrir_both(pos)
            .slice(0..80)
            .plot_builder()
            .plot_line_stroke_width(1.5)
            .layout_title(format!("Pro @ {pos}"))
            .build()
            .plot_and_layout();
        plots.push(pro_plot);
        layouts.push(pro_layout);

        let hrir = binaur_raw.get_hrir_both(pos);
        let (smooth_plot, smooth_layout) = hrir
            .abs()
            .apply_gaussian_filter(10, 0.05)
            .normalize_to(hrir.get_abs_max())
            .slice(0..80)
            .plot_builder()
            .plot_line_stroke_width(1.5)
            .layout_title(format!("Smooth @ {pos}"))
            .build()
            .plot_and_layout();
        plots.push(smooth_plot);
        layouts.push(smooth_layout);
    }

    let scene = Figure::new(pos_array.len(), 3)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_shared_legend_bottom()
        .render();

    let svg = SvgBackend.render_scene(&scene);
    std::fs::write("check_binaurlizer.svg", svg)?;
    println!("Wrote output to 'check_binaurlizer.svg'");

    Ok(())
}

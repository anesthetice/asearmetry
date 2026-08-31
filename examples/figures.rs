/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

mod common;
use common::{get_figure_filepath, save_scene};

use asearmetry::signal::FreqSignal;
use tap::Tap;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    #[cfg(feature = "plot")]
    inner()?;

    #[cfg(not(feature = "plot"))]
    println!("\nThe feature `plot` must be enabled to run this example");

    Ok(())
}

fn inner() -> anyhow::Result<()> {
    use anyhow::anyhow;
    use asearmetry::prelude::*;
    use itertools::Itertools;
    use kuva::prelude::*;
    use std::f64::consts::PI;
    use tap::Pipe;

    const HRTF_FILEPATH: &str = "sofa_conversion/output/HATS051123_1_processed.hrir.asear";

    let binaur_precursor = BinauralizerPrecursor::load_from_file(HRTF_FILEPATH)?;
    let hrir_length = binaur_precursor.hrir_length;
    let raw_hrir_rtree = rstar::RTree::bulk_load(
        binaur_precursor
            .hrir_vec
            .clone()
            .into_iter()
            .zip(binaur_precursor.hrir_pos_vec.clone())
            .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
            .collect(),
    );

    // figure 1
    {
        let (plots, layouts) = [
            (Shell2D::new(0.0, PI / 2.0), "Straight ahead HRIR"),
            (Shell2D::new(PI, PI / 2.0), "Straight behind HRIR"),
            (Shell2D::new(PI / 2.0, PI / 2.0), "Straight left HRIR"),
            (Shell2D::new(-PI / 2.0, PI / 2.0), "Straight right HRIR"),
        ]
        .into_iter()
        .map(|(query, title)| {
            raw_hrir_rtree
                .nearest_neighbor(query.into())
                .ok_or_else(|| anyhow!("No HRIR found for {query}"))
                .unwrap()
                .data
                .slice(0..100)
                .plot_builder()
                .x_axis_use_samples(true)
                .plot_stroke_width(1.5)
                .layout_title(title)
                .build()
                .plot_and_layout()
        })
        .multiunzip::<(Vec<Vec<Plot>>, Vec<Layout>)>();

        let scene = Figure::new(2, 2)
            .with_plots(plots)
            .with_layouts(layouts)
            .with_shared_legend_bottom()
            .render();

        save_scene(scene, get_figure_filepath("1.svg")?)?;
    }

    // figure 2 - removing ITDs
    {
        const CENTER_INDEX: isize = 30;

        let (plots, layouts) = [
            (Shell2D::new(0.0, PI / 2.0), "Straight ahead HRIR"),
            (Shell2D::new(PI, PI / 2.0), "Straight behind HRIR"),
            (Shell2D::new(PI / 2.0, PI / 2.0), "Straight left HRIR"),
            (Shell2D::new(-PI / 2.0, PI / 2.0), "Straight right HRIR"),
        ]
        .into_iter()
        .flat_map(|(query, title)| {
            let raw_hrir = raw_hrir_rtree
                .nearest_neighbor(query.into())
                .ok_or_else(|| anyhow!("No HRIR found for {query}"))
                .unwrap()
                .data
                .view();

            let raw_hrir_spike_smooth = raw_hrir
                .abs()
                .apply_gaussian_filter(10, 0.05)
                .normalize_to(raw_hrir.get_abs_max());

            // We align both peaks to the currently arbitrary position of `CENTER_INDEX`
            let left_peak_idx = raw_hrir_spike_smooth.index_cha(0).get_abs_max_index() as isize;
            let right_peak_idx = raw_hrir_spike_smooth.index_cha(1).get_abs_max_index() as isize;

            let pro_hrir = raw_hrir.shift(
                [CENTER_INDEX - left_peak_idx, CENTER_INDEX - right_peak_idx],
                [0.0; 2],
            );

            let pro_hrir_spike_smooth = raw_hrir_spike_smooth.clone().shift(
                [CENTER_INDEX - left_peak_idx, CENTER_INDEX - right_peak_idx],
                [0.0; 2],
            );

            [
                (raw_hrir, raw_hrir_spike_smooth.view(), "with ITD"),
                (pro_hrir.view(), pro_hrir_spike_smooth.view(), "without ITD"),
            ]
            .into_iter()
            .map(|(a, b, s)| {
                let title = format!("{title} {s}");

                let a = a.slice(0..100);
                let b = b.slice(0..100);

                let mut plot = a
                    .plot_builder()
                    .x_axis_use_samples(true)
                    .plot_stroke_width(1.5)
                    .build()
                    .plot();

                plot.extend(
                    b.plot_builder()
                        .plot_extra_lp(|lp| lp.with_dashed())
                        .x_axis_use_samples(true)
                        .plot_stroke_width(1.5)
                        .build()
                        .plot(),
                );

                (plot, title)
            })
            .collect_vec()
        })
        .map(|(plot, title)| {
            let layout = Layout::auto_from_plots(&plot)
                .with_title(title)
                .with_reference_line(
                    ReferenceLine::vertical(CENTER_INDEX as f64)
                        .with_color("grey")
                        .with_label("desired alignment"),
                );
            (plot, layout)
        })
        .multiunzip::<(Vec<Vec<Plot>>, Vec<Layout>)>();

        let scene = Figure::new(4, 2)
            .with_plots(plots)
            .with_layouts(layouts)
            .with_shared_legend_bottom()
            .render();

        save_scene(scene, get_figure_filepath("2.svg")?)?;
    }

    // figure 3 - HRIR interpolation
    {
        const N_NEAREST: usize = 3;

        let hrir_vec_noitd = binaur_precursor
            .hrir_vec
            .clone()
            .into_iter()
            .map(|hrir| {
                let hrir_spike_smooth = hrir
                    .view()
                    .abs()
                    .apply_gaussian_filter(10, 0.05)
                    .normalize_to(hrir.get_abs_max());

                const CENTER_INDEX: isize = 30;

                // We align both peaks to the currently arbitrary position of `CENTER_INDEX`
                let left_peak_idx = hrir_spike_smooth.index_cha(0).get_abs_max_index() as isize;
                let right_peak_idx = hrir_spike_smooth.index_cha(1).get_abs_max_index() as isize;

                hrir.shift(
                    [CENTER_INDEX - left_peak_idx, CENTER_INDEX - right_peak_idx],
                    [0.0; 2],
                )
                // TODO: maybe apply window function here as well to avoid any discontinuities?
            })
            .collect_vec();

        let hrir_rtree_noitd = rstar::RTree::bulk_load(
            hrir_vec_noitd
                .into_iter()
                .zip(binaur_precursor.hrir_pos_vec.clone())
                .map(|(hrir, pos)| HrirProjection::new(pos, hrir))
                .collect(),
        );

        let (plots, layouts) = [
            (Shell2D::new(0.0, PI / 2.0), "Straight ahead HRIR"),
            (Shell2D::new(PI, PI / 2.0), "Straight behind HRIR"),
            (Shell2D::new(PI / 2.0, PI / 2.0), "Straight left HRIR"),
            (Shell2D::new(-PI / 2.0, PI / 2.0), "Straight right HRIR"),
        ]
        .into_iter()
        .map(|(query, title)| {
            let (pos, actual_hrir) = hrir_rtree_noitd
                .nearest_neighbor(query.into())
                .ok_or_else(|| anyhow!("No HRIR found for {query}"))
                .unwrap()
                .pipe(|proj| (*proj.geom(), proj.data.slice(0..100)));

            //println!("desired position: {}", new_pos);
            let mut total_dist = 0.0;
            let mut dists = Vec::with_capacity(N_NEAREST);
            let mut hrirs = Vec::with_capacity(N_NEAREST);

            hrir_rtree_noitd
                .nearest_neighbor_iter(pos.into())
                .skip(1)
                .take(N_NEAREST)
                .for_each(|gwd| {
                    let dist = gwd.geom().dist_angular(pos);
                    //println!("• close to: {}, with a dist of: {}", gwd.geom(), dist);
                    total_dist += dist;
                    dists.push(dist);
                    hrirs.push(gwd.data.view());
                });

            assert!(total_dist > 0.0);

            let interp_hrir = hrirs.into_iter().zip(dists).fold(
                StereoAudioBuf::with_capacity(hrir_length, None),
                |new_hrir, (hrir, dist)| {
                    let weight = (dist / total_dist) as f32;
                    new_hrir.merge_with(hrir.apply(|x| x * weight))
                },
            );

            let mut plot = actual_hrir
                .plot_builder()
                .x_axis_use_samples(true)
                .plot_stroke_width(1.5)
                .build()
                .plot();

            plot.extend(
                interp_hrir
                    .slice(0..100)
                    .plot_builder()
                    .plot_extra_lp(|lp| lp.with_dotted())
                    .x_axis_use_samples(true)
                    .plot_stroke_width(1.5)
                    .build()
                    .plot(),
            );

            let layout =
                Layout::auto_from_plots(&plot).with_title(format!("{title} interpolation"));

            (plot, layout)
        })
        .multiunzip::<(Vec<Vec<Plot>>, Vec<Layout>)>();

        let scene = Figure::new(2, 2)
            .with_plots(plots)
            .with_layouts(layouts)
            .with_shared_legend_bottom()
            .render();

        save_scene(scene, get_figure_filepath("3.svg")?)?;
    }

    // Pitch-scaling figures
    {
        let mut plots: Vec<Vec<Plot>> = Vec::new();
        let mut layouts: Vec<Layout> = Vec::new();

        let sr_usize: usize = 2000;
        let sr_f64: Hertz = sr_usize as f64;
        let frequency: Hertz = 100.0;

        let duration = 0.5;
        let n_samples_time = 2000 / 2;
        let n_samples_freq = n_samples_time / 2 + 1;
        MonoAudioBuf::sinusoidal(duration, sr_f64, 1.0, frequency, 0.0)
            .tap(|sig| {
                sig.plot_builder()
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            })
            .pipe(DSP::dft_halved)
            .tap(|sig| {
                sig.complex_norm()
                    .plot_builder()
                    .dft_is_halved(n_samples_time)
                    .layout_extra(|l| l.with_x_axis_min(0.0).with_x_axis_max(150.0))
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            })
            .pipe(|sig| {
                sig.resample_pure(sr_f64 / 2.0)
                    .tap_mut(|sig| {
                        sig.set(
                            0,
                            [sig.index(0)[0].norm().pipe(|norm| cf32::new(norm, 0.0))],
                        )
                    })
                    .with_sr(sr_f64)
                    .pad_right_to_len(n_samples_freq)
            })
            .tap(|sig| {
                sig.complex_norm()
                    .plot_builder()
                    .dft_is_halved(n_samples_time)
                    .layout_extra(|l| l.with_x_axis_min(0.0).with_x_axis_max(150.0))
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            })
            .pipe(|sig| sig.idft_halved(n_samples_time))
            .tap(|sig| {
                sig.plot_builder()
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            });

        let duration = 1.0;
        let n_samples_time = 2000;
        let n_samples_freq = n_samples_time / 2 + 1;
        MonoAudioBuf::sinusoidal(duration / 2.0, sr_f64, 1.0, frequency, 0.0)
            .pad_right(1000)
            .tap(|sig| {
                sig.plot_builder()
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            })
            .pipe(DSP::dft_halved)
            .tap(|sig| {
                sig.complex_norm()
                    .plot_builder()
                    .dft_is_halved(n_samples_time)
                    .layout_extra(|l| l.with_x_axis_min(0.0).with_x_axis_max(150.0))
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            })
            .pipe(|sig| {
                let re = sig.clone().apply_into(|v| v.re).resample(sr_f64 / 2.0, 101);
                let im = sig
                    .apply_into(|v| v.im)
                    .resample(sr_f64 / 2.0, 101)
                    .apply(|_| 0.0);

                re.apply_with(im, cf32::new)
                    .tap_mut(|sig| {
                        sig.set(
                            0,
                            [sig.index(0)[0].norm().pipe(|norm| cf32::new(norm, 0.0))],
                        )
                    })
                    .with_sr(sr_f64)
                    .pad_right_to_len(n_samples_freq)
            })
            .tap(|sig| {
                sig.complex_norm()
                    .plot_builder()
                    .dft_is_halved(n_samples_time)
                    .layout_extra(|l| l.with_x_axis_min(0.0).with_x_axis_max(150.0))
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            })
            .pipe(|sig| sig.idft_halved(n_samples_time))
            .tap(|sig| {
                sig.plot_builder()
                    .build()
                    .plot_and_layout_to(&mut plots, &mut layouts);
            });

        let scene = Figure::new(2, 4)
            .with_plots(plots)
            .with_layouts(layouts)
            .with_shared_legend_bottom()
            .render();

        save_scene(scene, get_figure_filepath("pitch.svg")?)?;
    }

    // Supplementary figure 2: effect of speedup on frequencies
    {
        let mut plots = Vec::new();
        let mut layouts = Vec::new();

        let signal_1 = FreqSignal::new(
            [vec![cf32::new(0.0, 0.0); 48_000 / 2 + 1].tap_mut(|vec| {
                vec[100].re = 1.0;
                vec[500].re = 2.0;
            })],
            Some(48_000.0),
        );

        let signal_2 = signal_1
            .view()
            .idft_halved(48_000)
            .scale_time(2.0)
            .dft_halved();

        signal_1
            .complex_norm()
            .plot_builder()
            .dft_is_halved(48_000)
            .plot_stroke_width(1.5)
            .layout_extra(|l| {
                l.with_x_axis_min(0.0)
                    .with_x_axis_max(1100.0)
                    .with_y_axis_min(0.0)
            })
            .build()
            .plot_and_layout_to(&mut plots, &mut layouts);

        signal_2
            .complex_norm()
            .plot_builder()
            .dft_is_halved(48_000 / 2)
            .plot_stroke_width(1.5)
            .layout_extra(|l| {
                l.with_x_axis_min(0.0)
                    .with_x_axis_max(1100.0)
                    .with_y_axis_min(0.0)
            })
            .build()
            .plot_and_layout_to(&mut plots, &mut layouts);

        let scene = Figure::new(1, 2)
            .with_plots(plots)
            .with_layouts(layouts)
            .with_shared_legend_bottom()
            .render();

        save_scene(scene, "docs/figures/supp_2.svg")?;
    };

    Ok(())
}

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
mod conv_lti; // linear time-invariant convolution
mod conv_ltv; // linear time-variant convolution
mod utils;

// Exports
pub use conv_lti::_convolve_lti;
pub use conv_ltv::{_convolve_ltv, LtvFilter};

pub(crate) use conv_lti::DefinedLtiConvolution;
pub(crate) use conv_ltv::DefinedLtvConvolution;
use num_traits::AsPrimitive;
pub(crate) use utils::DSPUtils;

// Imports
use crate::{
    math::{Hertz, Seconds, sinc},
    signal::{Domain, Sample, Signal, SignalSlice},
};
use itertools::Itertools;
use std::{f32::consts::PI as PI_F32, f64::consts::PI as PI_F64};

/// Digital Signal Processing trait.
#[allow(private_bounds)]
pub trait DSP<const C: usize, S: Sample, D: Domain>: Clone + DSPUtils<C, S, D> {
    fn is_empty(&self) -> bool;

    fn sampling_rate(&self) -> Option<Hertz>;

    fn domain(&self) -> D;

    /// Clones data if `self` is a [`SignalSlice`], returns `self` if already an owned [`Signal`].
    fn into_owned(self) -> Signal<C, S, D>;

    fn len(&self) -> usize {
        // Checks that all channels have the same length
        debug_assert!(self._map_cha(|cha| cha.len()).iter().all_equal());
        self._chas().first().map(|cha| cha.len()).unwrap_or(0)
    }

    fn lens(&self) -> [usize; C] {
        self._map_cha(|cha| cha.len())
    }

    fn max_len(&self) -> usize {
        self.lens().into_iter().max().unwrap_or(0)
    }

    fn sampling_rate_f32(&self) -> Option<f32> {
        self.sampling_rate().map(|sr| sr as f32)
    }

    fn sampling_rate_usize(&self) -> Option<usize> {
        self.sampling_rate().map(|sr| sr.round() as usize)
    }

    fn sampling_rate_u32(&self) -> Option<u32> {
        self.sampling_rate().map(|sr| sr.round() as u32)
    }

    /// Alias for [`Self::sampling_rate`].
    fn sr(&self) -> Option<Hertz> {
        self.sampling_rate()
    }

    fn sr_or_panic(&self) -> Hertz {
        self.sampling_rate()
            .expect("Signal is required to have a defined sampling rate")
    }

    fn sr_f32_or_panic(&self) -> f32 {
        self.sampling_rate_f32()
            .expect("Signal is required to have a defined sampling rate")
    }

    fn sr_usize_or_panic(&self) -> usize {
        self.sampling_rate_usize()
            .expect("Signal is required to have a defined sampling rate")
    }

    fn sr_u32_or_panic(&self) -> u32 {
        self.sampling_rate_u32()
            .expect("Signal is required to have a defined sampling rate")
    }

    /// Note: will only panic if sampling rates are incoherent when debug assertions are enabled.
    fn resolve_sampling_rate_pair(sr_1: Option<Hertz>, sr_2: Option<Hertz>) -> Option<Hertz> {
        debug_assert!(
            sr_1.zip(sr_2)
                .is_none_or(|(a, b)| approx::abs_diff_eq!(a, b, epsilon = 0.1)),
            "Sampling rates mismatch"
        );
        sr_1.or(sr_2)
    }

    /// Note: will only panic if sampling rates are incoherent when debug assertions are enabled.
    fn resolve_sampling_rate_many<I>(input: I) -> Option<Hertz>
    where
        I: IntoIterator<Item = Option<Hertz>>,
    {
        let mut iter = input.into_iter();
        let sampling_rate = iter.find_map(|e| e);

        debug_assert!(
            sampling_rate.is_none_or(|sr_1| iter.flatten().all(|sr_2| approx::abs_diff_eq!(
                sr_1,
                sr_2,
                epsilon = 0.1
            ))),
            "Sampling rates mismatch"
        );

        sampling_rate
    }

    fn seconds_to_index(&self, s: Seconds) -> usize {
        (self.sr_or_panic() * s).round() as usize
    }

    fn first_n_owned(self, n: usize) -> Signal<C, S, D> {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| v.truncate(n));
        out
    }

    fn skip_first_n_owned(self, n: usize) -> Signal<C, S, D> {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| {
            v.drain(0..n);
        });
        out
    }

    fn last_n_owned(self, n: usize) -> Signal<C, S, D> {
        let len = self.len();
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| {
            v.drain(0..(len - n));
        });
        out
    }

    fn skip_last_n_owned(self, n: usize) -> Signal<C, S, D> {
        let t_len = self.len() - n;
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| {
            v.truncate(t_len);
        });
        out
    }

    fn stack_owned<const C1: usize, const C2: usize, T1, T2>(a: T1, b: T2) -> Signal<C, S, D>
    where
        T1: DSP<C1, S, D>,
        T2: DSP<C2, S, D>,
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(a.sr(), b.sr());
        let channels = itertools::chain!(
            a.into_owned().channels.into_iter(),
            b.into_owned().channels.into_iter(),
        )
        .collect_array::<C>()
        .unwrap();

        Signal::new(channels, sampling_rate)
    }

    fn merge<T1, T2>(a: T1, b: T2) -> Signal<C, S, D>
    where
        T1: DSP<C, S, D>,
        T2: DSP<C, S, D>,
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(a.sr(), b.sr());

        let (mut acc, other) = if a.len() >= b.len() {
            (a.into_owned(), b._as_view())
        } else {
            (b.into_owned(), a._as_view())
        };

        for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other.iter_cha()) {
            cha_acc.iter_mut().zip(cha_other).for_each(|(l, r)| *l += r)
        }

        acc.with_sr_opt(sampling_rate)
    }

    fn merge_with<T>(self, other: T) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
    {
        Self::merge(self, other)
    }

    fn merge_with_at<T>(self, other: T, offset: usize) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(self.sr(), other.sr());
        let total_length = self.max_len().max(offset + other.max_len());

        let mut acc = self.pad_right_to_len(total_length);

        for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other._iter_cha()) {
            cha_acc[offset..]
                .iter_mut()
                .zip(cha_other)
                .for_each(|(l, r)| *l += r)
        }

        acc.with_sr_opt(sampling_rate)
    }

    fn merge_many<I>(input: I) -> Signal<C, S, D>
    where
        I: IntoIterator<Item = Self>,
    {
        let mut input = input.into_iter().collect_vec();

        let sampling_rate =
            Self::resolve_sampling_rate_many(input.iter().map(|e| e.sampling_rate()));

        let mut acc = input
            .swap_remove(input.iter().position_max_by_key(DSP::len).unwrap())
            .into_owned();

        for other in input {
            for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other._iter_cha()) {
                cha_acc
                    .iter_mut()
                    .zip(cha_other)
                    .for_each(|(l, r)| *l += *r)
            }
        }

        acc.with_sr_opt(sampling_rate)
    }

    fn apply<F>(self, mut op: F) -> Signal<C, S, D>
    where
        F: FnMut(S) -> S,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x = op(*x)));
        out
    }

    fn apply_enumerate<F>(self, mut op: F) -> Signal<C, S, D>
    where
        F: FnMut((usize, S)) -> S,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|cha| {
            cha.iter_mut()
                .enumerate()
                .for_each(|(n, x)| *x = op((n, *x)))
        });
        out
    }

    #[allow(non_snake_case)]
    /// The parameter M is such that the window is "M elements - center element - M elements".
    /// Of course, for elements near the boundary, their window size will be less than 2M+1 and
    /// not centered directly on themselves (i.e. near left, right side of the window is larger than left).
    fn apply_centered_window<F>(&self, M: usize, mut op: F) -> Signal<C, S, D>
    where
        F: for<'a> FnMut(&[S]) -> S,
    {
        let channels = self._map_cha(|cha| {
            let mut buffer: Vec<S> = Vec::with_capacity(cha.len());
            for i in 0..cha.len() {
                let range = i.saturating_sub(M)..(i + M + 1).min(cha.len());
                let x = unsafe { cha.get_unchecked(range) };
                buffer.push(op(x))
            }
            buffer
        });
        Signal::new(channels, self.sampling_rate())
    }

    fn abs(self) -> Signal<C, S, D>
    where
        S: num_traits::Signed,
    {
        self.apply(|x| x.abs())
    }

    fn get_abs_max(&self) -> S
    where
        S: num_traits::Float,
    {
        self._iter_cha()
            .map(|cha| cha.iter().copied().map(S::abs).reduce(S::max).unwrap())
            .reduce(S::max)
            .unwrap()
    }

    fn get_abs_max_index(&self) -> usize
    where
        S: num_traits::Float,
    {
        self._iter_cha()
            .map(|cha| {
                cha.iter()
                    .copied()
                    .map(S::abs)
                    .enumerate()
                    .reduce(|a, b| if a.1 > b.1 { a } else { b })
                    .unwrap()
            })
            .reduce(|a, b| if a.1 > b.1 { a } else { b })
            .unwrap()
            .0
    }

    fn normalize(self) -> Signal<C, S, D>
    where
        S: num_traits::Float,
    {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned();
        if abs_max == S::zero() {
            return out;
        }
        let factor = S::one() / abs_max;
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x *= factor));
        out
    }

    fn normalize_to(self, val: S) -> Signal<C, S, D>
    where
        S: num_traits::Float,
    {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned();
        if abs_max == S::zero() {
            return out;
        }
        let factor = val / abs_max;
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x *= factor));
        out
    }

    fn clamp(self) -> Signal<C, S, D>
    where
        S: num_traits::Float,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|cha| {
            cha.iter_mut()
                .for_each(|x| *x = S::clamp(*x, -S::one(), S::one()))
        });
        out
    }

    /// Normalize only if we have a value 'x' somewhere such that x ∉ [-1, 1]
    fn clamp_normalize(self) -> Signal<C, S, D>
    where
        S: num_traits::Float,
    {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned();
        if abs_max > S::one() {
            out.iter_cha_mut()
                .for_each(|cha| cha.iter_mut().for_each(|x| *x /= abs_max));
        }
        out
    }

    #[allow(non_snake_case)]
    fn resample(&self, sampling_rate: Hertz, M: usize) -> Signal<C, S, D>
    where
        f64: num_traits::AsPrimitive<S>,
    {
        assert!(sampling_rate > 0.0);
        let f1 = self.sr_or_panic();
        let T1 = 1.0 / f1;
        let f2 = sampling_rate;
        let T2 = 1.0 / f2;

        let channels = self._map_cha(|x_f1| {
            let x_f1_len = x_f1.len();
            let x_f2_len = (f2 * x_f1_len as f64 / f1).ceil() as usize;

            let mut x_f2 = Vec::with_capacity(x_f2_len);

            for n2 in 0..x_f2_len {
                // Alternative approach, the maximum argument allowed, as when |x| -> ∞, sinc(x) -> 0.
                //let n1_start = (n2 as f64 * f1 / f2 - sinc_arg_abs_max).floor().max(0.0) as usize;
                //let n1_stop = (1 + (n2 as f64 * f1 / f2 + sinc_arg_abs_max).ceil() as usize).min(x_f1_len);

                let n1_center_ideal = (n2 as f64 * f1 / f2) as f64;
                let n1_start = (n1_center_ideal.round() as usize).saturating_sub(M);
                let n1_stop = (n1_center_ideal.round() as usize + M + 1).min(x_f1_len);

                let x_f2_n2 = (n1_start..n1_stop)
                    .map(|n1| {
                        let hann_like =
                            f64::cos(PI_F64 * (n1 as f64 - n1_center_ideal) / (2 * M) as f64)
                                .powi(2);
                        x_f1[n1] * (hann_like * sinc(f1 * (n2 as f64 * T2 - n1 as f64 * T1))).as_()
                    })
                    .sum::<S>();

                x_f2.push(x_f2_n2);
            }

            x_f2
        });

        Signal::new(channels, Some(sampling_rate))
    }

    #[allow(non_snake_case)]
    fn resample_pure(&self, sampling_rate: Hertz) -> Signal<C, S, D>
    where
        f64: num_traits::AsPrimitive<S>,
    {
        assert!(sampling_rate > 0.0);
        let f1 = self.sr_or_panic();
        let T1 = 1.0 / f1;
        let f2 = sampling_rate;
        let T2 = 1.0 / f2;

        let channels = self._map_cha(|x_f1| {
            let x_f1_len = x_f1.len();
            let x_f2_len = (f2 * x_f1_len as f64 / f1).ceil() as usize;

            let mut x_f2 = Vec::with_capacity(x_f2_len);

            for n2 in 0..x_f2_len {
                let x_f2_n2 = x_f1
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(n1, x_f1_n1)| {
                        x_f1_n1 * sinc(f1 * (n2 as f64 * T2 - n1 as f64 * T1)).as_()
                    })
                    .sum::<S>();

                x_f2.push(x_f2_n2);
            }

            x_f2
        });

        Signal::new(channels, Some(sampling_rate))
    }

    fn concatenate<T, I>(input: I) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
        I: IntoIterator<Item = T>,
    {
        let input = input.into_iter().collect_vec();
        let sampling_rate =
            Self::resolve_sampling_rate_many(input.iter().map(|e| e.sampling_rate()));

        let capacity: usize = input.iter().map(|buf| buf.len()).sum();
        let mut out = Signal::<C, S, D>::with_capacity(capacity, sampling_rate);

        for buf in input.into_iter() {
            for (cha, other) in out.channels.iter_mut().zip(buf._iter_cha()) {
                cha.extend_from_slice(other);
            }
        }

        out
    }

    fn pad_right(self, by: usize) -> Signal<C, S, D> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut()
            .for_each(|buf| buf.resize(buf.len() + by, S::zero()));
        buf
    }

    fn pad_right_to_len(self, len: usize) -> Signal<C, S, D> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut()
            .filter(|buf| buf.len() < len)
            .for_each(|buf| buf.resize(len, S::zero()));
        buf
    }

    fn pad_right_with(self, by: usize, val: S) -> Signal<C, S, D> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut()
            .for_each(|buf| buf.resize(buf.len() + by, val));
        buf
    }

    fn pad_right_with_last(self, by: usize) -> Signal<C, S, D> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut().for_each(|buf| {
            let val = *buf.last().unwrap_or(&S::zero());
            buf.resize(buf.len() + by, val)
        });
        buf
    }

    fn pad_right_with_last_to_len(self, len: usize) -> Signal<C, S, D> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut()
            .filter(|buf| buf.len() < len)
            .for_each(|buf| {
                let val = *buf.last().unwrap_or(&S::zero());
                buf.resize(len, val);
            });
        buf
    }

    fn pad_left(self, by: usize) -> Signal<C, S, D> {
        Signal::new(
            self._map_cha(|cha| {
                let mut vec: Vec<S> = vec![S::zero(); by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn pad_left_with(self, by: usize, val: S) -> Signal<C, S, D> {
        Signal::new(
            self._map_cha(|cha| {
                let mut vec: Vec<S> = vec![val; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn pad_left_with_first(self, by: usize) -> Signal<C, S, D> {
        Signal::new(
            self._map_cha(|cha| {
                let val = *cha.first().unwrap_or(&S::zero());
                let mut vec: Vec<S> = vec![val; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn convolve<T, const C2: usize, const C3: usize>(
        &self,
        h: T,
    ) -> <Self as DefinedLtiConvolution<C, C2, C3, S, D>>::Output
    where
        Self: DefinedLtiConvolution<C, C2, C3, S, D>,
        T: DSP<C2, S, D>,
    {
        self.convolve_with(h)
    }

    /// Similar to using `numpy.convolve(a, v, mode='same')`, we keep the output "aligned" with
    /// our input; left-biased in the case where the filter `h` has an even length.
    fn convolve_then_crop<T, const C2: usize, const C3: usize>(&self, h: T) -> Signal<C3, S, D>
    where
        Self: DefinedLtiConvolution<C, C2, C3, S, D>,
        T: DSP<C2, S, D>,
    {
        let filter_size = h.len();
        let y = self.convolve_with(h);

        #[allow(non_snake_case)]
        let M = filter_size / 2;

        if filter_size % 2 == 1 {
            // the ideal case, from M to N-M-1
            y.skip_first_n_owned(M).skip_last_n_owned(M)
        } else {
            eprintln!("Warning, the filter has a length that is even");
            y.skip_first_n_owned(M - 1).skip_last_n_owned(M)
        }
    }

    fn convolve_ltv<STATE, H, F1, F2>(&self, h: LtvFilter<STATE, H, F1, F2, S>) -> Signal<1, S, D>
    where
        Self: DefinedLtvConvolution<S, D>,
        F1: Fn(usize, &mut STATE, &mut Option<H>),
        F2: for<'a> Fn(&'a H) -> &'a [S],
    {
        self.convolve_ltv_with(h)
    }

    fn interleaved_samples(&self) -> Vec<S> {
        let size = self.len() * C;
        let mut out: Vec<S> = vec![S::zero(); size];
        for (cha_idx, cha) in self._iter_cha().enumerate() {
            cha.iter()
                .copied()
                .enumerate()
                .for_each(|(i, x)| out[cha_idx + C * i] = x);
        }
        out
    }

    #[cfg(feature = "plot")]
    fn plot_builder(&self) -> plot_impl::SignalPlotterBuilder
    where
        S: AsPrimitive<f64>,
    {
        let points = self._map_cha(|cha| {
            cha.iter()
                .enumerate()
                .map(|(i, sam)| (i as f64, sam.as_()))
                .collect_vec()
        });

        plot_impl::SignalPlotter::builder(points.to_vec(), self.sr(), self.domain().to_string())
    }
}

#[cfg(feature = "plot")]
mod plot_impl {
    use core::ops::DivAssign;
    use kuva::prelude::*;
    use std::io::Write;
    use tap::{Pipe, Tap};

    #[derive(bon::Builder)]
    pub struct SignalPlotter {
        #[builder(start_fn)]
        points: Vec<Vec<(f64, f64)>>,
        #[builder(start_fn)]
        sampling_rate: Option<f64>,
        #[builder(start_fn)]
        domain_str: String,

        #[builder(default = false)]
        force_discrete_x_axis: bool,

        #[builder(default = Palette::wong())]
        plot_palette: Palette,
        #[builder(default = 1.0_f64)]
        plot_line_stroke_width: f64,
        #[builder(with = |f: impl FnMut(LinePlot) -> LinePlot + 'static| {Box::new(f)})]
        plot_extra: Option<Box<dyn FnMut(LinePlot) -> LinePlot>>,

        #[builder(into)]
        layout_title: Option<String>,
        #[builder(with = |f: impl FnMut(Layout) -> Layout + 'static| {Box::new(f)})]
        layout_extra: Option<Box<dyn FnMut(Layout) -> Layout>>,
    }

    impl SignalPlotter {
        pub fn plot(mut self) -> Vec<Plot> {
            let n_channels = self.points.len();

            if let Some(sr) = self.sampling_rate
                && !self.force_discrete_x_axis
            {
                self.points
                    .iter_mut()
                    .for_each(|points| points.iter_mut().for_each(|(x, _)| x.div_assign(sr)));
            }

            self.points
                .into_iter()
                .enumerate()
                .map(|(i, points)| {
                    let legend: String = match (i, n_channels) {
                        (0, 2) => "channel 0 (left ear)".to_string(),
                        (1, 2) => "channel 1 (right ear)".to_string(),
                        (i, _) => format!("channel {i}"),
                    };
                    let color = &self.plot_palette.colors()[i % self.plot_palette.len()];

                    LinePlot::new()
                        .with_data(points)
                        .with_color(color)
                        .with_legend(legend)
                        .with_stroke_width(self.plot_line_stroke_width)
                        .pipe(|lp| {
                            if let Some(extra_fn) = self.plot_extra.as_mut() {
                                extra_fn(lp)
                            } else {
                                lp
                            }
                        })
                })
                .map(Plot::from)
                .collect()
        }

        pub fn plot_and_layout(mut self) -> (Vec<Plot>, Layout) {
            let title = self.layout_title.as_ref().cloned().unwrap_or_else(|| {
                format!(
                    "Signal ({n_channels} channels, {dom} domain)",
                    n_channels = self.points.len(),
                    dom = self.domain_str
                )
            });
            let mut layout_extra_fn = self.layout_extra.take();

            let plot = self.plot();

            let layout = Layout::auto_from_plots(&plot).with_title(title).pipe(|la| {
                if let Some(extra_fn) = layout_extra_fn.as_mut() {
                    extra_fn(la)
                } else {
                    la
                }
            });

            (plot, layout)
        }

        pub fn direct_to_file<Q: AsRef<std::path::Path>>(self, fp: Q) -> anyhow::Result<()> {
            let (plot, layout) = self.plot_and_layout();
            let data = render_to_svg(plot, layout);

            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(fp)?;

            file.write_all(data.as_bytes())?;
            file.sync_all()?;

            Ok(())
        }
    }
}

impl<const C: usize, S: Sample, D: Domain> DSP<C, S, D> for Signal<C, S, D> {
    fn is_empty(&self) -> bool {
        self.channels.first().map(Vec::is_empty).unwrap_or(true)
    }
    fn sampling_rate(&self) -> Option<Hertz> {
        self.sampling_rate
    }
    fn domain(&self) -> D {
        self._domain
    }
    fn into_owned(self) -> Signal<C, S, D> {
        self
    }
}

impl<const C: usize, S: Sample, D: Domain> DSP<C, S, D> for SignalSlice<'_, C, S, D> {
    fn is_empty(&self) -> bool {
        self.channels
            .first()
            .map(|cha| cha.is_empty())
            .unwrap_or(true)
    }
    fn sampling_rate(&self) -> Option<Hertz> {
        self.sampling_rate
    }
    fn domain(&self) -> D {
        self._domain
    }
    fn into_owned(self) -> Signal<C, S, D> {
        Signal::new(self.map_cha(|cha| cha.to_vec()), self.sampling_rate)
    }
}

impl<const C: usize, S: Sample, D: Domain, T> DSP<C, S, D> for &T
where
    T: DSP<C, S, D>,
{
    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }
    fn sampling_rate(&self) -> Option<Hertz> {
        (*self).sampling_rate()
    }
    fn domain(&self) -> D {
        (*self).domain()
    }
    fn into_owned(self) -> Signal<C, S, D> {
        (*self).clone().into_owned()
    }
}

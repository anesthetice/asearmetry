/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
mod conv_lti; // linear time-invariant convolution
mod conv_ltv; // linear time-variant convolution
mod fourier; // DFT, FFT, IDFT, IFFT
mod utils;

// Exports
pub use conv_lti::_convolve_lti;
pub use conv_ltv::{_convolve_ltv, LtvFilter};
pub use fourier::{_dft, _dft_rayon, _fft_full, _fft_halved, _idft, _idft_rayon, _ifft};

pub(crate) use conv_lti::DefinedLtiConvolution;
pub(crate) use conv_ltv::DefinedLtvConvolution;
use num_complex::ComplexFloat;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
pub(crate) use utils::DSPUtils;

#[cfg(feature = "plot")]
use crate::signal::IsKnownSampleType;
// Imports
use crate::{
    math::{Hertz, cf32, hann_window_iter, sinc},
    signal::{
        Domain, FreqDomain, Sample, Signal, SignalSlice, StftDomain, TimeDomain, core::StftInfo,
    },
};
use itertools::Itertools;
use num_traits::AsPrimitive;
use std::{
    f32::consts::PI as PI_F32,
    f64::consts::PI as PI_F64,
    marker::PhantomData,
    ops::{AddAssign, MulAssign},
};
use tap::{Pipe, Tap};

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

    fn index<I>(&self, index: I) -> [S; C]
    where
        I: std::slice::SliceIndex<[S], Output = S> + Copy,
    {
        self._map_cha(|cha| cha[index])
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

    fn clarify_owned(self) -> Signal<C, S::Inner, D> {
        let sampling_rate = self.sr();
        let channels = self.into_owned().channels.map(|src| unsafe {
            let (ptr, length, capacity) = src.into_raw_parts();
            let ptr: *mut S::Inner = ptr as *mut S::Inner;
            Vec::from_raw_parts(ptr, length, capacity)
        });

        Signal {
            channels,
            sampling_rate,
            _domain: PhantomData,
        }
    }

    fn clarify_ref(&self) -> SignalSlice<'_, C, S::Inner, D> {
        let sampling_rate = self.sr();
        let channels = self._map_cha(|src| unsafe {
            core::slice::from_raw_parts(src.as_ptr() as *const S::Inner, src.len())
        });

        SignalSlice {
            channels,
            sampling_rate,
            _domain: PhantomData,
        }
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
            (a.into_owned(), b._view())
        } else {
            (b.into_owned(), a._view())
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
        let total_length = self.len().max(offset + other.len());

        let mut acc = self.pad_right_to_len(total_length);

        for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other._iter_cha()) {
            cha_acc[offset..]
                .iter_mut()
                .zip(cha_other)
                .for_each(|(l, r)| *l += r)
        }

        acc.with_sr_opt(sampling_rate)
    }

    fn merge_many<T>(input: impl IntoIterator<Item = T>, max_len: Option<usize>) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
    {
        let mut sampling_rate = None;
        let mut acc = Signal::<C, S, D>::new_zeros(max_len.unwrap_or(0));

        input.into_iter().for_each(|signal| {
            sampling_rate = Self::resolve_sampling_rate_pair(sampling_rate, signal.sr());

            acc.iter_cha_mut()
                .zip(signal._iter_cha())
                .for_each(|(cha_acc, cha)| {
                    cha_acc
                        .iter_mut()
                        .zip(cha)
                        .for_each(|(l, r)| l.add_assign(r));

                    if cha_acc.len() < cha.len() {
                        cha_acc.extend_from_slice(&cha[cha_acc.len()..]);
                    }
                });
        });

        acc.with_sr_opt(sampling_rate)
    }

    fn merge_many_at<T>(
        input: impl IntoIterator<Item = (T, usize)>,
        max_len: Option<usize>,
    ) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
    {
        let mut sampling_rate = None;
        let mut acc = Signal::<C, S, D>::new_zeros(max_len.unwrap_or(0));

        input.into_iter().for_each(|(signal, offset)| {
            sampling_rate = Self::resolve_sampling_rate_pair(sampling_rate, signal.sr());

            acc.iter_cha_mut()
                .zip(signal._iter_cha())
                .for_each(|(cha_acc, cha)| {
                    if cha_acc.len() < offset {
                        cha_acc.resize(offset, S::zero());
                    }
                    cha_acc[offset..]
                        .iter_mut()
                        .zip(cha)
                        .for_each(|(l, r)| l.add_assign(r));

                    if cha_acc.len() < offset + cha.len() {
                        cha_acc.extend_from_slice(&cha[cha_acc.len() - offset..]);
                    }
                });
        });

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

    fn apply_into<S2, F>(self, mut op: F) -> Signal<C, S2, D>
    where
        S2: Sample,
        F: FnMut(S) -> S2,
    {
        let channels: [Vec<S2>; C] =
            self._map_cha(|cha| cha.iter().copied().map(|x| op(x)).collect());
        Signal::new(channels, self.sr())
    }

    fn apply_into_enumerate<S2, F>(self, mut op: F) -> Signal<C, S2, D>
    where
        S2: Sample,
        F: FnMut((usize, S)) -> S2,
    {
        let channels: [Vec<S2>; C] =
            self._map_cha(|cha| cha.iter().copied().enumerate().map(|tup| op(tup)).collect());
        Signal::new(channels, self.sr())
    }

    #[allow(non_snake_case)]
    /// The parameter M is such that the window is "M elements - center element - M elements".
    /// Of course, for elements near the boundary, their window size will be less than 2M+1 and
    /// not centered directly on themselves (i.e. near left, right side of the window is larger than left).
    fn apply_into_centered_window<S2, F>(&self, M: usize, mut op: F) -> Signal<C, S2, D>
    where
        S2: Sample,
        F: for<'a> FnMut(&[S]) -> S2,
    {
        let channels = self._map_cha(|cha| {
            let mut buffer: Vec<S2> = Vec::with_capacity(cha.len());
            for i in 0..cha.len() {
                let range = i.saturating_sub(M)..(i + M + 1).min(cha.len());
                let x = unsafe { cha.get_unchecked(range) };
                buffer.push(op(x))
            }
            buffer
        });
        Signal::new(channels, self.sampling_rate())
    }

    fn apply_with<S1, T1, S2, F>(self, other: T1, mut op: F) -> Signal<C, S2, D>
    where
        S1: Sample,
        T1: DSP<C, S1, D>,
        S2: Sample,
        F: FnMut(S, S1) -> S2,
    {
        let mut out = Signal::new_zeros(self.len().min(other.len()))
            .with_sr_opt(Self::resolve_sampling_rate_pair(self.sr(), other.sr()));

        out.iter_cha_mut().enumerate().for_each(|(c, cha_mut)| {
            itertools::izip!(cha_mut, self._cha(c), other._cha(c))
                .for_each(|(y, &x1, &x2)| *y = op(x1, x2));
        });

        out
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

    fn clamp_to(self, val: S) -> Signal<C, S, D>
    where
        S: num_traits::Float,
    {
        let max = val.abs();
        let min = -max;

        let mut out = self.into_owned();
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x = S::clamp(*x, min, max)));
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
        S: ComplexFloat + MulAssign<S::Real>,
        f64: AsPrimitive<S::Real>,
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
                // Alternative approach, using the maximum argument allowed, as when |x| -> ∞, sinc(x) -> 0.
                //let n1_start = (n2 as f64 * f1 / f2 - sinc_arg_abs_max).floor().max(0.0) as usize;
                //let n1_stop = (1 + (n2 as f64 * f1 / f2 + sinc_arg_abs_max).ceil() as usize).min(x_f1_len);

                let n1_center_ideal = n2 as f64 * f1 / f2;
                let n1_start = (n1_center_ideal.round() as usize).saturating_sub(M);
                let n1_stop = (n1_center_ideal.round() as usize + M + 1).min(x_f1_len);

                let x_f2_n2 = (n1_start..n1_stop)
                    .map(|n1| {
                        let hann_like =
                            f64::cos(PI_F64 * (n1 as f64 - n1_center_ideal) / (2 * M) as f64)
                                .powi(2);
                        let mul_by = hann_like * sinc(f1 * (n2 as f64 * T2 - n1 as f64 * T1));

                        unsafe { *x_f1.get_unchecked(n1) }
                            .tap_mut(|x_f1_n1| x_f1_n1.mul_assign(mul_by.as_()))
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
        S: ComplexFloat + MulAssign<S::Real>,
        f64: AsPrimitive<S::Real>,
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
                        let mul_by = sinc(f1 * (n2 as f64 * T2 - n1 as f64 * T1));
                        x_f1_n1.tap_mut(|x| x.mul_assign(mul_by.as_()))
                    })
                    .sum::<S>();

                x_f2.push(x_f2_n2);
            }

            x_f2
        });

        Signal::new(channels, Some(sampling_rate))
    }

    fn concatenate_with<T>(self, other: T) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
    {
        self.into_owned().tap_mut(|sig| {
            sig.set_sr_opt(Self::resolve_sampling_rate_pair(sig.sr(), other.sr()))
                .channels
                .iter_mut()
                .zip(other._iter_cha())
                .for_each(|(cha_mut, cha)| cha_mut.extend_from_slice(cha))
        })
    }

    fn concatenate<T, I>(input: I) -> Signal<C, S, D>
    where
        T: DSP<C, S, D>,
        I: IntoIterator<Item = T>,
    {
        input
            .into_iter()
            .fold(Signal::new_empty(), |acc, sig| acc.concatenate_with(sig))
    }

    /// Negative integer -> roll to the left, positive integer -> roll to the right.
    fn shift(self, by: [isize; C], replacement: [S; C]) -> Signal<C, S, D> {
        let len = self.len();
        let mut out = self.into_owned();
        itertools::izip!(out.channels.iter_mut(), by, replacement).for_each(|(cha, by, repl)| {
            if by < 0 {
                cha.drain(0..by.unsigned_abs());
                cha.resize(len, repl);
            } else if by > 0 {
                let mut vec = vec![repl; by as usize];
                vec.extend_from_slice(&cha[0..len.saturating_sub(by as usize)]);
                *cha = vec;
            }
        });
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

    fn pad_left(&self, by: usize) -> Signal<C, S, D> {
        Signal::new(
            self._map_cha(|cha| {
                let mut vec: Vec<S> = vec![S::zero(); by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn pad_left_with(&self, by: usize, val: S) -> Signal<C, S, D> {
        Signal::new(
            self._map_cha(|cha| {
                let mut vec: Vec<S> = vec![val; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn pad_left_with_first(&self, by: usize) -> Signal<C, S, D> {
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

    fn mul(self, rhs: Self) -> Signal<C, S, D> {
        let (mut out, other) = if self.len() <= rhs.len() {
            (self.into_owned(), rhs._view())
        } else {
            (rhs.into_owned(), self._view())
        };

        out.iter_cha_mut()
            .zip(other.iter_cha())
            .for_each(|(cha_mut, cha_other)| {
                cha_mut
                    .iter_mut()
                    .zip(cha_other)
                    .for_each(|(s, s_)| s.mul_assign(*s_));
            });

        out
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

    /// Computes the Discrete Fourier Transform (DFT) of the signal.
    ///
    /// Note that for the sake of simplicity the entire DFT will be kept instead
    /// of it being truncated to a length of ⌊N/2⌋+1 as one might expect when working
    /// with signals in the time domain which only contain real values.
    fn dft(self) -> Signal<C, cf32, FreqDomain>
    where
        S: Sample<Inner = f32>,
        D: Domain<Inner = TimeDomain>,
    {
        self.clarify_owned()
            .with_transform_cha(fourier::_fft_full)
            .with_domain(FreqDomain {})
    }

    fn dft_halved(self) -> Signal<C, cf32, FreqDomain>
    where
        S: Sample<Inner = f32>,
        D: Domain<Inner = TimeDomain>,
    {
        self.clarify_owned()
            .with_transform_cha(fourier::_fft_halved)
            .with_domain(FreqDomain {})
    }

    /// Computes the Inverse Discrete Fourier Transform (IDFT) of the signal.
    ///
    /// Expects `self` to contain the entire DFT of length N, identical
    /// to the length of the signal in the time domain. Instead of what
    /// one may be accustomed to, for transforms where the signal in the
    /// time domain only contains real values, wherein only ⌊N/2⌋+1 of
    /// the frequency samples are kept (as F[k] = F*[N-k] for k ∈ {0, …, N-1}).
    #[allow(non_snake_case)]
    fn idft(self) -> Signal<C, f32, TimeDomain>
    where
        S: Sample<Inner = cf32>,
        D: Domain<Inner = FreqDomain>,
    {
        let N_t = self.len();
        let N_f = (N_t / 2) + 1;
        self.clarify_owned()
            .with_transform_cha(|mut cha| {
                cha.truncate(N_f);
                fourier::_ifft(cha, N_t)
            })
            .with_domain(TimeDomain {})
    }

    /// The parameter `N_t` corresponds to the number of samples in the time domain,
    /// and if we assume `N_f` to be the number of samples kept in the frequency domain,
    /// then N_t = 2 ⋅ N_f - 1 if N_t was originally odd, N_t = 2 ⋅ N_f - 2 otherwise.
    #[allow(non_snake_case)]
    fn idft_halved(self, N_t: usize) -> Signal<C, f32, TimeDomain>
    where
        S: Sample<Inner = cf32>,
        D: Domain<Inner = FreqDomain>,
    {
        self.clarify_owned()
            .with_transform_cha(|cha| fourier::_ifft(cha, N_t))
            .with_domain(TimeDomain {})
    }

    /// Computes the Short-time Discrete Fourier Transform (STFT) of the signal.
    #[allow(non_snake_case)]
    fn stft(
        &self,
        hop_length: usize,
        frame_length: usize,
    ) -> (Signal<C, cf32, StftDomain>, StftInfo)
    where
        S: Sample<Inner = f32>,
        D: Domain<Inner = TimeDomain>,
    {
        assert!(
            hop_length > 0,
            "The hop length (i.e., the stride or step size) cannot be zero"
        );
        assert!(
            frame_length > 0,
            "The frame length (i.e., the window size) cannot be zero"
        );

        assert!(
            hop_length <= frame_length - 2,
            "The hop length (i.e., the stride or step size) must be smaller than the frame length minus two"
        );
        assert!(
            frame_length % 2 == 1,
            "The frame length (i.e., the window size) must be odd"
        );

        let N = self.len();
        let W = frame_length;
        let M = W / 2; // W is odd, W = 2M + 1
        let H = hop_length;

        let left_pad_amount = M;
        let right_pad_amount = W - (M + N) % H;

        let x_padded = self
            .clarify_ref()
            .pad_left(left_pad_amount)
            .pad_right(right_pad_amount);

        let hann_window = hann_window_iter::<f32>(W).collect_vec();

        let apply_hann_window = |to: &[f32]| {
            to.iter()
                .zip(&hann_window)
                .map(|(x, hann)| x * hann)
                .collect_vec()
        };

        let 𝒳 = Signal::<C, cf32, FreqDomain>::concatenate(
            x_padded
                .windowize(frame_length, hop_length)
                .collect_vec()
                .into_par_iter() // do not use `par_bridge` as it does not preserve order
                .map(|signal_slice| signal_slice.with_transform(|chas| chas.map(apply_hann_window)))
                .map(DSP::dft_halved)
                .collect::<Vec<Signal<C, cf32, FreqDomain>>>(),
        )
        .with_domain(StftDomain {});

        let stft_info = StftInfo {
            hop_length,
            frame_length,
            padding: (left_pad_amount, right_pad_amount),
        };

        (𝒳, stft_info)
    }

    #[allow(non_snake_case)]
    fn istft(&self, stft_info: StftInfo) -> Signal<C, f32, TimeDomain>
    where
        S: Sample<Inner = cf32>,
        D: Domain<Inner = StftDomain>,
    {
        let 𝒳 = self.clarify_ref();

        let H = stft_info.hop_length;
        let W = stft_info.frame_length;
        let W_f = W / 2 + 1;
        let I = 𝒳.len() / W_f; // number of frames
        let N_padded = W + (I - 1) * H;

        let (left_pad_amount, right_pad_amount) = stft_info.padding;

        assert!(𝒳.len() % W_f == 0);

        let Ω_weight_padded: Signal<C, f32, TimeDomain> = {
            let hann_window = hann_window_iter::<f32>(W).collect_vec();
            let mut Ω: Vec<f32> = vec![0.0; N_padded];
            for i in 0..I {
                let slice_mut = &mut Ω[i * H..(i * H) + W];
                slice_mut
                    .iter_mut()
                    .zip(&hann_window)
                    .for_each(|(x, hann)| x.add_assign(hann));
            }
            Signal::new(std::array::repeat(Ω), None)
        };

        𝒳.as_blocks(W_f)
            .collect_vec()
            .into_par_iter()
            .map(|frame| frame.with_transform_cha(|cha| fourier::_ifft(cha.to_vec(), W)))
            .collect::<Vec<Signal<C, f32, D>>>()
            .into_iter()
            .enumerate()
            .map(|(frame_idx, frame)| (frame, frame_idx * H))
            .pipe(|input| Signal::merge_many_at(input, Some(N_padded)).with_domain(TimeDomain {}))
            .apply_with(Ω_weight_padded, |x, w| x * w.powi(-1))
            .skip_first_n_owned(left_pad_amount)
            .skip_last_n_owned(right_pad_amount)
    }

    fn complex_norm(&self) -> Signal<C, S::Real, D>
    where
        S: num_complex::ComplexFloat,
        <S as num_complex::ComplexFloat>::Real: Sample,
    {
        self._view().with_transform_cha(|cha| {
            cha.iter()
                .copied()
                .map(num_complex::ComplexFloat::abs)
                .collect()
        })
    }

    fn complex_arg(&self) -> Signal<C, S::Real, D>
    where
        S: num_complex::ComplexFloat,
        <S as num_complex::ComplexFloat>::Real: Sample,
    {
        self._view().with_transform_cha(|cha| {
            cha.iter()
                .copied()
                .map(num_complex::ComplexFloat::arg)
                .collect()
        })
    }

    #[allow(non_snake_case)]
    fn into_halved(self) -> Signal<C, S, D>
    where
        S: num_complex::ComplexFloat,
        D: Domain<Inner = FreqDomain>,
    {
        let N_t = self.len();
        let N_f = (N_t / 2) + 1;

        self.into_owned().with_transform_cha(|mut cha| {
            cha.truncate(N_f);
            cha
        })
    }

    #[allow(non_snake_case)]
    fn into_full(self, N_t: usize) -> Signal<C, S, D>
    where
        S: num_complex::ComplexFloat,
        D: Domain<Inner = FreqDomain>,
    {
        let N_f = self.len();
        assert!(N_t == 2 * N_f + 1 || N_t == 2 * (N_f + 1));

        self.into_owned().with_transform_cha(|mut cha| {
            cha.reserve_exact(N_t - N_f);
            for m in N_f..N_t {
                cha.push(cha[N_t - m].conj())
            }
            cha
        })
    }

    fn interleave_samples(&self) -> Vec<S> {
        let final_length = self.len() * C;
        let mut out = Vec::<S>::with_capacity(final_length);

        let input = self._view();

        unsafe {
            let buf = out.spare_capacity_mut();

            let mut i_output = 0;
            for i_sample in 0..input.len() {
                for i_channel in 0..C {
                    buf.get_unchecked_mut(i_output)
                        .write(*input.cha_uc(i_channel).get_unchecked(i_sample));
                    i_output += 1;
                }
            }

            out.set_len(final_length);
        }

        out
    }

    #[cfg(feature = "plot")]
    fn plot_builder(&self) -> plot_impl::SignalPlotterBuilder
    where
        S: AsPrimitive<f64> + IsKnownSampleType,
    {
        use crate::signal::KnownSampleType;

        if matches!(
            <S as IsKnownSampleType>::SAMPLE_TYPE,
            KnownSampleType::CF32 | KnownSampleType::CF64
        ) {
            log::error!("Complex floating points passed directly to `plot_builder`");
            panic!(
                "Complex floating points cannot be plotted. You may wish to apply the `complex_norm` or `complex_arg` methods first"
            );
        }

        let points = self._map_cha(|cha| {
            cha.iter()
                .enumerate()
                .map(|(i, s)| (i as f64, s.as_()))
                .collect_vec()
        });

        plot_impl::SignalPlotter::builder(points.to_vec(), self.sr(), self.domain().into())
    }

    #[cfg(test)]
    fn _test_clarify_methods<T: Sample>(self, expected: Signal<C, T, D>)
    where
        S: Sample<Inner = T>,
    {
        assert_eq!(self.clarify_ref(), expected.view());
        assert_eq!(self.clarify_owned(), expected)
    }
}

#[cfg(feature = "plot")]
mod plot_impl {
    use crate::signal::AnyDomain;
    use core::ops::DivAssign;
    use kuva::prelude::*;
    use std::io::Write;
    use tap::Pipe;

    #[derive(bon::Builder)]
    pub struct SignalPlotter {
        #[builder(start_fn)]
        points: Vec<Vec<(f64, f64)>>,
        #[builder(start_fn)]
        sampling_rate: Option<f64>,
        #[builder(start_fn)]
        domain: AnyDomain,

        #[builder(default = false)]
        x_axis_use_samples: bool,
        #[builder(default = false)]
        use_lineplot_for_freq: bool,
        dft_is_halved: Option<usize>,

        #[builder(default = Palette::wong())]
        plot_palette: Palette,
        #[builder(default = 1.0_f64)]
        plot_stroke_width: f64,
        #[builder(default = 2.0_f64)]
        plot_dot_radius: f64,
        #[builder(with = |f: impl FnMut(LinePlot) -> LinePlot + 'static| {Box::new(f)})]
        plot_extra_lp: Option<Box<dyn FnMut(LinePlot) -> LinePlot>>,
        #[builder(with = |f: impl FnMut(LollipopPlot) -> LollipopPlot + 'static| {Box::new(f)})]
        plot_extra_sp: Option<Box<dyn FnMut(LollipopPlot) -> LollipopPlot>>,

        #[builder(into)]
        layout_title: Option<String>,
        #[builder(with = |f: impl FnMut(Layout) -> Layout + 'static| {Box::new(f)})]
        layout_extra: Option<Box<dyn FnMut(Layout) -> Layout>>,
    }

    impl SignalPlotter {
        pub fn plot(mut self) -> Vec<Plot> {
            let n_channels = self.points.len();

            // Go from full representation of the DFT to the halved one.
            if matches!(self.domain, AnyDomain::Freq) {
                #[allow(non_snake_case)]
                let N_t = match self.dft_is_halved {
                    Some(N_t) => N_t,
                    None => {
                        let N_t = self.points.first().unwrap().len();
                        self.points
                            .iter_mut()
                            .for_each(|points| points.truncate((points.len() / 2) + 1));
                        N_t
                    }
                };

                if let Some(f_s) = self.sampling_rate
                    && !self.x_axis_use_samples
                {
                    #[allow(non_snake_case)]
                    let Δf = f_s / N_t as f64;

                    self.points.iter_mut().for_each(|points| {
                        points
                            .iter_mut()
                            .enumerate()
                            .for_each(|(i, (x, _))| *x = Δf * i as f64);
                    });
                }
            }

            if let Some(sr) = self.sampling_rate
                && !self.x_axis_use_samples
                && matches!(self.domain, AnyDomain::Time)
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

                    match (self.domain, self.use_lineplot_for_freq) {
                        (AnyDomain::Freq, false) => LollipopPlot::new()
                            .with_points(points)
                            .with_color(color)
                            .with_legend(legend)
                            .with_stem_width(self.plot_stroke_width)
                            .with_dot_radius(self.plot_dot_radius)
                            .pipe(|lp| {
                                if let Some(extra_fn) = self.plot_extra_sp.as_mut() {
                                    extra_fn(lp)
                                } else {
                                    lp
                                }
                            })
                            .pipe(Plot::from),
                        _ => LinePlot::new()
                            .with_data(points)
                            .with_color(color)
                            .with_legend(legend)
                            .with_stroke_width(self.plot_stroke_width)
                            .pipe(|lp| {
                                if let Some(extra_fn) = self.plot_extra_lp.as_mut() {
                                    extra_fn(lp)
                                } else {
                                    lp
                                }
                            })
                            .pipe(Plot::from),
                    }
                })
                .collect()
        }

        pub fn plot_and_layout(mut self) -> (Vec<Plot>, Layout) {
            let title = self.layout_title.as_ref().cloned().unwrap_or_else(|| {
                format!(
                    "Signal ({channel_info}, {dom} domain)",
                    channel_info = match self.points.len() {
                        0 => "why are you trying to plot a signal with zero channels?".to_string(),
                        1 => "mono".to_string(),
                        2 => "stereo".to_string(),
                        more => format!("{more} channels"),
                    },
                    dom = self.domain
                )
            });

            let x_label = match (
                self.sampling_rate.is_some(),
                self.domain,
                self.x_axis_use_samples,
            ) {
                (true, AnyDomain::Time, false) => "time [s]",
                (true, AnyDomain::Freq, false) => "frequency [Hz]",
                _ => "sample",
            };

            let mut layout_extra_fn = self.layout_extra.take();

            let plot = self.plot();

            let layout = Layout::auto_from_plots(&plot)
                .with_title(title)
                .with_x_label(x_label)
                .with_x_axis_min(0.0)
                .pipe(|la| {
                    if let Some(extra_fn) = layout_extra_fn.as_mut() {
                        extra_fn(la)
                    } else {
                        la
                    }
                });

            (plot, layout)
        }

        pub fn plot_and_layout_to(self, plots: &mut Vec<Vec<Plot>>, layouts: &mut Vec<Layout>) {
            let (plot, layout) = self.plot_and_layout();
            plots.push(plot);
            layouts.push(layout);
        }

        pub fn render_to_file<Q: AsRef<std::path::Path>>(self, fp: Q) -> anyhow::Result<()> {
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
        D::default()
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
        D::default()
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

#[cfg(test)]
mod test {
    use crate::math::cf32;
    use crate::signal::{DSP, FreqDomain, Signal, TimeDomain};

    #[test]
    fn check_clarify_methods() {
        let input: Signal<2, f32, TimeDomain> =
            Signal::new([vec![0.0, 1.0], vec![1.0, 0.0]], Some(100.0));
        input.clone()._test_clarify_methods(input);

        let input: Signal<2, cf32, FreqDomain> = Signal::new(
            [
                vec![cf32::new(1.0, 1.0), cf32::new(0.0, 0.0)],
                vec![cf32::new(0.0, 0.0), cf32::new(1.0, 1.0)],
            ],
            Some(100.0),
        );
        input.clone()._test_clarify_methods(input);

        let input: Signal<1, i16, TimeDomain> =
            Signal::new([vec![1, 2, 3, -1, -2, -3]], Some(100.0));
        input.clone()._test_clarify_methods(input);
    }

    #[test]
    fn check_interleave_method() {
        let input: Signal<2, f32, TimeDomain> = Signal::new([vec![0.0; 100], vec![1.0; 100]], None);
        let expected: Vec<f32> = (0..200).map(|i| (i % 2) as f32).collect();
        assert_eq!(input.interleave_samples(), expected)
    }
}

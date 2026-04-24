// Imports
use crate::{
    Seconds,
    audio::{AudioBuffer, AudioBufferSlice, DiscreteSignalUtils},
};
use itertools::Itertools;
use std::f32::consts::PI;

#[allow(private_bounds)]
pub trait DiscreteSignal<const C: usize>: Clone + DiscreteSignalUtils<C> {
    fn is_empty(&self) -> bool;

    fn sampling_rate(&self) -> Option<u32>;

    /// Clones data if `self` is a [`AudioBufferSlice`], returns `self` if already a [`AudioBuffer`].
    fn into_owned(self) -> AudioBuffer<C>;

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

    #[inline(always)]
    fn sampling_rate_f32(&self) -> Option<f32> {
        self.sampling_rate().map(|sr| sr as f32)
    }

    #[inline(always)]
    fn resolve_sampling_rate_pair(left: Option<u32>, right: Option<u32>) -> Option<u32> {
        debug_assert!(
            left.zip(right).is_none_or(|(l, r)| l == r),
            "Sampling rates mismatch"
        );
        left.or(right)
    }

    #[inline(always)]
    fn resolve_sampling_rate_many<I>(input: I) -> Option<u32>
    where
        I: IntoIterator<Item = Option<u32>>,
    {
        let mut iter = input.into_iter();
        let sampling_rate = iter.find_map(|e| e);

        debug_assert!(
            sampling_rate.is_none_or(|sr| iter.flatten().all(|other_sr| sr == other_sr)),
            "Sampling rates mismatch"
        );

        sampling_rate
    }

    fn seconds_to_index(&self, s: Seconds) -> usize {
        let Some(sampling_rate) = self.sampling_rate_f32() else {
            panic!("Sampling rate not defined");
        };
        (sampling_rate * s).floor() as usize
    }

    fn first_n_owned(self, n: usize) -> AudioBuffer<C> {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| v.truncate(n));
        out
    }

    fn skip_first_n_owned(self, n: usize) -> AudioBuffer<C> {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| {
            v.drain(0..n);
        });
        out
    }

    fn last_n_owned(self, n: usize) -> AudioBuffer<C> {
        let len = self.len();
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| {
            v.drain(0..(len - n));
        });
        out
    }

    fn skip_last_n_owned(self, n: usize) -> AudioBuffer<C> {
        let t_len = self.len() - n;
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|v| {
            v.truncate(t_len);
        });
        out
    }

    fn merge<T1, T2>(a: T1, b: T2) -> AudioBuffer<C>
    where
        T1: DiscreteSignal<C>,
        T2: DiscreteSignal<C>,
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(a.sampling_rate(), b.sampling_rate());

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

    fn merge_with<T>(self, other: T) -> AudioBuffer<C>
    where
        T: DiscreteSignal<C>,
    {
        Self::merge(self, other)
    }

    fn merge_many<T, I>(input: I) -> AudioBuffer<C>
    where
        T: DiscreteSignal<C>,
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter().collect_vec();

        let sampling_rate =
            Self::resolve_sampling_rate_many(input.iter().map(|e| e.sampling_rate()));

        let mut acc = input
            .swap_remove(
                input
                    .iter()
                    .position_max_by_key(DiscreteSignal::len)
                    .unwrap(),
            )
            .into_owned();

        for other in input {
            for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other._iter_cha()) {
                cha_acc.iter_mut().zip(cha_other).for_each(|(l, r)| *l += r)
            }
        }

        acc.with_sr_opt(sampling_rate)
    }

    fn apply<F>(self, mut op: F) -> AudioBuffer<C>
    where
        F: FnMut(f32) -> f32,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x = (&mut op)(*x)));
        out
    }

    fn apply_enumerate<F>(self, mut op: F) -> AudioBuffer<C>
    where
        F: FnMut((usize, f32)) -> f32,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|cha| {
            cha.iter_mut()
                .enumerate()
                .for_each(|(n, x)| *x = (&mut op)((n, *x)))
        });
        out
    }

    #[allow(non_snake_case)]
    /// The parameter M is such that the window is "M elements - center element - M elements".
    /// Of course, for elements near the boundary, their window size will be less than 2M+1 and
    /// not centered directly on themselves (i.e. near left, right side of the window is larger than left).
    fn apply_centered_window<F>(&self, M: usize, mut op: F) -> AudioBuffer<C>
    where
        F: for<'a> FnMut(&[f32]) -> f32,
    {
        let channels = self._map_cha(|cha| {
            let mut buffer: Vec<f32> = Vec::with_capacity(cha.len());
            for i in 0..cha.len() {
                let range = i.saturating_sub(M)..(i + M + 1).min(cha.len());
                let x = unsafe { cha.get_unchecked(range) };
                buffer.push((&mut op)(x))
            }
            buffer
        });
        AudioBuffer {
            channels,
            sampling_rate: self.sampling_rate(),
        }
    }

    fn abs(self) -> AudioBuffer<C> {
        self.apply(|x| x.abs())
    }

    fn get_abs_max(&self) -> f32 {
        self._iter_cha()
            .map(|cha| cha.iter().copied().map(f32::abs).reduce(f32::max).unwrap())
            .reduce(f32::max)
            .unwrap()
    }

    fn get_abs_max_index(&self) -> usize {
        self._iter_cha()
            .map(|cha| {
                cha.iter()
                    .copied()
                    .map(f32::abs)
                    .enumerate()
                    .reduce(|a, b| if a.1 > b.1 { a } else { b })
                    .unwrap()
            })
            .reduce(|a, b| if a.1 > b.1 { a } else { b })
            .unwrap()
            .0
    }

    fn normalize(self) -> AudioBuffer<C> {
        let mut out = self.into_owned();
        let abs_max = out.get_abs_max();
        if abs_max == 0.0 {
            return out;
        }
        let factor = 1.0 / abs_max;
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x *= factor));
        out
    }

    fn normalize_to(self, val: f32) -> AudioBuffer<C> {
        let mut out = self.into_owned();
        let abs_max = out.get_abs_max();
        if abs_max == 0.0 {
            return out;
        }
        let factor = val / abs_max;
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x *= factor));
        out
    }

    fn clamp(self) -> AudioBuffer<C> {
        let mut out = self.into_owned();
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0)));
        out
    }

    /// Normalize only if we have a value 'x' somewhere such that x ∉ [-1, 1]
    fn clamp_normalize(self) -> AudioBuffer<C> {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned();
        if abs_max > 1.0 {
            out.iter_cha_mut()
                .for_each(|cha| cha.iter_mut().for_each(|x| *x /= abs_max));
        }
        out
    }

    fn concatenate<T, I>(input: I) -> AudioBuffer<C>
    where
        T: DiscreteSignal<C>,
        I: IntoIterator<Item = T>,
    {
        let input = input.into_iter().collect_vec();
        let sampling_rate =
            Self::resolve_sampling_rate_many(input.iter().map(|e| e.sampling_rate()));

        let capacity: usize = input.iter().map(|buf| buf.len()).sum();
        let mut out = AudioBuffer::<C>::with_capacity(capacity, sampling_rate);

        for buf in input.into_iter() {
            for (cha, other) in out.channels.iter_mut().zip(buf._iter_cha()) {
                cha.extend_from_slice(other);
            }
        }

        out
    }

    // The resulting signal will have a length of `first.len() + second.len() - n_overlap`.
    fn crossfade<T1, T2>(first: T1, second: T2, n_overlap: usize) -> AudioBuffer<C>
    where
        T1: DiscreteSignal<C>,
        T2: DiscreteSignal<C>,
    {
        #[allow(non_snake_case)]
        let N = n_overlap;

        let (start, mid_1) = first._as_view().last_n_split(N);
        let (mid_2, end) = second._as_view().first_n_split(N);

        let transition = Self::merge(
            mid_1.apply_enumerate(&mut |(n, x_n)| {
                (PI * n as f32 / (2.0 * N as f32)).cos().powi(2) * x_n
            }),
            mid_2.apply_enumerate(&mut |(n, x_n)| {
                (PI * n as f32 / (2.0 * N as f32)).sin().powi(2) * x_n
            }),
        );

        Self::concatenate([start, transition._as_view(), end])
    }

    fn crossfade_concatenate<T, I>(input: I, n_overlap: usize) -> AudioBuffer<C>
    where
        T: DiscreteSignal<C>,
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter();
        let acc = input.next().expect("Input is empty").into_owned();
        input.fold(acc, |acc, other| Self::crossfade(acc, other, n_overlap))
    }

    fn pad_right(self, by: usize) -> AudioBuffer<C> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut().for_each(|buf| buf.extend(vec![0.0; by]));
        buf
    }

    fn pad_right_to_len(self, len: usize) -> AudioBuffer<C> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut().for_each(|buf| {
            let by = len.saturating_sub(buf.len());
            buf.extend(vec![0.0; by])
        });
        buf
    }

    fn pad_right_with(self, by: usize, val: f32) -> AudioBuffer<C> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut().for_each(|buf| buf.extend(vec![val; by]));
        buf
    }

    fn pad_right_with_last(self, by: usize) -> AudioBuffer<C> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut().for_each(|buf| {
            let val = *buf.last().unwrap_or(&0.0);
            buf.extend(vec![val; by])
        });
        buf
    }

    fn pad_right_with_last_to_len(self, len: usize) -> AudioBuffer<C> {
        let mut buf = self.into_owned();
        buf.iter_cha_mut().for_each(|buf| {
            let by = len.saturating_sub(buf.len());
            let val = *buf.last().unwrap_or(&0.0);
            buf.extend(vec![val; by])
        });
        buf
    }

    fn pad_left(self, by: usize) -> AudioBuffer<C> {
        AudioBuffer::new(
            self._map_cha(|cha| {
                let mut vec: Vec<f32> = vec![0.0; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn pad_left_with(self, by: usize, val: f32) -> AudioBuffer<C> {
        AudioBuffer::new(
            self._map_cha(|cha| {
                let mut vec: Vec<f32> = vec![val; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn pad_left_with_first(self, by: usize) -> AudioBuffer<C> {
        AudioBuffer::new(
            self._map_cha(|cha| {
                let val = *cha.first().unwrap_or(&0.0);
                let mut vec: Vec<f32> = vec![val; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn convolve<T, const C2: usize>(
        &self,
        h: T,
    ) -> <Self as super::DefinedConvolution<C, C2>>::ConvolutionOutput
    where
        Self: super::DefinedConvolution<C, C2>,
        T: DiscreteSignal<C2>,
    {
        self.convolve_with(h)
    }

    /// Similar to using `numpy.convolve(a, v, mode='same')`, we keep the output "aligned" with
    /// our input; left-biased in the case where the filter `h` has an even length.
    fn convolve_then_crop<T, const C2: usize, const C_OUT: usize>(&self, h: T) -> AudioBuffer<C_OUT>
    where
        Self: super::DefinedConvolution<C, C2, ConvolutionOutput = AudioBuffer<C_OUT>>,
        T: DiscreteSignal<C2>,
    {
        let filter_size = h.len();
        let y = self.convolve_with(h);

        #[allow(non_snake_case)]
        let M = filter_size / 2;

        if filter_size % 2 == 1 {
            // the ideal case, from M to N-M-1
            y.skip_first_n_owned(M).skip_last_n_owned(M)
        } else {
            eprintln!("Warning, filter length is even");
            y.skip_first_n_owned(M - 1).skip_last_n_owned(M)
        }
    }

    fn interleaved_samples_f32(&self) -> Vec<f32> {
        let size = self.len() * C;
        let mut out: Vec<f32> = vec![0.0; size];
        for (cha_idx, cha) in self._iter_cha().enumerate() {
            cha.iter()
                .copied()
                .enumerate()
                .for_each(|(i, x)| out[cha_idx + C * i] = x);
        }
        out
    }

    fn interleaved_samples_i16(&self) -> Vec<i16> {
        let size = self.len() * C;
        let mut out: Vec<i16> = vec![0; size];
        for (cha_idx, cha) in self._iter_cha().enumerate() {
            cha.iter()
                .copied()
                .enumerate()
                .for_each(|(i, x)| out[cha_idx + C * i] = (x * 32767.0).floor() as i16);
        }
        out
    }

    #[cfg(feature = "plot")]
    fn plot(&self) -> Vec<kuva::prelude::Plot> {
        use kuva::prelude::*;

        self._map_cha_enumerate(|i, cha| {
            let legend: String = match i {
                0 => "channel 0 (left ear)".to_string(),
                1 => "channel 1 (right ear)".to_string(),
                i => format!("channel {i}"),
            };
            let palette = Palette::wong();
            let color = &palette.colors()[i % palette.len()];

            LinePlot::new()
                .with_data(cha.iter().enumerate().map(|(x, y)| (x as f64, *y as f64)))
                .with_color(color)
                .with_legend(legend)
        })
        .into_iter()
        .map(Plot::from)
        .collect()
    }
}

impl<const C: usize> DiscreteSignal<C> for AudioBuffer<C> {
    fn is_empty(&self) -> bool {
        self.channels.first().map(Vec::is_empty).unwrap_or(true)
    }
    fn sampling_rate(&self) -> Option<u32> {
        self.sampling_rate
    }
    fn into_owned(self) -> AudioBuffer<C> {
        self
    }
}

impl<const C: usize> DiscreteSignal<C> for AudioBufferSlice<'_, C> {
    fn is_empty(&self) -> bool {
        self.channels
            .first()
            .map(|cha| cha.is_empty())
            .unwrap_or(true)
    }
    fn sampling_rate(&self) -> Option<u32> {
        self.sampling_rate
    }
    fn into_owned(self) -> AudioBuffer<C> {
        AudioBuffer::new(self.map_cha(|cha| cha.to_vec()), self.sampling_rate)
    }
}

impl<const C: usize, T> DiscreteSignal<C> for &T
where
    T: DiscreteSignal<C>,
{
    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }
    fn sampling_rate(&self) -> Option<u32> {
        (*self).sampling_rate()
    }
    fn into_owned(self) -> AudioBuffer<C> {
        (*self).clone().into_owned()
    }
}

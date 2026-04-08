// Imports
use crate::audio::{AudioBuffer, AudioBufferSlice, DiscreteSignalUtils};
use itertools::Itertools;
use std::f32::consts::PI;

#[allow(private_bounds)]
pub trait DiscreteSignal<const C: usize>: Clone + super::DiscreteSignalUtils {
    fn as_ref<'a>(&'a self) -> AudioBufferSlice<'a, C>;

    fn is_empty(&self) -> bool;

    fn len(&self) -> usize;

    fn sampling_rate(&self) -> Option<u32>;

    /// Clones data if `self` is a [`AudioBufferSlice`], returns `self` otherwise.
    fn into_owned(self) -> AudioBuffer<C>;

    fn cha(&self, c: usize) -> &[f32];

    /// Iterate through the channels
    fn iter_cha(&self) -> impl Iterator<Item = &[f32]>;

    /// Apply a function across channels
    fn map_cha<'a, T>(&'a self, f: impl FnMut(&'a [f32]) -> T) -> [T; C];

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

    fn slice<I>(&self, index: I) -> AudioBufferSlice<'_, C>
    where
        I: std::slice::SliceIndex<[f32], Output = [f32]> + Clone,
    {
        AudioBufferSlice::new(
            self.map_cha(|cha| &cha[index.clone()]),
            self.sampling_rate(),
        )
    }

    fn first_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(0..n)
    }

    /// The first element of the returned tuple contains the first n elements.
    fn first_n_split(&self, n: usize) -> (AudioBufferSlice<'_, C>, AudioBufferSlice<'_, C>) {
        (self.slice(0..n), self.slice(n..))
    }

    fn last_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(self.len() - n..)
    }

    /// The second element of the returned tuple contains the last n elements.
    fn last_n_split(&self, n: usize) -> (AudioBufferSlice<'_, C>, AudioBufferSlice<'_, C>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }

    fn merge<T1, T2>(a: T1, b: T2) -> AudioBuffer<C>
    where
        T1: DiscreteSignal<C>,
        T2: DiscreteSignal<C>,
    {
        let sampling_rate = Self::resolve_sampling_rate_pair(a.sampling_rate(), b.sampling_rate());

        let (mut acc, other) = if a.len() >= b.len() {
            (a.into_owned(), b.as_ref())
        } else {
            (b.into_owned(), a.as_ref())
        };

        for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other.iter_cha()) {
            cha_acc.iter_mut().zip(cha_other).for_each(|(l, r)| *l += r)
        }

        acc.with_sampling_rate(sampling_rate)
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
            for (cha_acc, cha_other) in acc.iter_cha_mut().zip(other.iter_cha()) {
                cha_acc.iter_mut().zip(cha_other).for_each(|(l, r)| *l += r)
            }
        }

        acc.with_sampling_rate(sampling_rate)
    }

    fn apply<F>(self, op: F) -> AudioBuffer<C>
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x = op(*x)));
        out
    }

    fn apply_with_context<F>(self, op: F) -> AudioBuffer<C>
    where
        F: Fn((usize, f32)) -> f32 + Copy,
    {
        let mut out = self.into_owned();
        out.iter_cha_mut().for_each(|cha| {
            cha.iter_mut()
                .enumerate()
                .for_each(|(n, x)| *x = op((n, *x)))
        });
        out
    }

    fn get_abs_max(&self) -> f32 {
        self.iter_cha()
            .map(|cha| cha.iter().copied().map(f32::abs).reduce(f32::max).unwrap())
            .reduce(f32::max)
            .unwrap()
    }

    fn normalize(self) -> AudioBuffer<C> {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned();
        out.iter_cha_mut()
            .for_each(|cha| cha.iter_mut().for_each(|x| *x /= abs_max));
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
            for (cha, other) in out.channels.iter_mut().zip(buf.iter_cha()) {
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

        let (start, mid_1) = first.last_n_split(N);
        let (mid_2, end) = second.first_n_split(N);

        let transition = Self::merge(
            mid_1.apply_with_context(|(n, x_n)| {
                (PI * n as f32 / (2.0 * N as f32)).cos().powi(2) * x_n
            }),
            mid_2.apply_with_context(|(n, x_n)| {
                (PI * n as f32 / (2.0 * N as f32)).sin().powi(2) * x_n
            }),
        );

        Self::concatenate([start, transition.as_ref(), end])
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
        buf.channels
            .iter_mut()
            .for_each(|buf| buf.extend(vec![0.0; by]));
        buf
    }

    fn pad_left(self, by: usize) -> AudioBuffer<C> {
        AudioBuffer::new(
            self.map_cha(|cha| {
                let mut vec: Vec<f32> = vec![0.0; by];
                vec.extend_from_slice(cha);
                vec
            }),
            self.sampling_rate(),
        )
    }

    fn as_blocks<'a>(&'a self, block_size: usize) -> Vec<AudioBufferSlice<'a, C>> {
        let rem = self.len() % block_size;
        let rhs = (rem > 0).then(|| self.last_n(rem));

        (0..self.len() / block_size)
            .map(|i| i * block_size..(i + 1) * block_size)
            .map(|i| self.slice(i))
            .chain(rhs)
            .collect_vec()
    }

    fn as_blocks_strict<'a>(
        &'a self,
        block_size: usize,
    ) -> (
        Vec<AudioBufferSlice<'a, C>>,
        Option<AudioBufferSlice<'a, C>>,
    ) {
        let lhs = (0..self.len() / block_size)
            .map(|i| i * block_size..(i + 1) * block_size)
            .map(|i| self.slice(i))
            .collect_vec();

        let rem = self.len() % block_size;
        let rhs = (rem > 0).then(|| self.last_n(rem));

        (lhs, rhs)
    }

    fn convolve<T, const C_OTHER: usize>(
        &self,
        other: &T,
    ) -> <Self as super::DefinedConvolution<C, C_OTHER>>::ConvolutionOutput
    where
        Self: super::DefinedConvolution<C, C_OTHER>,
        T: DiscreteSignal<C_OTHER>,
    {
        self.convolve_with(other)
    }

    fn interleaved_samples_f32(&self) -> Vec<f32> {
        let size = self.len() * C;
        let mut out: Vec<f32> = vec![0.0; size];
        for (cha_idx, cha) in self.iter_cha().enumerate() {
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
        for (cha_idx, cha) in self.iter_cha().enumerate() {
            cha.iter()
                .copied()
                .enumerate()
                .for_each(|(i, x)| out[cha_idx + C * i] = (x * 32767.0).floor() as i16);
        }
        out
    }
}

impl<const C: usize> DiscreteSignal<C> for AudioBuffer<C> {
    fn as_ref<'a>(&'a self) -> AudioBufferSlice<'a, C> {
        self.into()
    }
    fn is_empty(&self) -> bool {
        self.channels.first().map(Vec::is_empty).unwrap_or(true)
    }
    fn len(&self) -> usize {
        // Checks that all channels have the same length
        debug_assert!(self.map_cha(|cha| cha.len()).iter().all_equal());
        self.channels.first().map(Vec::len).unwrap_or(0)
    }
    fn sampling_rate(&self) -> Option<u32> {
        self.sampling_rate
    }
    fn into_owned(self) -> AudioBuffer<C> {
        self
    }
    fn cha(&self, c: usize) -> &[f32] {
        &self.channels[c]
    }
    fn iter_cha(&self) -> impl Iterator<Item = &[f32]> {
        self.channels.iter().map(Vec::as_slice)
    }
    fn map_cha<'a, T>(&'a self, mut f: impl FnMut(&'a [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
}

impl<const C: usize> DiscreteSignal<C> for AudioBufferSlice<'_, C> {
    fn as_ref<'a>(&'a self) -> AudioBufferSlice<'a, C> {
        *self
    }
    fn is_empty(&self) -> bool {
        self.channels
            .first()
            .map(|cha| cha.is_empty())
            .unwrap_or(true)
    }
    fn len(&self) -> usize {
        // Checks that all channels have the same length
        debug_assert!(self.map_cha(|cha| cha.len()).iter().all_equal());
        self.channels.first().map(|cha| cha.len()).unwrap_or(0)
    }
    fn sampling_rate(&self) -> Option<u32> {
        self.sampling_rate
    }
    fn into_owned(self) -> AudioBuffer<C> {
        AudioBuffer::new(self.map_cha(|cha| cha.to_vec()), self.sampling_rate)
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.channels[c]
    }
    fn iter_cha(&self) -> impl Iterator<Item = &[f32]> {
        self.channels.into_iter()
    }
    fn map_cha<'a, T>(&'a self, mut f: impl FnMut(&'a [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
}

impl<const C: usize, T> DiscreteSignal<C> for &T
where
    T: DiscreteSignal<C>,
{
    fn as_ref<'a>(&'a self) -> AudioBufferSlice<'a, C> {
        (*self).as_ref()
    }
    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }
    fn len(&self) -> usize {
        (*self).len()
    }
    fn sampling_rate(&self) -> Option<u32> {
        (*self).sampling_rate()
    }
    fn into_owned(self) -> AudioBuffer<C> {
        (*self).clone().into_owned()
    }
    fn cha(&self, c: usize) -> &[f32] {
        (*self).cha(c)
    }
    fn iter_cha(&self) -> impl Iterator<Item = &[f32]> {
        (*self).iter_cha()
    }
    fn map_cha<'a, T2>(&'a self, f: impl FnMut(&'a [f32]) -> T2) -> [T2; C] {
        (*self).map_cha(f)
    }
}

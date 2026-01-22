use itertools::Itertools;

use super::*;

pub trait AudioSignalCore<const C: usize>: Clone {
    fn as_ref<'a>(&'a self) -> ChannelBuffersSlice<'a, C>;

    fn is_empty(&self) -> bool;

    fn len(&self) -> usize;

    /// Returns the full slice of data contained by the specified channel.
    /// Note that `c` should be such that c < C to avoid panics.
    fn cha(&self, c: usize) -> &[f32];

    fn into_owned_buf(self) -> ChannelBuffers<C>;

    fn to_owned_buf(&self) -> ChannelBuffers<C> {
        self.clone().into_owned_buf()
    }

    fn slice<I>(&self, index: I) -> ChannelBuffersSlice<'_, C>
    where
        I: std::slice::SliceIndex<[f32], Output = [f32]> + Clone,
    {
        ChannelBuffersSlice(std::array::from_fn(|c| &self.cha(c)[index.clone()]))
    }

    fn first_n(&self, n: usize) -> ChannelBuffersSlice<'_, C> {
        self.slice(0..n)
    }

    /// The first element of the returned tuple contains the first n elements.
    fn first_n_split(&self, n: usize) -> (ChannelBuffersSlice<'_, C>, ChannelBuffersSlice<'_, C>) {
        (self.slice(0..n), self.slice(n..))
    }

    fn last_n(&self, n: usize) -> ChannelBuffersSlice<'_, C> {
        self.slice(self.len() - n..)
    }

    /// The second element of the returned tuple contains the last n elements.
    fn last_n_split(&self, n: usize) -> (ChannelBuffersSlice<'_, C>, ChannelBuffersSlice<'_, C>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }

    fn merge<T1, T2>(a: T1, b: T2) -> ChannelBuffers<C>
    where
        T1: AudioSignalCore<C>,
        T2: AudioSignalCore<C>,
    {
        let (mut acc, other) = if a.len() >= b.len() {
            (a.into_owned_buf(), b.as_ref())
        } else {
            (b.into_owned_buf(), a.as_ref())
        };

        for c in 0..C {
            acc.cha_mut(c)
                .iter_mut()
                .zip(other.cha(c).iter())
                .for_each(|(acc, x)| *acc += x);
        }

        acc
    }

    fn merge_with<T>(self, other: T) -> ChannelBuffers<C>
    where
        T: AudioSignalCore<C>,
    {
        Self::merge(self, other)
    }

    fn merge_many<T, I>(input: I) -> ChannelBuffers<C>
    where
        T: AudioSignalCore<C>,
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter().collect_vec();

        let mut acc = input
            .swap_remove(
                input
                    .iter()
                    .position_max_by_key(AudioSignalCore::len)
                    .unwrap(),
            )
            .into_owned_buf();

        for element in input {
            for c in 0..C {
                acc.cha_mut(c)
                    .iter_mut()
                    .zip(element.cha(c).iter())
                    .for_each(|(a, b)| *a += b);
            }
        }

        acc
    }

    fn apply<F>(self, op: F) -> ChannelBuffers<C>
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let mut out = self.into_owned_buf();
        for c in 0..C {
            out.cha_mut(c).iter_mut().for_each(|x| *x = op(*x));
        }
        out
    }

    fn apply_with_context<F>(self, op: F) -> ChannelBuffers<C>
    where
        F: Fn((usize, f32)) -> f32 + Copy,
    {
        let mut out = self.into_owned_buf();
        for c in 0..C {
            out.cha_mut(c)
                .iter_mut()
                .enumerate()
                .for_each(|(n, x)| *x = op((n, *x)));
        }
        out
    }

    fn get_abs_max(&self) -> f32 {
        std::array::from_fn::<_, C, _>(|c| {
            self.cha(c)
                .iter()
                .copied()
                .map(f32::abs)
                .reduce(f32::max)
                .unwrap()
        })
        .into_iter()
        .reduce(f32::max)
        .unwrap()
    }

    fn normalize(self) -> ChannelBuffers<C> {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned_buf();

        for c in 0..C {
            out.cha_mut(c).iter_mut().for_each(|x| *x /= abs_max);
        }

        out
    }

    fn clamp(self) -> ChannelBuffers<C> {
        let mut out = self.into_owned_buf();
        for c in 0..C {
            out.cha_mut(c)
                .iter_mut()
                .for_each(|x| *x = x.clamp(-1.0, 1.0));
        }
        out
    }

    fn clamp_normalize(self) -> ChannelBuffers<C> {
        let abs_max = self.get_abs_max();
        let mut out = self.into_owned_buf();
        if abs_max > 1.0 {
            for c in 0..C {
                out.cha_mut(c).iter_mut().for_each(|x| *x /= abs_max);
            }
        }
        out
    }

    fn concatenate<I, T>(input: I) -> ChannelBuffers<C>
    where
        I: AsRef<[T]>,
        T: AudioSignalCore<C>,
    {
        let input = input.as_ref();
        let capacity: usize = input.iter().map(|buf| buf.len()).sum();
        let mut out = ChannelBuffers::<C>::with_capacity(capacity);

        for buf in input.iter() {
            for c in 0..C {
                out.0[c].extend_from_slice(buf.cha(c));
            }
        }

        out
    }

    // The resulting signal will have a length of `first.len() + second.len() - n_overlap`.
    fn crossfade<T1, T2>(first: T1, second: T2, n_overlap: usize) -> ChannelBuffers<C>
    where
        T1: AudioSignalCore<C>,
        T2: AudioSignalCore<C>,
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

    fn crossfade_concatenate<T, I>(input: I, n_overlap: usize) -> ChannelBuffers<C>
    where
        T: AudioSignalCore<C>,
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter();
        let acc = input.next().expect("Empty input").into_owned_buf();
        input.fold(acc, |acc, other| Self::crossfade(acc, other, n_overlap))
    }
}

impl<const C: usize> AudioSignalCore<C> for ChannelBuffers<C> {
    fn as_ref<'a>(&'a self) -> ChannelBuffersSlice<'a, C> {
        self.into()
    }
    fn is_empty(&self) -> bool {
        self.0.first().map(Vec::is_empty).unwrap_or(true)
    }
    fn len(&self) -> usize {
        self.0.first().map(Vec::len).unwrap_or(0)
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.0[c].as_slice()
    }
    fn into_owned_buf(self) -> ChannelBuffers<C> {
        self
    }
}

impl<const C: usize> AudioSignalCore<C> for ChannelBuffersSlice<'_, C> {
    fn as_ref<'a>(&'a self) -> ChannelBuffersSlice<'a, C> {
        *self
    }
    fn is_empty(&self) -> bool {
        self.0.first().map(|arr| arr.is_empty()).unwrap_or(true)
    }
    fn len(&self) -> usize {
        self.0.first().map(|arr| arr.len()).unwrap_or(0)
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.0[c]
    }
    fn into_owned_buf(self) -> ChannelBuffers<C> {
        std::array::from_fn(|c| self.cha(c).to_vec()).into()
    }
}

impl<const C: usize> AudioSignalCore<C> for AudioBuffer<C> {
    fn as_ref<'a>(&'a self) -> ChannelBuffersSlice<'a, C> {
        self.into()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.inner.cha(c)
    }
    fn into_owned_buf(self) -> ChannelBuffers<C> {
        self.inner
    }
}

impl<const C: usize, T> AudioSignalCore<C> for &T
where
    T: AudioSignalCore<C>,
{
    fn as_ref<'a>(&'a self) -> ChannelBuffersSlice<'a, C> {
        (*self).as_ref()
    }
    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }
    fn len(&self) -> usize {
        (*self).len()
    }
    fn cha(&self, c: usize) -> &[f32] {
        (*self).cha(c)
    }
    fn into_owned_buf(self) -> ChannelBuffers<C> {
        (*self).to_owned_buf()
    }
}

pub trait AudioSignalConvolution {
    fn convolve<T, const C1: usize, const C2: usize, const C3: usize>(
        &self,
        other: &T,
    ) -> ChannelBuffers<C3>
    where
        Self: AudioSignalCore<C1> + DefinedConvolution<C1, C2, C3>,
        T: AudioSignalCore<C2>,
    {
        self.convolve_with(other)
    }
}

impl AudioSignalConvolution for ChannelBuffers<1> {}
impl AudioSignalConvolution for ChannelBuffers<2> {}
impl AudioSignalConvolution for ChannelBuffersSlice<'_, 1> {}
impl AudioSignalConvolution for ChannelBuffersSlice<'_, 2> {}
impl AudioSignalConvolution for AudioBuffer<1> {}
impl AudioSignalConvolution for AudioBuffer<2> {}

pub trait DefinedConvolution<const C1: usize, const C2: usize, const C3: usize> {
    fn convolve_with<T2>(&self, other: &T2) -> ChannelBuffers<C3>
    where
        T2: AudioSignalCore<C2>;
}

impl<T1> DefinedConvolution<1, 1, 1> for T1
where
    T1: AudioSignalCore<1>,
{
    fn convolve_with<T2>(&self, other: &T2) -> ChannelBuffers<1>
    where
        T2: AudioSignalCore<1>,
    {
        ChannelBuffers([_convolve(self.cha(0), other.cha(0))])
    }
}

impl<T1> DefinedConvolution<1, 2, 2> for T1
where
    T1: AudioSignalCore<1>,
{
    fn convolve_with<T2>(&self, other: &T2) -> ChannelBuffers<2>
    where
        T2: AudioSignalCore<2>,
    {
        ChannelBuffers(std::array::from_fn(|c| {
            _convolve(self.cha(0), other.cha(c))
        }))
    }
}

impl<T1> DefinedConvolution<2, 1, 2> for T1
where
    T1: AudioSignalCore<2>,
{
    fn convolve_with<T2>(&self, other: &T2) -> ChannelBuffers<2>
    where
        T2: AudioSignalCore<1>,
    {
        ChannelBuffers(std::array::from_fn(|c| {
            _convolve(self.cha(c), other.cha(0))
        }))
    }
}

impl<T1> DefinedConvolution<2, 2, 2> for T1
where
    T1: AudioSignalCore<2>,
{
    fn convolve_with<T2>(&self, other: &T2) -> ChannelBuffers<2>
    where
        T2: AudioSignalCore<2>,
    {
        ChannelBuffers(std::array::from_fn(|c| {
            _convolve(self.cha(c), other.cha(c))
        }))
    }
}

fn _convolve(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::with_capacity(0);
    }

    let max_n = lhs.len() + rhs.len() - 1;
    let mut out: Vec<f32> = Vec::with_capacity(max_n);

    unsafe {
        let c: *mut f32 = out.as_mut_ptr();

        for n in 0..max_n {
            let mut val: f32 = 0.0;

            // 0..len(a) ∩ n-len(b)+1..n+1
            let k_range = std::ops::Range::<usize> {
                start: (n + 1).saturating_sub(rhs.len()), // equivalent to max(0, n - rhs.len + 1)
                end: usize::min(lhs.len(), n + 1),
            };

            //println!("n={n}, k ∈ {k_range:?}");

            for k in k_range {
                val += lhs.get_unchecked(k) * rhs.get_unchecked(n - k);
            }
            c.add(n).write(val);
        }

        out.set_len(max_n);
    }

    out
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn convolution_test() {
        let a: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![7.0, 8.0];

        assert!(
            _convolve(&a, &b)
                .into_iter()
                .zip_eq(vec![0.0, 7.0, 22.0, 37.0, 24.0])
                .all(|(a, b)| approx::relative_eq!(a, b))
        );

        assert!(
            _convolve(&b, &a)
                .into_iter()
                .zip_eq(vec![0.0, 7.0, 22.0, 37.0, 24.0])
                .all(|(a, b)| approx::relative_eq!(a, b))
        );

        let a: Vec<f32> = vec![53.0, 7.0, 19.0];
        let b: Vec<f32> = vec![-4.0, 10.0, -1.2, 8.0, 1.0, 1.0];

        assert!(
            _convolve(&a, &b)
                .into_iter()
                .zip_eq(vec![-212.0, 502.0, -69.6, 605.6, 86.2, 212.0, 26.0, 19.0])
                .all(|(a, b)| approx::relative_eq!(a, b))
        );
    }
}

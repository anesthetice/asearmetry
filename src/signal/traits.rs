use super::*;

pub trait AudioSignalCore<const C: usize> {
    fn as_ref<'a>(&'a self) -> ChannelBuffersSlice<'a, C>
    where
        ChannelBuffersSlice<'a, C>: From<&'a Self>,
    {
        self.into()
    }

    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn cha(&self, c: usize) -> &[f32];

    fn get<'a, I>(&'a self, index: I) -> ChannelBuffersSlice<'a, C>
    where
        I: std::slice::SliceIndex<[f32], Output = &'a [f32]> + Copy,
    {
        ChannelBuffersSlice(std::array::from_fn(|i| self.cha(i)[index]))
    }
}

impl<const C: usize> AudioSignalCore<C> for ChannelBuffers<C> {
    fn is_empty(&self) -> bool {
        self.0.first().map(Vec::is_empty).unwrap_or(true)
    }
    fn len(&self) -> usize {
        self.0.first().map(Vec::len).unwrap_or(0)
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.0[c].as_slice()
    }
}

impl<const C: usize> AudioSignalCore<C> for ChannelBuffersSlice<'_, C> {
    fn is_empty(&self) -> bool {
        self.0.first().map(|arr| arr.is_empty()).unwrap_or(true)
    }
    fn len(&self) -> usize {
        self.0.first().map(|arr| arr.len()).unwrap_or(0)
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.0[c]
    }
}

impl<const C: usize> AudioSignalCore<C> for AudioBuffer<C> {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn cha(&self, c: usize) -> &[f32] {
        self.inner.cha(c)
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
        ChannelBuffers(std::array::from_fn(|i| {
            _convolve(self.cha(0), other.cha(i))
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
        ChannelBuffers(std::array::from_fn(|i| {
            _convolve(self.cha(i), other.cha(0))
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
        ChannelBuffers(std::array::from_fn(|i| {
            _convolve(self.cha(i), other.cha(i))
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

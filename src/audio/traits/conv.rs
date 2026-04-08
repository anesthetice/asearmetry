// Imports
use crate::audio::{AudioBuffer, DiscreteSignal};

pub trait DefinedConvolution<const C1: usize, const C2: usize> {
    type ConvolutionOutput;
    fn convolve_with<T2>(&self, other: T2) -> Self::ConvolutionOutput
    where
        T2: DiscreteSignal<C2>;
}

impl<T1> DefinedConvolution<1, 1> for T1
where
    T1: DiscreteSignal<1>,
{
    type ConvolutionOutput = AudioBuffer<1>;
    fn convolve_with<T2>(&self, other: T2) -> Self::ConvolutionOutput
    where
        T2: DiscreteSignal<1>,
    {
        AudioBuffer::new(
            [_convolve(self.cha_uc(0), other.cha_uc(0))],
            Self::resolve_sampling_rate_pair(self.sampling_rate(), other.sampling_rate()),
        )
    }
}

impl<T1> DefinedConvolution<1, 2> for T1
where
    T1: DiscreteSignal<1>,
{
    type ConvolutionOutput = AudioBuffer<2>;
    fn convolve_with<T2>(&self, other: T2) -> Self::ConvolutionOutput
    where
        T2: DiscreteSignal<2>,
    {
        AudioBuffer::new(
            std::array::from_fn(|c| _convolve(self.cha_uc(0), other.cha_uc(c))),
            Self::resolve_sampling_rate_pair(self.sampling_rate(), other.sampling_rate()),
        )
    }
}

impl<T1> DefinedConvolution<2, 1> for T1
where
    T1: DiscreteSignal<2>,
{
    type ConvolutionOutput = AudioBuffer<2>;
    fn convolve_with<T2>(&self, other: T2) -> Self::ConvolutionOutput
    where
        T2: DiscreteSignal<1>,
    {
        AudioBuffer::new(
            std::array::from_fn(|c| _convolve(self.cha_uc(c), other.cha_uc(0))),
            Self::resolve_sampling_rate_pair(self.sampling_rate(), other.sampling_rate()),
        )
    }
}

impl<T1> DefinedConvolution<2, 2> for T1
where
    T1: DiscreteSignal<2>,
{
    type ConvolutionOutput = AudioBuffer<2>;
    fn convolve_with<T2>(&self, other: T2) -> Self::ConvolutionOutput
    where
        T2: DiscreteSignal<2>,
    {
        AudioBuffer::new(
            std::array::from_fn(|c| _convolve(self.cha_uc(c), other.cha_uc(c))),
            Self::resolve_sampling_rate_pair(self.sampling_rate(), other.sampling_rate()),
        )
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

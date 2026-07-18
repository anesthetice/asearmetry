/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::signal::{DSP, Domain, Sample, Signal};

pub trait DefinedLtiConvolution<
    const C1: usize,
    const C2: usize,
    const C3: usize,
    S: Sample,
    D: Domain,
>
{
    type Output: DSP<C3, S, D>;
    fn convolve_with<T2>(&self, other: T2) -> Self::Output
    where
        T2: DSP<C2, S, D>;
}

impl<const C: usize, S: Sample, D: Domain, T1> DefinedLtiConvolution<C, C, C, S, D> for T1
where
    T1: DSP<C, S, D>,
{
    type Output = Signal<C, S, D>;
    fn convolve_with<T2>(&self, other: T2) -> Self::Output
    where
        T2: DSP<C, S, D>,
    {
        Signal::new(
            self._map_cha_enumerate(|c, samples| _convolve_lti(samples, other._cha(c))),
            Self::resolve_sampling_rate_pair(self.sampling_rate(), other.sampling_rate()),
        )
    }
}

impl<T1, S: Sample, D: Domain> DefinedLtiConvolution<1, 2, 2, S, D> for T1
where
    T1: DSP<1, S, D>,
{
    type Output = Signal<2, S, D>;
    fn convolve_with<T2>(&self, other: T2) -> Self::Output
    where
        T2: DSP<2, S, D>,
    {
        Signal::new(
            std::array::from_fn(|c| _convolve_lti(self._cha(0), other._cha(c))),
            Self::resolve_sampling_rate_pair(self.sampling_rate(), other.sampling_rate()),
        )
    }
}

impl<T1, S: Sample, D: Domain> DefinedLtiConvolution<2, 1, 2, S, D> for T1
where
    T1: DSP<2, S, D>,
{
    type Output = Signal<2, S, D>;
    fn convolve_with<T2>(&self, other: T2) -> Self::Output
    where
        T2: DSP<1, S, D>,
    {
        other.convolve_with(self)
    }
}

pub fn _convolve_lti<V>(lhs: &[V], rhs: &[V]) -> Vec<V>
where
    V: Copy + num_traits::Num + num_traits::NumAssign,
{
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::with_capacity(0);
    }

    let max_n = lhs.len() + rhs.len() - 1;
    let mut out: Vec<V> = Vec::with_capacity(max_n);

    unsafe {
        let c: *mut V = out.as_mut_ptr();

        for n in 0..max_n {
            let mut val = V::zero();

            // 0..len(a) ∩ n-len(b)+1..n+1
            let k_range = std::ops::Range::<usize> {
                start: (n + 1).saturating_sub(rhs.len()), // equivalent to max(0, n - rhs.len + 1)
                end: usize::min(lhs.len(), n + 1),
            };

            for k in k_range {
                val += lhs.get_unchecked(k).mul(*rhs.get_unchecked(n - k));
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
            _convolve_lti(&a, &b)
                .into_iter()
                .zip_eq(vec![0.0, 7.0, 22.0, 37.0, 24.0])
                .all(|(a, b)| approx::relative_eq!(a, b))
        );

        assert!(
            _convolve_lti(&b, &a)
                .into_iter()
                .zip_eq(vec![0.0, 7.0, 22.0, 37.0, 24.0])
                .all(|(a, b)| approx::relative_eq!(a, b))
        );

        let a: Vec<f32> = vec![53.0, 7.0, 19.0];
        let b: Vec<f32> = vec![-4.0, 10.0, -1.2, 8.0, 1.0, 1.0];

        assert!(
            _convolve_lti(&a, &b)
                .into_iter()
                .zip_eq(vec![-212.0, 502.0, -69.6, 605.6, 86.2, 212.0, 26.0, 19.0])
                .all(|(a, b)| approx::relative_eq!(a, b))
        );
    }
}

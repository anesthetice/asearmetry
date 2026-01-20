use super::*;

/// Only works for Signal<1> and Signal<2> for now, I could use nightly features to expand this in the future if need be.
pub trait ImplConvolution<'a, const C_OUT: usize> {
    fn convolve_with(&self, other: impl Into<AudioFrameRef<'a, C_OUT>>) -> AudioFrame<C_OUT>;
}

impl<'a, const C_OUT: usize> ImplConvolution<'a, C_OUT> for Signal<1> {
    fn convolve_with(&self, other: impl Into<AudioFrameRef<'a, C_OUT>>) -> AudioFrame<C_OUT> {
        let rhs = other.into();
        AudioFrame::from_unchecked(std::array::from_fn(|i| _convolve(self.cha(0), rhs.cha(i))))
    }
}

impl<'a, const C_OUT: usize> ImplConvolution<'a, C_OUT> for AudioFrameRef<'a, 1> {
    fn convolve_with(&self, other: impl Into<AudioFrameRef<'a, C_OUT>>) -> AudioFrame<C_OUT> {
        let rhs = other.into();
        AudioFrame::from_unchecked(std::array::from_fn(|i| _convolve(self.cha(0), rhs.cha(i))))
    }
}

impl<'a> ImplConvolution<'a, 2> for Signal<2> {
    fn convolve_with(&self, other: impl Into<AudioFrameRef<'a, 2>>) -> AudioFrame<2> {
        let rhs = other.into();
        AudioFrame::from_unchecked(std::array::from_fn(|i| _convolve(self.cha(i), rhs.cha(i))))
    }
}

impl<'a> ImplConvolution<'a, 2> for AudioFrameRef<'a, 2> {
    fn convolve_with(&self, other: impl Into<AudioFrameRef<'a, 2>>) -> AudioFrame<2> {
        let rhs = other.into();
        AudioFrame::from_unchecked(std::array::from_fn(|i| _convolve(self.cha(i), rhs.cha(i))))
    }
}

impl Signal<1> {
    pub fn convolve<'a, const CHANNELS: usize>(
        &self,
        other: impl Into<AudioFrameRef<'a, CHANNELS>>,
    ) -> AudioFrame<CHANNELS> {
        let other = other.into();
        AudioFrame::from_unchecked(std::array::from_fn(|i| {
            _convolve(self.cha(0), other.cha(i))
        }))
    }
}

impl<'a> AudioFrameRef<'a, 1> {
    pub fn convolve<'b, const CHANNELS: usize>(
        &self,
        other: impl Into<AudioFrameRef<'b, CHANNELS>>,
    ) -> AudioFrame<CHANNELS> {
        let other = other.into();
        AudioFrame::from_unchecked(std::array::from_fn(|i| {
            _convolve(self.cha(0), other.cha(i))
        }))
    }
}

impl Signal<2> {
    pub fn convolve<'a>(&'a self, other: &impl ImplConvolution<'a, 2>) -> AudioFrame<2> {
        other.convolve_with(self)
    }
}

impl<'a> AudioFrameRef<'a, 2> {
    pub fn convolve(&'a self, other: impl ImplConvolution<'a, 2>) -> AudioFrame<2> {
        other.convolve_with(*self)
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

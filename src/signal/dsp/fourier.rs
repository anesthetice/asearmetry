/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#![allow(non_snake_case)]

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{f32::consts::TAU, ops::MulAssign};

#[allow(non_camel_case_types)]
type cf32 = num_complex::Complex32;

pub fn _dft(s: &[f32]) -> Vec<cf32> {
    // Math reminder:
    //  The DTFT (discrete-time Fourier transform) of a discrete signal is 2π-periodic within (-π, +π],
    //  and the DFT (discrete Fourier transform) is simply a sampling of the the DTFT,
    //  with F[m] = DTFT{f}(2π ⋅ m/N)   where N is the length of the signal.
    //
    // As such, we only need to compute F[m] for m in {0, …, N-1} to known all possible F[m] values.
    //
    // But we can go one step further. For our application, we only actually need to compute
    // {0, …, ⌊N/2⌋}, as our signal in the time domain only contains real (thus non-complex) values.
    // As such, F[m] = DFT{f}[m] is even symmetric, meaning F[-m] = F[m], which we can write as
    // F[m] = F*[-m mod N]. For m ∈ {0, …, N-1} we can write F[m] = F*[N-m], therefore:
    // - F[0] gives no extra info due to aforementioned symmetry
    // - if F[1] is known, then so is F[N-1] = F*[1]
    // - if F[2] is known, then so is F[N-2] = F*[2]
    // - and so on…
    //
    // This further constricts the prior range of {0, …, N-1} to {0, …, ⌈(N-1)/2⌉}, or equivalently {0, …, ⌊N/2⌋}
    // So we only need to compute ⌊N/2⌋ + 1 values in total. Note that we will store all {0, …, ⌊N/2⌋} values however
    // to simplify things (a real pain to keep track of things otherwise).
    let N_t = s.len();
    let N_f = (s.len() / 2) + 1;
    let mut S: Vec<cf32> = Vec::with_capacity(N_t);

    for m in 0..N_f {
        let val = s
            .iter()
            .copied()
            .enumerate()
            .map(|(n, s_n)| {
                s_n * cf32::exp(cf32::new(0.0, -TAU * n as f32 * m as f32 / N_t as f32))
            })
            .sum();

        S.push(val);
    }

    for m in N_f..N_t {
        S.push(S[N_t - m].conj())
    }

    S
}

pub fn _dft_rayon(s: &[f32]) -> Vec<cf32> {
    let N_t = s.len();
    let N_f = (s.len() / 2) + 1;
    let mut S: Vec<cf32> = Vec::with_capacity(N_t);

    (0..N_f)
        .into_par_iter()
        .map(|m| {
            s.iter()
                .copied()
                .enumerate()
                .map(|(n, s_n)| {
                    s_n * cf32::exp(cf32::new(0.0, -TAU * n as f32 * m as f32 / N_t as f32))
                })
                .sum::<cf32>()
        })
        .collect_into_vec(&mut S);

    for m in N_f..N_t {
        S.push(S[N_t - m].conj())
    }

    S
}

pub fn _fft(mut s: Vec<f32>) -> Vec<cf32> {
    let N_t = s.len();
    let N_f = (s.len() / 2) + 1;

    let mut real_planner = realfft::RealFftPlanner::new();
    let r2c = real_planner.plan_fft_forward(s.len());
    let mut S: Vec<cf32> = r2c.make_output_vec();
    r2c.process(&mut s, &mut S).unwrap();

    S.reserve_exact(N_t - S.len());
    for m in N_f..N_t {
        S.push(S[N_t - m].conj())
    }
    S
}

/// Expects the parameter `S` to be the entire DFT.
pub fn _idft(S: &[cf32]) -> Vec<f32> {
    let N = S.len();
    let mut s: Vec<f32> = Vec::with_capacity(N);

    let factor = 1.0 / N as f32;

    for n in 0..N {
        let val = S
            .iter()
            .copied()
            .enumerate()
            .map(|(m, S_m)| S_m * cf32::exp(cf32::new(0.0, TAU * n as f32 * m as f32 / N as f32)))
            .sum::<cf32>()
            .re
            * factor;

        s.push(val);
    }

    s
}

/// Expects the parameter `S` to be the entire DFT.
pub fn _idft_rayon(S: &[cf32]) -> Vec<f32> {
    let N = S.len();
    let mut s: Vec<f32> = Vec::with_capacity(N);

    let factor = 1.0 / N as f32;

    (0..N)
        .into_par_iter()
        .map(|n| {
            S.iter()
                .copied()
                .enumerate()
                .map(|(m, S_m)| {
                    S_m * cf32::exp(cf32::new(0.0, TAU * n as f32 * m as f32 / N as f32))
                })
                .sum::<cf32>()
                .re
                * factor
        })
        .collect_into_vec(&mut s);

    s
}

/// Expects the parameter `S` to be the "halved" DFT. As such,
/// `S` should be truncated to ⌊N/2⌋ + 1 beforehand. This is because
/// `realfft` expects "halved" representation of the DFT as input.
pub fn _ifft(mut S: Vec<cf32>, N: usize) -> Vec<f32> {
    let mut real_planner = realfft::RealFftPlanner::new();
    let c2r = real_planner.plan_fft_inverse(N);
    let mut s: Vec<f32> = c2r.make_output_vec();
    c2r.process(&mut S, &mut s).unwrap();

    let factor = 1.0 / N as f32;
    s.iter_mut().for_each(|x| x.mul_assign(factor));
    s
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    fn f32_array_abs_diff_eq(a: &[f32], b: &[f32]) -> bool {
        a.iter()
            .zip_eq(b)
            .all(|(a, b)| approx::abs_diff_eq!(a, b, epsilon = 1e-6))
    }

    fn cf32_array_abs_diff_eq(a: &[cf32], b: &[cf32]) -> bool {
        a.iter().zip_eq(b).all(|(a, b)| {
            approx::abs_diff_eq!(a.re, b.re, epsilon = 1e-6)
                && approx::abs_diff_eq!(a.im, b.im, epsilon = 1e-6)
        })
    }

    #[test]
    fn dft_and_idft() {
        let s: &'static [f32] = &[1.0, 2.0, 3.0];
        let N = s.len();

        let mut dft_expected_output = vec![
            cf32::new(6.0, 0.0),
            cf32::new(-1.5, 0.8660254),
            cf32::new(-1.5, -0.8660254),
        ];

        assert!(cf32_array_abs_diff_eq(&_dft(s), &dft_expected_output));
        assert!(cf32_array_abs_diff_eq(&_dft_rayon(s), &dft_expected_output));
        assert!(cf32_array_abs_diff_eq(
            &_fft(s.to_vec()),
            &dft_expected_output
        ));

        assert!(f32_array_abs_diff_eq(&_idft(&dft_expected_output), s));
        assert!(f32_array_abs_diff_eq(&_idft_rayon(&dft_expected_output), s));

        dft_expected_output.truncate((N / 2) + 1);
        assert!(f32_array_abs_diff_eq(&_ifft(dft_expected_output, N), s));
    }
}

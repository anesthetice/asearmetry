/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::signal::{DSP, Domain, Sample, Signal};

pub trait DefinedLtvConvolution<S: Sample, D: Domain> {
    fn convolve_ltv_with<STATE, H, F1, F2>(
        &self,
        h: LtvFilter<STATE, H, F1, F2, S>,
    ) -> Signal<1, S, D>
    where
        F1: Fn(usize, &mut STATE, &mut Option<H>),
        F2: for<'a> Fn(&'a H) -> &'a [S];
}

impl<T1, S: Sample, D: Domain> DefinedLtvConvolution<S, D> for T1
where
    T1: DSP<1, S, D>,
{
    fn convolve_ltv_with<STATE, H, F1, F2>(
        &self,
        h: LtvFilter<STATE, H, F1, F2, S>,
    ) -> Signal<1, S, D>
    where
        F1: Fn(usize, &mut STATE, &mut Option<H>),
        F2: for<'a> Fn(&'a H) -> &'a [S],
    {
        Signal::new([_convolve_ltv(self._cha(0), h)], self.sampling_rate())
    }
}

#[derive(bon::Builder)]
#[builder(start_fn = new_with_state)]
pub struct LtvFilter<STATE, H, F1, F2, V: std::ops::Mul + num_traits::Zero>
where
    F1: Fn(usize, &mut STATE, &mut Option<H>),
    F2: for<'a> Fn(&'a H) -> &'a [V],
{
    #[builder(start_fn)]
    state: STATE,
    filter: Option<H>,
    update_fn: F1,
    get_fn: F2,
}

impl<STATE, H, F1, F2, V: std::ops::Mul + num_traits::Zero> LtvFilter<STATE, H, F1, F2, V>
where
    F1: Fn(usize, &mut STATE, &mut Option<H>),
    F2: for<'a> Fn(&'a H) -> &'a [V],
{
    pub fn update(&mut self, n: usize) {
        (self.update_fn)(n, &mut self.state, &mut self.filter);
    }
    pub fn get(&self) -> &[V] {
        let filter = self.filter.as_ref().expect("No filter present");
        (self.get_fn)(filter)
    }
}

pub fn _convolve_ltv<STATE, H, F1, F2, V>(x: &[V], mut h: LtvFilter<STATE, H, F1, F2, V>) -> Vec<V>
where
    V: Copy + num_traits::Num + num_traits::NumAssign,
    F1: Fn(usize, &mut STATE, &mut Option<H>),
    F2: for<'a> Fn(&'a H) -> &'a [V],
{
    if x.is_empty() {
        return Vec::with_capacity(0);
    }

    // This assumes that h[n, k] = 0 ∀k if n ∉ {0, ..., x.len - 1}
    let max_n = x.len();
    let mut out = Vec::with_capacity(max_n);

    unsafe {
        let y_ptr: *mut V = out.as_mut_ptr();
        let mut y_val = V::zero();

        for n in 0..max_n {
            h.update(n);
            let h_n = h.get(); // time-varying impulse response at "time" n
            let h_n_len = h_n.len();

            // k ∈ [max(0, n - h_n_len + 1), min(n+1, x.len))
            let k_start = (n + 1).saturating_sub(h_n_len);
            let k_end = usize::min(x.len(), n + 1);
            for k in k_start..k_end {
                y_val += x.get_unchecked(k).mul(*h_n.get_unchecked(n - k));
            }

            y_ptr.add(n).write(y_val);
            y_val = V::zero();
        }

        out.set_len(max_n);
    }

    out
}

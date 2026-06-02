// Imports
use crate::audio::{AudioBuffer, DiscreteSignal};

pub trait DefinedLtvConvolution {
    fn convolve_ltv_with<S, H, F1, F2>(&self, h: LtvFilter<S, H, F1, F2>) -> AudioBuffer<1>
    where
        F1: Fn(usize, &mut S, &mut H),
        F2: for<'a> Fn(&'a H) -> &'a [f32];
}

impl<T1> DefinedLtvConvolution for T1
where
    T1: DiscreteSignal<1>,
{
    fn convolve_ltv_with<S, H, F1, F2>(&self, h: LtvFilter<S, H, F1, F2>) -> AudioBuffer<1>
    where
        F1: Fn(usize, &mut S, &mut H),
        F2: for<'a> Fn(&'a H) -> &'a [f32],
    {
        AudioBuffer::new_mono(_convolve_ltv(self._cha(0), h), self.sampling_rate())
    }
}

pub struct LtvFilter<S, H, F1, F2>
where
    F1: Fn(usize, &mut S, &mut H),
    F2: for<'a> Fn(&'a H) -> &'a [f32],
{
    state: S,
    filter: H,
    update_fn: F1,
    get_fn: F2,
}

impl<S, H, F1, F2> LtvFilter<S, H, F1, F2>
where
    F1: Fn(usize, &mut S, &mut H),
    F2: for<'a> Fn(&'a H) -> &'a [f32],
{
    pub fn new(
        // The filter will immediately be updated, you can just return a default if
        // the next filter doesn't depend on the previous one.
        init: impl FnOnce() -> (S, H),
        update_fn: F1,
        get_fn: F2,
    ) -> Self {
        let (mut state, mut filter) = init();
        update_fn(0, &mut state, &mut filter);

        Self {
            state,
            filter,
            update_fn,
            get_fn,
        }
    }
    pub fn update(&mut self, n: usize) {
        (self.update_fn)(n, &mut self.state, &mut self.filter);
    }
    pub fn get(&self) -> &[f32] {
        (self.get_fn)(&self.filter)
    }
}

pub fn _convolve_ltv<S, H, F1, F2>(x: &[f32], mut h: LtvFilter<S, H, F1, F2>) -> Vec<f32>
where
    F1: Fn(usize, &mut S, &mut H),
    F2: for<'a> Fn(&'a H) -> &'a [f32],
{
    if x.is_empty() {
        return Vec::with_capacity(0);
    }

    // This assumes that h[n, k] = 0 ∀k if n ∉ {0, ..., x.len - 1}
    let max_n = x.len();
    let mut out = Vec::with_capacity(max_n);

    unsafe {
        let y_ptr: *mut f32 = out.as_mut_ptr();
        let mut y_val: f32 = 0.0;

        for n in 0..max_n {
            h.update(n);
            let h_n = h.get(); // time-varying impulse response at "time" n
            let h_n_len = h_n.len();

            // k ∈ [max(0, n - h_n_len + 1), min(n+1, x.len))
            let k_start = (n + 1).saturating_sub(h_n_len);
            let k_end = usize::min(x.len(), n + 1);
            for k in k_start..k_end {
                y_val += x.get_unchecked(k) * h_n.get_unchecked(n - k);
            }

            y_ptr.add(n).write(y_val);
            y_val = 0.0;
        }

        out.set_len(max_n);
    }

    out
}

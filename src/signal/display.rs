/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use crate::signal::{DSP, Domain, Sample, Signal, SignalSlice};
use core::fmt::Write;
use indoc::writedoc;
use itertools::Itertools;
use num_complex::Complex32;

impl<const C: usize, S: Sample + FloatDisplay, D: Domain> core::fmt::Debug for Signal<C, S, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        display_debug_impl("Signal", self, f)
    }
}

impl<const C: usize, S: Sample + FloatDisplay, D: Domain> core::fmt::Display for Signal<C, S, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self, f)
    }
}

impl<const C: usize, S: Sample + FloatDisplay, D: Domain> core::fmt::Debug
    for SignalSlice<'_, C, S, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        display_debug_impl("SignalSlice", self, f)
    }
}

impl<const C: usize, S: Sample + FloatDisplay, D: Domain> core::fmt::Display
    for SignalSlice<'_, C, S, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self, f)
    }
}

fn display_debug_impl<const C: usize, S: Sample + FloatDisplay, D: Domain, T>(
    name: &str,
    s: T,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result
where
    T: DSP<C, S, D>,
{
    let n_displayed_per_line: usize = <S as FloatDisplay>::CHUNK_SIZE;

    writedoc! {
    f,
    "
        {name} {{
            domain: {dom}
            sampling rate: {sr}
            number of channels: {C}
            samples per channel: {len}

        {channels}
        }}
    ",
    dom = s.domain(),
    sr = s.sampling_rate().map_or_else(|| "None".to_string(), |x| x.to_string()),
    len = s.len(),
    channels = s._iter_cha().enumerate().map(|(i, cha)| {
        let spine: &str = if f.alternate() {
            "     "
        } else {
            "    ┃ "
        };
        format!(
            "    ┏━━ channel {i}{opt} ━━━━\n{data}",
            opt = { if C!=2 {""} else if i==0 {" (left ear)"} else if i==1 {" (right ear)"} else {""} },
            data = {
                let mut out = String::new();
                cha.chunks(n_displayed_per_line).for_each(|chunk| {
                    out.push_str(spine);
                    for x in chunk.iter() {
                        x.print_to(&mut out);
                    }
                    out.push('\n');
                });
                out.push_str("    ┗━━━━━━━━━━━━━━━━━");
                out
            }
        )
    }).join("\n\n")
    }
}

trait FloatDisplay {
    const CHUNK_SIZE: usize;

    fn print_to(&self, out: &mut String);
}

impl FloatDisplay for f32 {
    const CHUNK_SIZE: usize = 10;

    fn print_to(&self, mut out: &mut String) {
        let x = self;
        if x.abs() > 0.001 || x.abs() == 0.0 {
            let _ = write!(&mut out, "{x:.3}, ");
        } else {
            let _ = write!(&mut out, "{x:.1E}, ");
        }
    }
}

impl FloatDisplay for Complex32 {
    const CHUNK_SIZE: usize = 5;

    fn print_to(&self, mut out: &mut String) {
        let x = self.re;
        if x.abs() > 0.001 || x.abs() == 0.0 {
            let _ = write!(&mut out, "{x:.3}");
        } else {
            let _ = write!(&mut out, "{x:.1E}");
        }

        let mut y = self.im;
        if y.is_sign_positive() {
            let _ = write!(&mut out, "+");
        } else {
            let _ = write!(&mut out, "−");
        }
        y = y.abs();
        if y > 0.001 || y == 0.0 {
            let _ = write!(&mut out, "{y:.3}j, ");
        } else {
            let _ = write!(&mut out, "{y:.1E}j, ");
        }
    }
}

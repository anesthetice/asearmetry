/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use crate::signal::{DSP, Domain, Signal, SignalSlice};
use core::fmt::Write;
use indoc::writedoc;
use itertools::Itertools;

impl<const C: usize, D: Domain> core::fmt::Debug for Signal<C, f32, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        display_debug_impl("Signal", self, f)
    }
}

impl<const C: usize, D: Domain> core::fmt::Display for Signal<C, f32, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self, f)
    }
}

impl<const C: usize, D: Domain> core::fmt::Debug for SignalSlice<'_, C, f32, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        display_debug_impl("SignalSlice", self, f)
    }
}

impl<const C: usize, D: Domain> core::fmt::Display for SignalSlice<'_, C, f32, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self, f)
    }
}

fn display_debug_impl<const C: usize, D: Domain, T: DSP<C, f32, D>>(
    name: &str,
    s: T,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    const CHUNK_SIZE: usize = 10;
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
                let (chunks, rem) = cha.as_chunks::<CHUNK_SIZE>();
                chunks
                    .iter()
                    .map(|chunk| chunk.as_slice())
                    .chain((!rem.is_empty()).then_some(rem))
                    .for_each(|chunk| {
                        out.push_str(spine);
                        for x in chunk.iter() {
                            if x.abs() > 0.001 || x.abs() == 0.0  {
                                write!(&mut out, "{x:.3}, ").unwrap();
                            } else {
                                write!(&mut out, "{x:.1E}, ").unwrap();
                            }
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

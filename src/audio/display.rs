use crate::audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal};
use indoc::writedoc;
use itertools::Itertools;
use std::fmt::Write;

impl<const C: usize> std::fmt::Debug for AudioBuffer<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display_debug_impl("AudioBuffer", self, f)
    }
}

impl<const C: usize> std::fmt::Display for AudioBuffer<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

impl<const C: usize> std::fmt::Debug for AudioBufferSlice<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display_debug_impl("AudioBufferSlice", self, f)
    }
}

impl<const C: usize> std::fmt::Display for AudioBufferSlice<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

fn display_debug_impl<const C: usize, T: DiscreteSignal<C>>(
    name: &str,
    s: T,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    const CHUNK_SIZE: usize = 10;
    writedoc! {
    f,
    "
        {name} {{
            sampling rate: {sr}
            number of channels: {C}
            samples per channel: {len}

        {channels}
        }}
    ",
    sr = s.sampling_rate().map_or_else(|| "None".to_string(), |x| x.to_string()),
    len = s.len(),
    channels = s.iter_cha().enumerate().map(|(i, cha)| {
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
                        out.push_str("    ┃ ");
                        for (i, x) in chunk.iter().enumerate() {
                            if i > 0 {out.push_str(", ")}
                            if x.abs() > 0.001 || x.abs() == 0.0 {
                                write!(&mut out, "{:.3}", x).unwrap();
                            } else {
                                write!(&mut out, "{:.1E}", x).unwrap();
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

use super::*;

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct AudioFrameRef<'a, const CHANNELS: usize>([&'a [f32]; CHANNELS]);

impl<'a, const CHANNELS: usize> AudioBufferCore<'a, CHANNELS> for AudioFrameRef<'a, CHANNELS> {
    fn is_empty(&self) -> bool {
        self.0.first().map(|s| s.is_empty()).unwrap_or(true)
    }
    fn len(&self) -> usize {
        self.0.first().map(|s| s.len()).unwrap_or(0)
    }
    fn cha(&'a self, c: usize) -> &'a [f32] {
        self.0[c]
    }
    fn get<I>(&'a self, index: I) -> AudioFrameRef<'a, CHANNELS>
    where
        I: std::slice::SliceIndex<[f32], Output = &'a [f32]> + Copy,
    {
        Self(std::array::from_fn(|i| self.cha(i)[index]))
    }
}

impl<'a, const CHANNELS: usize> AsRef<[&'a [f32]; CHANNELS]> for AudioFrameRef<'a, CHANNELS> {
    fn as_ref(&self) -> &[&'a [f32]; CHANNELS] {
        &self.0
    }
}

impl<'a, const CHANNELS: usize> From<&'a AudioFrame<CHANNELS>> for AudioFrameRef<'a, CHANNELS> {
    fn from(value: &'a AudioFrame<CHANNELS>) -> Self {
        Self(std::array::from_fn(|i| value.cha(i)))
    }
}

#[allow(clippy::from_over_into)]
impl<'a, const CHANNELS: usize> From<&'a Signal<CHANNELS>> for AudioFrameRef<'a, CHANNELS> {
    fn from(value: &'a Signal<CHANNELS>) -> Self {
        (&value.samples).into()
    }
}

#[allow(clippy::from_over_into)]
impl<'a, const CHANNELS: usize> From<[&'a [f32]; CHANNELS]> for AudioFrameRef<'a, CHANNELS> {
    fn from(value: [&'a [f32]; CHANNELS]) -> Self {
        Self(value)
    }
}

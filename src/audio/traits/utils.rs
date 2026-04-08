// Imports
use crate::audio::{AudioBuffer, AudioBufferSlice};

pub(crate) trait DiscreteSignalUtils {
    /// Unsafe, thus kept private to this crate, access a channel without checking.
    fn cha_uc(&self, c: usize) -> &[f32];
}

impl<const C: usize> DiscreteSignalUtils for AudioBuffer<C> {
    fn cha_uc(&self, c: usize) -> &[f32] {
        unsafe { self.channels.get_unchecked(c) }
    }
}

impl<const C: usize> DiscreteSignalUtils for AudioBufferSlice<'_, C> {
    fn cha_uc(&self, c: usize) -> &[f32] {
        unsafe { self.channels.get_unchecked(c) }
    }
}

impl<T> DiscreteSignalUtils for &T
where
    T: DiscreteSignalUtils,
{
    fn cha_uc(&self, c: usize) -> &[f32] {
        (*self).cha_uc(c)
    }
}

// Imports
use crate::audio::{AudioBuffer, AudioBufferSlice};

/// Contains either unsafe or duplicated utils that pay less attention to lifetimes
pub(crate) trait DiscreteSignalUtils<const C: usize> {
    fn _as_ref(&self) -> AudioBufferSlice<'_, C>;

    /// Unsafe, access a channel without checking.
    fn _cha(&self, c: usize) -> &[f32];

    /// Iterate through the channels
    fn _iter_cha(&self) -> impl Iterator<Item = &[f32]>;

    /// Apply a function across channels
    fn _map_cha<'s, T>(&'s self, f: impl FnMut(&'s [f32]) -> T) -> [T; C];
}

impl<const C: usize> DiscreteSignalUtils<C> for AudioBuffer<C> {
    fn _as_ref(&self) -> AudioBufferSlice<'_, C> {
        self.into()
    }
    fn _cha(&self, c: usize) -> &[f32] {
        unsafe { self.channels.get_unchecked(c) }
    }
    fn _iter_cha(&self) -> impl Iterator<Item = &[f32]> {
        self.channels.iter().map(|v| v.as_slice())
    }
    fn _map_cha<'s, T>(&'s self, mut f: impl FnMut(&'s [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self._cha(c)))
    }
}

impl<const C: usize> DiscreteSignalUtils<C> for AudioBufferSlice<'_, C> {
    fn _as_ref(&self) -> AudioBufferSlice<'_, C> {
        *self
    }
    fn _cha(&self, c: usize) -> &[f32] {
        unsafe { self.channels.get_unchecked(c) }
    }
    fn _iter_cha(&self) -> impl Iterator<Item = &[f32]> {
        self.channels.into_iter()
    }
    fn _map_cha<'s, T>(&'s self, mut f: impl FnMut(&'s [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self._cha(c)))
    }
}

impl<const C: usize, T> DiscreteSignalUtils<C> for &T
where
    T: DiscreteSignalUtils<C>,
{
    fn _as_ref(&self) -> AudioBufferSlice<'_, C> {
        (*self)._as_ref()
    }
    fn _cha(&self, c: usize) -> &[f32] {
        (*self)._cha(c)
    }
    fn _iter_cha(&self) -> impl Iterator<Item = &[f32]> {
        (*self)._iter_cha()
    }
    fn _map_cha<'s, O>(&'s self, f: impl FnMut(&'s [f32]) -> O) -> [O; C] {
        (*self)._map_cha(f)
    }
}

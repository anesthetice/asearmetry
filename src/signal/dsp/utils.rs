/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::signal::{Domain, Sample, Signal, SignalSlice};

/// Contains either unsafe or duplicated utils that pay less attention to lifetimes
pub(crate) trait DSPUtils<const C: usize, S: Sample, D: Domain> {
    fn _as_view(&self) -> SignalSlice<'_, C, S, D>;

    /// Unsafe, access a channel without checking.
    fn _cha(&self, c: usize) -> &[S];

    fn _chas(&self) -> [&[S]; C];

    /// Iterate through the channels
    fn _iter_cha(&self) -> impl Iterator<Item = &[S]>;

    /// Apply a function across channels
    fn _map_cha<'s, T>(&'s self, mut f: impl FnMut(&'s [S]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self._cha(c)))
    }

    /// Apply a function across channels with channel index available
    fn _map_cha_enumerate<'s, T>(&'s self, mut f: impl FnMut(usize, &'s [S]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(c, self._cha(c)))
    }
}

impl<const C: usize, S: Sample, D: Domain> DSPUtils<C, S, D> for Signal<C, S, D> {
    fn _as_view(&self) -> SignalSlice<'_, C, S, D> {
        self.into()
    }
    fn _cha(&self, c: usize) -> &[S] {
        unsafe { self.channels.get_unchecked(c) }
    }
    fn _chas(&self) -> [&[S]; C] {
        self.channels.each_ref().map(|v| v.as_slice())
    }
    fn _iter_cha(&self) -> impl Iterator<Item = &[S]> {
        self.channels.iter().map(|v| v.as_slice())
    }
}

impl<const C: usize, S: Sample, D: Domain> DSPUtils<C, S, D> for SignalSlice<'_, C, S, D> {
    fn _as_view(&self) -> SignalSlice<'_, C, S, D> {
        *self
    }
    fn _cha(&self, c: usize) -> &[S] {
        unsafe { self.channels.get_unchecked(c) }
    }
    fn _chas(&self) -> [&[S]; C] {
        self.channels
    }
    fn _iter_cha(&self) -> impl Iterator<Item = &[S]> {
        self.channels.into_iter()
    }
}

impl<const C: usize, T, S: Sample, D: Domain> DSPUtils<C, S, D> for &T
where
    T: DSPUtils<C, S, D>,
{
    fn _as_view(&self) -> SignalSlice<'_, C, S, D> {
        (*self)._as_view()
    }
    fn _cha(&self, c: usize) -> &[S] {
        (*self)._cha(c)
    }
    fn _chas(&self) -> [&[S]; C] {
        (*self)._chas()
    }
    fn _iter_cha(&self) -> impl Iterator<Item = &[S]> {
        (*self)._iter_cha()
    }
}

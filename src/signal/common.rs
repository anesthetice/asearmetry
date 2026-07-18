/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

//! Why not include all of this in the [`DSP`] trait? Because then for instance
//! the declaration of `cha` would have to be `fn cha(&self, c: usize) -> &[S]`. And
//! even if we used `fn cha(&self, c: usize) -> &'data [S]`, the prior lifetime "squash"
//! would still apply to this function and thus the returned value would not be able
//! to outlive our `SignalSlice<'data, C, S, D>` instead of the lifetime `'data` on which
//! it truly depends on.
//!
//! To be fair I'm not sure if this is worth the added cost in bad ergonomics.
//!
//! Maybe I could use a procedural macro to generate all of this instead, but probably not worth it currently.

use crate::signal::{DSP, Domain, Sample, Signal, SignalSlice};

#[allow(unused)]
impl<const C: usize, S: Sample, D: Domain> Signal<C, S, D> {
    pub fn view(&self) -> SignalSlice<'_, C, S, D> {
        self.into()
    }
    pub fn cha(&self, c: usize) -> &Vec<S> {
        &self.channels[c]
    }
    pub fn chas(&self) -> [&Vec<S>; C] {
        self.channels.each_ref()
    }
    pub(crate) fn cha_uc(&self, c: usize) -> &Vec<S> {
        unsafe { self.channels.get_unchecked(c) }
    }
    pub fn iter_cha(&self) -> impl Iterator<Item = &Vec<S>> {
        self.channels.iter()
    }
    pub fn map_cha<'s, T>(&'s self, mut f: impl FnMut(&'s Vec<S>) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
    pub fn map_cha_enumerate<'s, T>(&'s self, mut f: impl FnMut(usize, &'s [S]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(c, self.cha_uc(c)))
    }
    pub fn slice<I>(&self, index: I) -> SignalSlice<'_, C, S, D>
    where
        I: std::slice::SliceIndex<[S], Output = [S]> + Clone,
    {
        SignalSlice::new(self.map_cha(|cha| &cha[index.clone()]), self.sampling_rate)
    }
    pub fn slice_by_time(&self, start: f64, end: f64) -> SignalSlice<'_, C, S, D> {
        let sr = self.sr_or_panic();
        let start_idx = (sr * start).round() as usize;
        let end_idx = (sr * end).round() as usize + 1;
        assert!(start_idx < end_idx);
        self.slice(start_idx..end_idx)
    }
    pub fn slice_cha<'s, const C2: usize, I>(&'s self, slice: I) -> SignalSlice<'s, C2, S, D>
    where
        I: std::slice::SliceIndex<[&'s [S]], Output = [&'s [S]]>,
    {
        let channels: [&'s [S]; C2] = self.chas().map(|v| v.as_slice())[slice].try_into().unwrap();
        SignalSlice::new(channels, self.sampling_rate)
    }
    pub fn index_cha<'s, I>(&'s self, index: I) -> SignalSlice<'s, 1, S, D>
    where
        I: std::slice::SliceIndex<[&'s [S]], Output = &'s [S]>,
    {
        let channel: &'s [S] = self.chas().map(|v| v.as_slice())[index];
        SignalSlice::new([channel], self.sampling_rate)
    }
    pub fn first_n(&self, n: usize) -> SignalSlice<'_, C, S, D> {
        self.slice(0..n)
    }
    pub fn skip_first_n(&self, n: usize) -> SignalSlice<'_, C, S, D> {
        self.slice(n..)
    }
    /// The first element of the returned tuple contains the first n samples across all channels.
    pub fn first_n_split(&self, n: usize) -> (SignalSlice<'_, C, S, D>, SignalSlice<'_, C, S, D>) {
        (self.slice(0..n), self.slice(n..))
    }
    pub fn last_n(&self, n: usize) -> SignalSlice<'_, C, S, D> {
        self.slice(self.len() - n..)
    }
    fn skip_last_n(&self, n: usize) -> SignalSlice<'_, C, S, D> {
        self.slice(0..self.len() - n)
    }
    /// The second element of the returned tuple contains the last n samples across all channels.
    pub fn last_n_split(&self, n: usize) -> (SignalSlice<'_, C, S, D>, SignalSlice<'_, C, S, D>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }
    fn as_blocks(&self, block_size: usize) -> Vec<SignalSlice<'_, C, S, D>> {
        let rem = self.len() % block_size;
        let rhs = (rem > 0).then(|| self.last_n(rem));

        (0..self.len() / block_size)
            .map(|i| i * block_size..(i + 1) * block_size)
            .map(|i| self.slice(i))
            .chain(rhs)
            .collect()
    }
    pub fn as_blocks_strict(
        &self,
        block_size: usize,
    ) -> (
        Vec<SignalSlice<'_, C, S, D>>,
        Option<SignalSlice<'_, C, S, D>>,
    ) {
        let lhs = (0..self.len() / block_size)
            .map(|i| i * block_size..(i + 1) * block_size)
            .map(|i| self.slice(i))
            .collect();

        let rem = self.len() % block_size;
        let rhs = (rem > 0).then(|| self.last_n(rem));

        (lhs, rhs)
    }
}

#[allow(unused)]
impl<'data, const C: usize, S: Sample, D: Domain> SignalSlice<'data, C, S, D> {
    pub fn view(&self) -> SignalSlice<'data, C, S, D> {
        *self
    }
    pub fn cha(&self, c: usize) -> &'data [S] {
        self.channels[c]
    }
    pub fn chas(&self) -> [&'data [S]; C] {
        self.channels
    }
    pub(crate) fn cha_uc(&self, c: usize) -> &'data [S] {
        unsafe { self.channels.get_unchecked(c) }
    }
    pub fn iter_cha(&self) -> impl Iterator<Item = &'data [S]> {
        self.channels.into_iter()
    }
    pub fn map_cha<T>(&self, mut f: impl FnMut(&'data [S]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
    pub fn map_cha_enumerate<T>(&self, mut f: impl FnMut(usize, &'data [S]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(c, self.cha_uc(c)))
    }
    pub fn slice<I>(&self, index: I) -> SignalSlice<'data, C, S, D>
    where
        I: std::slice::SliceIndex<[S], Output = [S]> + Clone,
    {
        SignalSlice::new(self.map_cha(|cha| &cha[index.clone()]), self.sampling_rate)
    }
    pub fn slice_by_time(&self, start: f64, end: f64) -> SignalSlice<'data, C, S, D> {
        let sr = self.sr_or_panic();
        let start_idx = (sr * start).round() as usize;
        let end_idx = (sr * end).round() as usize + 1;
        assert!(start_idx < end_idx);
        self.slice(start_idx..end_idx)
    }
    pub fn slice_cha<const C2: usize, I>(&self, slice: I) -> SignalSlice<'data, C2, S, D>
    where
        I: std::slice::SliceIndex<[&'data [S]], Output = [&'data [S]]>,
    {
        let channels: [&'data [S]; C2] = self.channels[slice].try_into().unwrap();
        SignalSlice::new(channels, self.sampling_rate)
    }
    pub fn index_cha<I>(&self, index: I) -> SignalSlice<'data, 1, S, D>
    where
        I: std::slice::SliceIndex<[&'data [S]], Output = &'data [S]>,
    {
        let channel: &'data [S] = self.channels[index];
        SignalSlice::new([channel], self.sampling_rate)
    }
    pub fn first_n(&self, n: usize) -> SignalSlice<'data, C, S, D> {
        self.slice(0..n)
    }
    pub fn skip_first_n(&self, n: usize) -> SignalSlice<'data, C, S, D> {
        self.slice(n..)
    }
    /// The first element of the returned tuple contains the first n samples across all channels.
    pub fn first_n_split(
        &self,
        n: usize,
    ) -> (SignalSlice<'data, C, S, D>, SignalSlice<'data, C, S, D>) {
        (self.slice(0..n), self.slice(n..))
    }
    pub fn last_n(&self, n: usize) -> SignalSlice<'data, C, S, D> {
        self.slice(self.len() - n..)
    }
    fn skip_last_n(&self, n: usize) -> SignalSlice<'data, C, S, D> {
        self.slice(0..self.len() - n)
    }
    /// The second element of the returned tuple contains the last n samples across all channels.
    pub fn last_n_split(
        &self,
        n: usize,
    ) -> (SignalSlice<'data, C, S, D>, SignalSlice<'data, C, S, D>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }
    fn as_blocks(&self, block_size: usize) -> Vec<SignalSlice<'data, C, S, D>> {
        let rem = self.len() % block_size;
        let rhs = (rem > 0).then(|| self.last_n(rem));

        (0..self.len() / block_size)
            .map(|i| i * block_size..(i + 1) * block_size)
            .map(|i| self.slice(i))
            .chain(rhs)
            .collect()
    }
    pub fn as_blocks_strict(
        &self,
        block_size: usize,
    ) -> (
        Vec<SignalSlice<'data, C, S, D>>,
        Option<SignalSlice<'data, C, S, D>>,
    ) {
        let lhs = (0..self.len() / block_size)
            .map(|i| i * block_size..(i + 1) * block_size)
            .map(|i| self.slice(i))
            .collect();

        let rem = self.len() % block_size;
        let rhs = (rem > 0).then(|| self.last_n(rem));

        (lhs, rhs)
    }
}

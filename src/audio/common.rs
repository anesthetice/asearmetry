/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

//! Why not include all of this in the [`DiscreteSignal`] trait? Because then for instance
//! the declaration of `cha` would have to be `fn cha(&self, c: usize) -> &[f32]`. And
//! even if we used `fn cha(&self, c: usize) -> &'data [f32]`, the prior lifetime "squash"
//! would still apply to this function and thus the returned value would not be able
//! to outlive our `AudioBufferSlice<'data, C>` instead of the lifetime `'data` on which
//! it truly depends on.
//!
//! To be fair I'm not sure if this is worth the added cost in bad ergonomics.
//!
//! Maybe I could use a procedural macro to generate all of this instead, but probably not worth it currently.

use crate::audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal};

#[allow(unused)]
impl<const C: usize> AudioBuffer<C> {
    pub fn view(&self) -> AudioBufferSlice<'_, C> {
        self.into()
    }
    pub fn cha(&self, c: usize) -> &Vec<f32> {
        &self.channels[c]
    }
    pub fn chas(&self) -> [&Vec<f32>; C] {
        self.channels.each_ref()
    }
    pub(crate) fn cha_uc(&self, c: usize) -> &Vec<f32> {
        unsafe { self.channels.get_unchecked(c) }
    }
    pub fn iter_cha(&self) -> impl Iterator<Item = &Vec<f32>> {
        self.channels.iter()
    }
    pub fn map_cha<'s, T>(&'s self, mut f: impl FnMut(&'s Vec<f32>) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
    pub fn map_cha_enumerate<'s, T>(&'s self, mut f: impl FnMut(usize, &'s [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(c, self.cha_uc(c)))
    }
    pub fn slice<I>(&self, index: I) -> AudioBufferSlice<'_, C>
    where
        I: std::slice::SliceIndex<[f32], Output = [f32]> + Clone,
    {
        AudioBufferSlice::new(self.map_cha(|cha| &cha[index.clone()]), self.sampling_rate)
    }
    pub fn slice_by_time(&self, start: f64, end: f64) -> AudioBufferSlice<'_, C> {
        let sr = self
            .sampling_rate()
            .expect("Sampling rate must be defined to use this operation.");
        let start_idx = (sr * start).round() as usize;
        let end_idx = (sr * end).round() as usize + 1;
        assert!(start_idx < end_idx);
        self.slice(start_idx..end_idx)
    }
    pub fn slice_cha<'s, const D: usize, I>(&'s self, slice: I) -> AudioBufferSlice<'s, D>
    where
        I: std::slice::SliceIndex<[&'s [f32]], Output = [&'s [f32]]>,
    {
        let channels: [&'s [f32]; D] = self.chas().map(|v| v.as_slice())[slice].try_into().unwrap();
        AudioBufferSlice::new(channels, self.sampling_rate)
    }
    pub fn index_cha<'s, I>(&'s self, index: I) -> AudioBufferSlice<'s, 1>
    where
        I: std::slice::SliceIndex<[&'s [f32]], Output = &'s [f32]>,
    {
        let channel: &'s [f32] = self.chas().map(|v| v.as_slice())[index];
        AudioBufferSlice::new([channel], self.sampling_rate)
    }
    pub fn first_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(0..n)
    }
    pub fn skip_first_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(n..)
    }
    /// The first element of the returned tuple contains the first n samples across all channels.
    pub fn first_n_split(&self, n: usize) -> (AudioBufferSlice<'_, C>, AudioBufferSlice<'_, C>) {
        (self.slice(0..n), self.slice(n..))
    }
    pub fn last_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(self.len() - n..)
    }
    fn skip_last_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(0..self.len() - n)
    }
    /// The second element of the returned tuple contains the last n samples across all channels.
    pub fn last_n_split(&self, n: usize) -> (AudioBufferSlice<'_, C>, AudioBufferSlice<'_, C>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }
    fn as_blocks(&self, block_size: usize) -> Vec<AudioBufferSlice<'_, C>> {
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
        Vec<AudioBufferSlice<'_, C>>,
        Option<AudioBufferSlice<'_, C>>,
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
impl<'data, const C: usize> AudioBufferSlice<'data, C> {
    pub fn view(&self) -> AudioBufferSlice<'data, C> {
        *self
    }
    pub fn cha(&self, c: usize) -> &'data [f32] {
        self.channels[c]
    }
    pub fn chas(&self) -> [&'data [f32]; C] {
        self.channels
    }
    pub(crate) fn cha_uc(&self, c: usize) -> &'data [f32] {
        unsafe { self.channels.get_unchecked(c) }
    }
    pub fn iter_cha(&self) -> impl Iterator<Item = &'data [f32]> {
        self.channels.into_iter()
    }
    pub fn map_cha<T>(&self, mut f: impl FnMut(&'data [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
    pub fn map_cha_enumerate<T>(&self, mut f: impl FnMut(usize, &'data [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(c, self.cha_uc(c)))
    }
    pub fn slice<I>(&self, index: I) -> AudioBufferSlice<'data, C>
    where
        I: std::slice::SliceIndex<[f32], Output = [f32]> + Clone,
    {
        AudioBufferSlice::new(self.map_cha(|cha| &cha[index.clone()]), self.sampling_rate)
    }
    pub fn slice_by_time(&self, start: f64, end: f64) -> AudioBufferSlice<'data, C> {
        let sr = self
            .sampling_rate()
            .expect("Sampling rate must be defined to use this operation.");
        let start_idx = (sr * start).round() as usize;
        let end_idx = (sr * end).round() as usize + 1;
        assert!(start_idx < end_idx);
        self.slice(start_idx..end_idx)
    }
    pub fn slice_cha<const D: usize, I>(&self, slice: I) -> AudioBufferSlice<'data, D>
    where
        I: std::slice::SliceIndex<[&'data [f32]], Output = [&'data [f32]]>,
    {
        let channels: [&'data [f32]; D] = self.channels[slice].try_into().unwrap();
        AudioBufferSlice::new(channels, self.sampling_rate)
    }
    pub fn index_cha<I>(&self, index: I) -> AudioBufferSlice<'data, 1>
    where
        I: std::slice::SliceIndex<[&'data [f32]], Output = &'data [f32]>,
    {
        let channel: &'data [f32] = self.channels[index];
        AudioBufferSlice::new([channel], self.sampling_rate)
    }
    pub fn first_n(&self, n: usize) -> AudioBufferSlice<'data, C> {
        self.slice(0..n)
    }
    pub fn skip_first_n(&self, n: usize) -> AudioBufferSlice<'data, C> {
        self.slice(n..)
    }
    /// The first element of the returned tuple contains the first n samples across all channels.
    pub fn first_n_split(
        &self,
        n: usize,
    ) -> (AudioBufferSlice<'data, C>, AudioBufferSlice<'data, C>) {
        (self.slice(0..n), self.slice(n..))
    }
    pub fn last_n(&self, n: usize) -> AudioBufferSlice<'data, C> {
        self.slice(self.len() - n..)
    }
    fn skip_last_n(&self, n: usize) -> AudioBufferSlice<'data, C> {
        self.slice(0..self.len() - n)
    }
    /// The second element of the returned tuple contains the last n samples across all channels.
    pub fn last_n_split(
        &self,
        n: usize,
    ) -> (AudioBufferSlice<'data, C>, AudioBufferSlice<'data, C>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }
    fn as_blocks(&self, block_size: usize) -> Vec<AudioBufferSlice<'data, C>> {
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
        Vec<AudioBufferSlice<'data, C>>,
        Option<AudioBufferSlice<'data, C>>,
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

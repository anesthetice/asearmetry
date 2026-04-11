use crate::audio::{AudioBuffer, AudioBufferSlice, DiscreteSignal};

impl<const C: usize> AudioBuffer<C> {
    pub fn as_ref(&self) -> AudioBufferSlice<'_, C> {
        self.into()
    }
    pub fn cha(&self, c: usize) -> &Vec<f32> {
        &self.channels[c]
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
    pub fn slice<I>(&self, index: I) -> AudioBufferSlice<'_, C>
    where
        I: std::slice::SliceIndex<[f32], Output = [f32]> + Clone,
    {
        AudioBufferSlice::new(self.map_cha(|cha| &cha[index.clone()]), self.sampling_rate)
    }
    pub fn first_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(0..n)
    }
    pub fn skip_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(n..)
    }
    /// The first element of the returned tuple contains the first n samples across all channels.
    pub fn first_n_split(&self, n: usize) -> (AudioBufferSlice<'_, C>, AudioBufferSlice<'_, C>) {
        (self.slice(0..n), self.slice(n..))
    }
    pub fn last_n(&self, n: usize) -> AudioBufferSlice<'_, C> {
        self.slice(self.len() - n..)
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

impl<'a, const C: usize> AudioBufferSlice<'a, C> {
    pub fn as_ref(&self) -> AudioBufferSlice<'a, C> {
        *self
    }
    pub fn cha(&self, c: usize) -> &'a [f32] {
        self.channels[c]
    }
    pub(crate) fn cha_uc(&self, c: usize) -> &'a [f32] {
        unsafe { self.channels.get_unchecked(c) }
    }
    pub fn iter_cha(&self) -> impl Iterator<Item = &'a [f32]> {
        self.channels.into_iter()
    }
    pub fn map_cha<T>(&self, mut f: impl FnMut(&'a [f32]) -> T) -> [T; C] {
        std::array::from_fn(|c| f(self.cha_uc(c)))
    }
    pub fn slice<I>(&self, index: I) -> AudioBufferSlice<'a, C>
    where
        I: std::slice::SliceIndex<[f32], Output = [f32]> + Clone,
    {
        AudioBufferSlice::new(self.map_cha(|cha| &cha[index.clone()]), self.sampling_rate)
    }
    pub fn first_n(&self, n: usize) -> AudioBufferSlice<'a, C> {
        self.slice(0..n)
    }
    pub fn skip_n(&self, n: usize) -> AudioBufferSlice<'a, C> {
        self.slice(n..)
    }
    /// The first element of the returned tuple contains the first n samples across all channels.
    pub fn first_n_split(&self, n: usize) -> (AudioBufferSlice<'a, C>, AudioBufferSlice<'a, C>) {
        (self.slice(0..n), self.slice(n..))
    }
    pub fn last_n(&self, n: usize) -> AudioBufferSlice<'a, C> {
        self.slice(self.len() - n..)
    }
    /// The second element of the returned tuple contains the last n samples across all channels.
    pub fn last_n_split(&self, n: usize) -> (AudioBufferSlice<'a, C>, AudioBufferSlice<'a, C>) {
        (self.slice(..self.len() - n), self.slice(self.len() - n..))
    }
    fn as_blocks(&self, block_size: usize) -> Vec<AudioBufferSlice<'a, C>> {
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
        Vec<AudioBufferSlice<'a, C>>,
        Option<AudioBufferSlice<'a, C>>,
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

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

pub trait Sample:
    num_traits::Num
    + num_traits::NumAssign
    + num_traits::NumAssignRef
    + core::fmt::Debug
    + core::fmt::Display
    + core::iter::Sum<Self>
    + Clone
    + Copy
    + Send
    + Sync
    + 'static
{
    type Inner;
    fn as_inner(&self) -> &Self::Inner;
    fn into_inner(self) -> Self::Inner;
}

impl<T> Sample for T
where
    T: num_traits::Num
        + num_traits::NumAssign
        + num_traits::NumAssignRef
        + core::fmt::Debug
        + core::fmt::Display
        + core::iter::Sum<Self>
        + Clone
        + Copy
        + Send
        + Sync
        + 'static,
{
    type Inner = T;
    fn as_inner(&self) -> &Self::Inner {
        self
    }
    fn into_inner(self) -> Self::Inner {
        self
    }
}

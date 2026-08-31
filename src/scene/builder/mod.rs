/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
mod action;

// Imports
#[cfg(feature = "viewer")]
use crate::scene::core::ActorShape;
use crate::{
    coordinates::Cart3D, math::Seconds, scene::core::Modifiers, signal::audio::AudioBufferSlice,
    trajectory::Trajectory,
};
use action::Action;
use petgraph::Directed;

pub struct SceneBuilder<'a> {
    inner: petgraph::Graph<Actor<'a>, (), Directed>,
    root_idx: petgraph::graph::NodeIndices,
}

pub enum SceneBuilderElement<'a> {
    Delay(Seconds),
    Actor(Actor<'a>),
}

pub struct Actor<'a> {
    pub(crate) source: AudioBufferSlice<'a, 1>,
    pub(crate) actions: petgraph::Graph<Action, (), Directed>,

    /// The name of the actor, should be kept short, for debugging or visualization purposes.
    pub(crate) name: Option<String>,
    /// For visualization only, what shape represents the element.
    #[cfg(feature = "viewer")]
    pub(crate) shape: Option<ActorShape>,
    /// For visualization only, what color is the shape that represents the element.
    #[cfg(feature = "viewer")]
    pub(crate) color: Option<String>,
}

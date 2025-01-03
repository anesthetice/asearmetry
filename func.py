from prelude import pf, np, jax, jnp, jsp, PI


def linear(
    desired_coords: pf.Coordinates,
    source_coords: pf.Coordinates,
    source_coords_radius: float,
    hrirs: pf.Signal,
    nb_points: int = 3,
):
    desired_coords.radius = source_coords_radius * jnp.ones(
        desired_coords.cshape[0]
    )

    indices, distances = source_coords.find_nearest(desired_coords, k=nb_points)

    weights = jnp.divide(distances, jnp.sum(distances, axis=0))

    results = pf.add(
        tuple(
            [
                jnp.repeat(sub_weights[:, None], 2, axis=1) * hrirs[[sub_indices[0]]]
                for sub_indices, sub_weights in zip(indices, weights)
            ]
        )
    )

    return results


def concat_signals(signals: list[pf.Signal]):
    sampling_rate = signals[0].sampling_rate
    # Efficiently concatenate the time-domain data
    concatenated_time = np.concatenate([signal.time for signal in signals], axis=-1)

    # Create a new pyfar.Signal with concatenated time-domain data
    return pf.Signal(concatenated_time, sampling_rate=sampling_rate)

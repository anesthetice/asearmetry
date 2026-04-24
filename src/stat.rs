use std::borrow::Borrow;

pub fn mean<I, T>(data: I) -> f32
where
    I: IntoIterator<Item = T>,
    T: Borrow<f32>,
{
    let mut sum = 0.0;
    let mut count = 0.0;

    for x in data {
        sum += *x.borrow();
        count += 1.0;
    }

    if count == 0.0 { 0.0 } else { sum / count }
}

/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
pub fn std<I, T>(data: I, ddof: usize) -> f32
where
    I: IntoIterator<Item = T>,
    T: Borrow<f32>,
{
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut count = 0.0;

    for x in data {
        let x = *x.borrow();
        count += 1.0;

        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    if count == 0.0 {
        return 0.0;
    } else if count <= ddof as f32 {
        panic!("Length cannot be smaller or equal to the ddof");
    }

    (m2 / (count - ddof as f32)).sqrt()
}

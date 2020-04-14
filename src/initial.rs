//! Initialization of the fluid state.

pub mod energy;
pub mod mass;
pub mod velocity;

use crate::num::fvar;

fn search_idx_of_coord(lower_edges: &[fvar], coord: fvar) -> Option<usize> {
    let mut low = 0;
    let mut high = lower_edges.len() - 1;
    let mut mid;

    if coord >= lower_edges[high] {
        return Some(high);
    }

    while coord >= lower_edges[low] && coord < lower_edges[high] {
        let low_float = low as fvar;
        let high_float = high as fvar;
        let mid_float = (low_float
            + (coord - lower_edges[low]) * (high_float - low_float)
                / (lower_edges[high] - lower_edges[low]))
            .floor();

        mid = mid_float as usize;

        if mid >= high {
            // Due to roundoff error, we might get `mid == high` even though `coord < lower_edges[high]`.
            // If this happens, `coord` will be very close to `lower_edges[high]`, so we return `high - 1`.
            return Some(high - 1);
        }

        if lower_edges[mid + 1] <= coord {
            low = mid + 1
        } else if lower_edges[mid] > coord {
            high = mid
        } else {
            return Some(mid);
        }
    }
    None
}

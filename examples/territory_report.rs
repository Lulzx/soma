//! Does cohorting survive distribution across execution territories?

use soma::experiments::irregular_arrival::{trace, IrregularKnobs};
use soma::experiments::territories::report;

fn main() {
    // A larger population, so territories are not starved by construction.
    let knobs = IrregularKnobs { roots: 64, arrival_span: 12, depth: 4, ..IrregularKnobs::default() };
    let t = trace(&knobs);
    print!("{}", report(&t, 32, 8, 64));
}

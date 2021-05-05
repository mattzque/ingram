use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::types::{Confidence, Id, InvertedMapType, Map, Offset, Posting};
use crate::InvNGramIndex;

#[pyclass]
struct InvNGramIndexPython {
    _internal: Box<InvNGramIndex>,
}

#[pymethods]
impl InvNGramIndexPython {
    #[new]
    fn new(n: usize) -> Self {
        let index = InvNGramIndex::new(n, Some(1), None);

        Self {
            _internal: Box::new(index),
        }
    }

    fn add(&mut self, doc: &str, id: u64) {
        self._internal.add(doc, id);
    }

    fn search(&self, query: &str) -> Vec<(Id, Vec<Offset>)> {
        self._internal.search(query)
    }

    #[args(kwargs = "**")]
    fn search_approx(&self, query: &str, kwargs: Option<&PyDict>) -> Vec<(Id, Confidence, Offset)> {
        let threshold: Option<f32> = kwargs.and_then(|x| {
            x.get_item("threshold").and_then(|f| {
                f.extract()
                    .expect("threshold has to be floating point number")
            })
        });

        let wrap: Option<f32> = kwargs.and_then(|x| {
            x.get_item("wrap")
                .and_then(|f| f.extract().expect("wrap has to be floating point number"))
        });

        let collapse: Option<bool> = kwargs.and_then(|x| {
            x.get_item("collapse")
                .and_then(|f| f.extract().expect("collapse has to be a boolean"))
        });

        self._internal
            .search_approx(query, threshold, wrap, collapse)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ingram(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InvNGramIndexPython>()?;

    Ok(())
}

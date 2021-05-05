/// Invered N-Gram Indices..
///
mod python;
mod types;
mod util;

// extern crate unicode_segmentation;

use std::cmp;
use types::{Confidence, Id, Map, Offset, Posting};
use util::{create_occurence_matrix, find_connected, inverted_map, sliding_window};

use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub enum NGramError {
    DuplicateDocumentError,
}

pub struct InvNGramIndex {
    n: usize,
    pad: Option<usize>,
    pad_char: Option<char>,
    postings: Map<String, Vec<Posting>>,
    lengths: Map<Id, usize>,
}

impl InvNGramIndex {
    pub fn new(n: usize, pad: Option<usize>, pad_char: Option<char>) -> Self {
        InvNGramIndex {
            n,
            pad,
            pad_char,
            postings: Map::new(),
            lengths: Map::new(),
        }
    }

    /*
    fn get_ngram_length(&self, len: usize) -> usize {
        // ABC ->   $A AB BA BC C$

        let pad = if n == 1 { 0 } else { pad.unwrap_or(n - 1) };

        self.pad.unwrap_or(self.n - 1)

    }
    */

    fn insert_posting(&mut self, ngram: String, posting: Posting) {
        self.postings.entry(ngram).or_default().push(posting);
    }

    pub fn add(&mut self, doc: &str, id: Id) -> Result<(), NGramError> {
        let ngrams = sliding_window(&doc.to_owned(), self.n, self.pad, self.pad_char);

        if self.lengths.contains_key(&id) {
            Err(NGramError::DuplicateDocumentError)
        } else {
            let len = (doc.len() - (self.n - 1)) as isize;

            self.lengths
                .insert(id, if len < 0 { 0 } else { len as usize });
            let inverted = inverted_map(ngrams);

            for (ngram, offsets) in inverted {
                self.insert_posting(ngram, (id, offsets.to_vec()));
            }
            Ok(())
        }
    }

    fn find_postings(&self, ngrams: &Vec<(Offset, String)>) -> Vec<(Offset, &Vec<Posting>)> {
        ngrams
            .iter()
            .filter_map(|(query_offset, ngram)| match self.postings.get(ngram) {
                Some(postings) => Some((query_offset.to_owned(), postings)),
                None => None,
            })
            .collect()
    }

    pub fn search(&self, query: &str) -> Vec<(Id, Vec<Offset>)> {
        let query: String = query.to_owned();

        // extract ngrams from query string
        let ngrams = sliding_window(&query, self.n, self.pad, self.pad_char);

        // collect matching postings for each ngram in the inverted index
        let query_postings = self.find_postings(&ngrams);

        // generate matrix of document id to query offset to document offsets
        let occurence_matrix = create_occurence_matrix(query_postings);

        // collect matches
        // doc-id -> [doc-offset, ...]
        let mut matches: Map<Id, Vec<Offset>> = Map::new();

        for (doc_id, offsets) in occurence_matrix {
            // query-offset -> connected offsets
            let connected: Map<Offset, Vec<Vec<Offset>>> = find_connected(offsets);

            for (_, offsets) in &connected {
                for off in offsets.iter() {
                    let query_ngrams_count = if query.len() < self.n {
                        self.n - 1
                    } else {
                        query.len() - (self.n - 1)
                    };

                    // if the query ngram count is equal or more than connected match its a perfect match
                    if off.len() >= query_ngrams_count {
                        let o = off.get(0).unwrap().to_owned();

                        matches.entry(doc_id).or_default().push(cmp::max(0, o));
                    }
                }
            }
        }

        matches
            .iter()
            .map(|(&doc_id, offsets)| (doc_id.to_owned(), offsets.clone()))
            .collect()
    }

    /// Returns list with tuples of document-id to matches, matches is a vector of tuples with the
    /// match fraction (how much of the query was matched in the document, 1.0 the entire query
    /// was found in the document, 0.5 half of the query was found in the document.)
    /// e.g.
    ///   D-1 ABCDEF
    ///   Q-AB -> (1, 1.0, (1.0, 0))
    ///   Q-ABCD -> (1, 1.0, (1.0, 0))
    ///   Q-ABCDEF -> (1, 1.0, (1.0, 0))
    ///   Q-ABZF -> (1, 0.5, (0.5, 0))
    ///   Q-ABZFUUGG -> (1, 0.25, (0.25, 0))
    ///
    /// threshold determines the minium ratio of query found in doc
    ///
    ///
    /// The formular to calculate similarity is taken from the ngram python library:
    ///
    /// https://pythonhosted.org/ngram/ngram.html#ngram.NGram.ngram_similarity
    ///
    ///    (a^e - d^e) / a^e
    ///
    /// a - number of ngrams in document
    /// d - number of different ngrams between query and document
    /// e - warp, use warp greater than 1.0 to increase the similarity of shorter string pairs.
    ///     defaults to 1.0
    ///
    ///
    /// query: the query string to look for
    /// threshold: (default: 0.0) the minimum scored match to return.
    /// warp: (default: 1.0) used to increase similarity of shorter string pairs.
    /// collapse: (default: true) return only the highest scored match per document instead of all
    /// matches in each document.
    ///
    /// Returns:
    ///
    /// list of tuples with document-id, score, document-offset
    ///
    pub fn search_approx(
        &self,
        query: &str,
        threshold: Option<Confidence>,
        warp: Option<f32>,
        collapse: Option<bool>,
    ) -> Vec<(Id, Confidence, Offset)> {
        let threshold: f32 = threshold.unwrap_or(0.0);
        let warp: f32 = warp.unwrap_or(1.0);
        let query: String = query.to_owned();
        let collapse: bool = collapse.unwrap_or(true);

        // extract ngrams from query string
        let ngrams = sliding_window(&query, self.n, self.pad, self.pad_char);

        // collect matching postings for each ngram in the inverted index
        let query_postings = self.find_postings(&ngrams);

        // generate matrix of document id to query offset to document offsets
        let occurence_matrix = create_occurence_matrix(query_postings);

        // collect matches
        // doc-id -> [doc-offset, ...]
        let mut matches: Map<Id, Vec<(f32, Offset)>> = Map::new();

        // println!("===");
        // println!("Q => {:?}", query);

        for (doc_id, offsets) in occurence_matrix {
            // query-offset -> connected offsets
            let connected: Map<Offset, Vec<Vec<Offset>>> = find_connected(offsets);

            for (_, offsets) in &connected {
                for off in offsets.iter() {
                    let doc_len = self.lengths.get(&doc_id).unwrap().to_owned() as i64;

                    let qmax = (doc_len - (self.n as i64 - 1)) as i64;
                    let qmax = if qmax < 0 { 0 } else { qmax };

                    let connected_len = off.iter().filter(|&&o| o >= 0 && o <= qmax).count();
                    let connected_len = if connected_len < 1 {
                        0 as f32
                    } else {
                        connected_len as f32
                    };

                    // how many ngrams of the query were found in doc
                    //
                    // l + (n - 1)
                    let connected_len = connected_len + ((self.n - 1) as f32);
                    let query_ngrams_count = if connected_len != 1.0 {
                        connected_len.powf(warp)
                    } else {
                        connected_len
                    };
                    let doc_len = doc_len as f32 + ((self.n - 1) as f32);
                    let document_ngrams_count = if warp != 1.0 {
                        doc_len.powf(warp)
                    } else {
                        doc_len
                    };

                    // calculate score
                    let q = (document_ngrams_count - (document_ngrams_count - query_ngrams_count))
                        / document_ngrams_count;

                    if q > threshold {
                        let offset = off.get(0).unwrap().to_owned();
                        let offset = cmp::max(0, offset);

                        matches.entry(doc_id).or_default().push((q, offset));
                    }
                }
            }
        }

        let mut result: Vec<(Id, f32, Offset)> = matches
            .iter()
            .flat_map(|(doc_id, matches)| {
                let doc_id = doc_id.to_owned();
                let it = matches
                    .iter()
                    .map(|(score, offset)| (doc_id, score.to_owned(), offset.to_owned()));

                let mut x: Vec<(Id, f32, Offset)> = it.collect();
                x.sort_by(|&a, &b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).reverse());

                if collapse {
                    [x.into_iter().nth(0).unwrap()].to_vec()
                } else {
                    x
                }
            })
            .collect();

        result.sort_by(|&a, &b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).reverse());

        result
    }

    pub fn debug(&self) {
        for (key, value) in &self.postings {
            println!("{:?} -> {:?}", key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{InvNGramIndex, NGramError};

    #[test]
    fn test_inv_ngram_index_add_duplicate() {
        let mut index = InvNGramIndex::new(2, Some(1), None);
        assert!(index.add("ABCD", 0).is_ok());
        assert_eq!(
            index.add("FAIL", 0),
            Err(NGramError::DuplicateDocumentError)
        );
    }

    #[test]
    fn test_inv_ngram_index_search() {
        // bigram with padding
        let mut index = InvNGramIndex::new(2, Some(1), None);
        assert!(index.add("ABCDEF", 23).is_ok());
        assert!(index.add("GHAB", 42).is_ok());
        assert_eq!(index.search("CDEF"), [(23, [2].to_vec())].to_vec());
        assert_eq!(
            index.search("AB"),
            [(23, [0].to_vec()), (42, [2].to_vec())].to_vec()
        );
        // the ngram index won't be able to find all substrings if the query is smaller than n
        // only document prefix/suffixes can be found this way
        assert_eq!(index.search("A"), [(23, [0].to_vec())].to_vec());
        assert_eq!(index.search("F"), [(23, [5].to_vec())].to_vec());
        assert_eq!(index.search("B"), [(42, [3].to_vec())].to_vec());

        // bigram without padding
        let mut index = InvNGramIndex::new(2, Some(0), None);
        assert!(index.add("ABCDEF", 23).is_ok());
        assert!(index.add("GHAB", 42).is_ok());
        assert_eq!(index.search("CD"), [(23, [2].to_vec())].to_vec());
        assert_eq!(index.search("CDE"), [(23, [2].to_vec())].to_vec());
        assert_eq!(index.search("CDEF"), [(23, [2].to_vec())].to_vec());
        assert_eq!(index.search("ABCDEF"), [(23, [0].to_vec())].to_vec());
        assert_eq!(index.search("GHAB"), [(42, [0].to_vec())].to_vec());
        assert_eq!(index.search("A"), [].to_vec());
        assert_eq!(index.search("F"), [].to_vec());
        assert_eq!(index.search(""), [].to_vec());

        // multiple query matches in same document
        let mut index = InvNGramIndex::new(2, Some(0), None);
        assert!(index.add("ABCDABEF", 23).is_ok());
        assert!(index.add("GHAB", 42).is_ok());
        assert_eq!(
            index.search("AB"),
            [(23, [0, 4].to_vec()), (42, [2].to_vec())].to_vec()
        );

        let mut index = InvNGramIndex::new(6, None, None);
        assert!(index.add("ABCDEF", 23).is_ok());
        assert!(index.add("GHAB", 42).is_ok());
        assert!(index.add("A", 43).is_ok());
        assert!(index.add("", 44).is_ok());
        assert_eq!(index.search("A"), [(43, [0].to_vec())].to_vec());
        assert_eq!(index.search("F"), [].to_vec());
        assert_eq!(index.search(""), [].to_vec());
        assert_eq!(
            index.search_approx("A", None, None, None),
            [(43, 1.2, 0), (23, 0.8333333, 0)].to_vec()
        );
        assert_eq!(
            index.search_approx("F", None, None, None),
            [(23, 0.8333333, 5)].to_vec()
        );
        assert_eq!(index.search_approx("", None, None, None), [].to_vec());
    }

    #[test]
    fn test_inv_ngram_index_search_approx() {
        // bigram with padding
        let mut index = InvNGramIndex::new(2, Some(1), None);
        index.add("ABCD", 1);
        // index.add("GHAB", 42);
        assert_eq!(index.search_approx("Z", None, None, None), [].to_vec());
        assert_eq!(index.search_approx("ZZZZ", None, None, None), [].to_vec());
        assert_eq!(
            index.search_approx("A", None, None, None),
            [(1, 0.25, 0)].to_vec()
        );
        assert_eq!(
            index.search_approx("AB", None, None, None),
            [(1, 0.5, 0)].to_vec()
        );
        assert_eq!(
            index.search_approx("CD", None, None, None),
            [(1, 0.5, 2)].to_vec()
        );
        assert_eq!(
            index.search_approx("BC", None, None, None),
            [(1, 0.5, 1)].to_vec()
        );
        assert_eq!(
            index.search_approx("ABC", None, None, None),
            [(1, 0.75, 0)].to_vec()
        );
        assert_eq!(
            index.search_approx("ABCD", None, None, None),
            [(1, 1.0, 0)].to_vec()
        );

        // test collapse parameter:
        let mut index = InvNGramIndex::new(2, Some(1), None);
        index.add("ABAB", 1);
        assert_eq!(
            index.search_approx("AB", None, None, None),
            [(1, 0.5, 0)].to_vec()
        );
        assert_eq!(
            index.search_approx("AB", None, None, Some(true)),
            [(1, 0.5, 0)].to_vec()
        );
        assert_eq!(
            index.search_approx("AB", None, None, Some(false)),
            [(1, 0.5, 0), (1, 0.5, 2)].to_vec()
        );
        let mut index = InvNGramIndex::new(2, Some(1), None);
        index.add("ABABCD", 1);
        assert_eq!(
            index.search_approx("ABCD", None, None, None),
            [(1, 0.6666667, 2)].to_vec()
        );
        assert_eq!(
            index.search_approx("ABCD", None, None, Some(true)),
            [(1, 0.6666667, 2)].to_vec()
        );
        assert_eq!(
            index.search_approx("ABCD", None, None, Some(false)),
            [(1, 0.6666667, 2), (1, 0.33333334, 0)].to_vec()
        );
    }
}

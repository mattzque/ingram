use unicode_segmentation::UnicodeSegmentation;

use crate::types::{Id, InvertedMapType, Map, Offset, Posting};

const DEFAULT_PAD_CHAR: char = '$'; // '\u{00a0}';

/// Segment document into characters using unicode graphemes.
/// Returns vector with string slices to doc.
pub fn graphemes(doc: &str) -> Vec<&str> {
    UnicodeSegmentation::graphemes(doc, true).collect::<Vec<&str>>()
}

/// Extract sliding window substrings from a string.
/// This is used to extract sliding windows out of a document.
///
/// doc: the document to extract windows from
/// n: the size of the sliding window
/// pad: optional size of padding (before/after)
/// pad_char: the padding character to use (defaults to no-break space)
pub fn sliding_window(
    doc: &str,
    n: usize,
    pad: Option<usize>,
    pad_char: Option<char>,
) -> Vec<(Offset, String)> {
    let pad = if n == 1 { 0 } else { pad.unwrap_or(n - 1) };
    let pad_char: String = pad_char.unwrap_or(DEFAULT_PAD_CHAR).to_string();

    // extract unicode graphemes / unicode characters
    let chars: Vec<&str> = graphemes(doc);

    if chars.len() == 0 {
        return [].to_vec();
    }

    // add paddings to chars
    let fill: Vec<&str> = vec![pad_char.as_str(); pad as usize];
    let chars: Vec<&str> = fill
        .iter()
        .chain(chars.iter())
        .chain(fill.iter())
        .copied()
        .collect();

    if chars.len() < n {
        return [(0, chars.iter().map(|&cp| cp).collect())].to_vec();
    }

    (0..=(chars.len() - n))
        .map(|i| {
            (
                (i as i64) - (pad as i64),
                chars[i..(i + n)].iter().map(|&cp| cp).collect(),
            )
        })
        .collect()
}

/// Build inverted map from list of tuples of offsets and strings.
pub fn inverted_map(docs: Vec<(Offset, String)>) -> InvertedMapType {
    let mut map: InvertedMapType = Map::new();

    for (offset, doc) in docs {
        match map.get_mut(&doc) {
            Some(offsets) => offsets.push(offset),
            None => {
                let mut offsets = Vec::new();
                offsets.push(offset);
                map.insert(doc, offsets);
            }
        }
    }

    map
}

/// Maps document ids to their query and document offset pairs.
///
/// Takes a list of query offsets to posting lists with document id and document offsets
/// constructs and returns a map by document id to their query offset to document offsets.
///
/// # Examples:
///
///     (Oq - query offset, D - document-id, Od - document offset)
///
///     Oq   Postings (D, Od)
///     -1, [(0, [-1]), (1, [-1])]
///      0, [(0, [0]), (1, [0, 6])]
///      1, [(1, [7])]
///
///     ->
///
///      D    Oq     Od
///      0 -> -1 -> [-1]
///        ->  0 -> [0]
///      1 -> -1 -> [-1]
///            0 -> [0, 6]
///            1 -> [7]
///
pub fn create_occurence_matrix(
    query_postings: Vec<(Offset, &Vec<Posting>)>,
) -> Map<Id, Map<Offset, &Vec<Offset>>> {
    // matrix of doc-id -> query-offset -> doc-offsets
    let mut matrix: Map<Id, Map<Offset, &Vec<Offset>>> = Map::new();

    for (query_offset, postings) in &query_postings {
        for (id, doc_offsets) in postings.iter() {
            matrix
                .entry(id.to_owned())
                .or_default()
                .insert(query_offset.to_owned(), &doc_offsets);
        }
    }

    matrix
}

///
///
///
/// # Examples:
///
///     (Oq - query offset, D - document-id, Od - document offset)
///
///      Oq     Od
///      -1 -> [-1]
///       0 -> [0, 6]
///       1 -> [7]
///
///       ->
///
///      [[-1 [-1, 0]], [0 [6, 7]]]
///
///              -1 -> [-1, 0]
///              0  -> [6, 7]
///
///
pub fn find_connected(offsets: Map<Offset, &Vec<Offset>>) -> Map<Offset, Vec<Vec<Offset>>> {
    // last-doc-offset -> [ (query-offset, doc-offset), ... ]
    let mut last_doc_offsets: Map<Offset, Vec<(Offset, Offset)>> = Map::new();

    for (query_offset, doc_offsets) in &offsets {
        for doc_offset in doc_offsets.iter() {
            let prev_doc_offset = doc_offset - 1;
            let value = (query_offset.to_owned(), doc_offset.to_owned());

            if last_doc_offsets.contains_key(&prev_doc_offset) {
                // TODO we copy the entire last offsets here stupidly
                let mut last_offsets = last_doc_offsets.get(&prev_doc_offset).unwrap().clone();

                last_offsets.push(value);
                last_doc_offsets.insert(doc_offset.to_owned(), last_offsets.to_owned());
                last_doc_offsets.remove(&prev_doc_offset);
            } else {
                // let mut last_offsets: Vec<(Offset, Offset)> = Vec::new();
                // last_offsets
                last_doc_offsets.insert(doc_offset.to_owned(), [value].to_vec());
            }
        }
    }

    let mut result: Map<Offset, Vec<Vec<Offset>>> = Map::new();

    for (_, connected) in last_doc_offsets {
        let (first_query_offset, _) = connected.get(0).unwrap();
        // getting rid of the query offsets in each tuple:
        let connected_doc_offsets: Vec<Offset> = connected
            .iter()
            .map(|(_, doc_offset)| doc_offset.to_owned())
            .collect();

        result
            .entry(first_query_offset.to_owned())
            .or_default()
            .push(connected_doc_offsets);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window() {
        let pad_char = Some('$');

        assert_eq!(
            sliding_window("ABCD", 1, Some(0), pad_char),
            [
                (0, "A".to_owned()),
                (1, "B".to_owned()),
                (2, "C".to_owned()),
                (3, "D".to_owned())
            ]
        );

        assert_eq!(
            sliding_window("ABCD", 2, Some(0), pad_char),
            [
                (0, "AB".to_owned()),
                (1, "BC".to_owned()),
                (2, "CD".to_owned())
            ]
        );

        assert_eq!(
            sliding_window("ABCD", 3, Some(0), pad_char),
            [(0, "ABC".to_owned()), (1, "BCD".to_owned())]
        );

        assert_eq!(
            sliding_window("ABCD", 1, Some(1), pad_char),
            [
                (0, "A".to_owned()),
                (1, "B".to_owned()),
                (2, "C".to_owned()),
                (3, "D".to_owned())
            ]
        );

        assert_eq!(
            sliding_window("ABCD", 2, Some(1), pad_char),
            [
                (-1, "$A".to_owned()),
                (0, "AB".to_owned()),
                (1, "BC".to_owned()),
                (2, "CD".to_owned()),
                (3, "D$".to_owned())
            ]
        );

        assert_eq!(
            sliding_window("ABCD", 3, Some(1), pad_char),
            [
                (-1, "$AB".to_owned()),
                (0, "ABC".to_owned()),
                (1, "BCD".to_owned()),
                (2, "CD$".to_owned())
            ]
        );

        assert_eq!(
            sliding_window("A", 3, None, pad_char),
            [
                (-2, "$$A".to_owned()),
                (-1, "$A$".to_owned()),
                (0, "A$$".to_owned())
            ]
        );

        assert_eq!(
            sliding_window("A", 2, Some(1), pad_char),
            [(-1, "$A".to_owned()), (0, "A$".to_owned())]
        );

        assert_eq!(
            sliding_window("A", 2, Some(0), pad_char),
            [(0, "A".to_owned())]
        );

        assert_eq!(sliding_window("", 2, Some(1), pad_char), []);

        // test with unicode grapheme n-grams
        let example = "\u{13072}\u{13051}\u{13032}\u{13055}";
        let g = graphemes(example);
        assert_eq!(example.len(), 16); // 16 bytes
        assert_eq!(g.len(), 4); // 4 graphemes
        assert_eq!(
            sliding_window(example, 2, Some(0), pad_char),
            [
                (0, g[0..2].iter().cloned().collect()),
                (1, g[1..3].iter().cloned().collect()),
                (2, g[2..].iter().cloned().collect())
            ]
        );
    }

    #[test]
    fn test_inverted_map() {
        let pad_char = Some('$');
        let bigrams = sliding_window("ABC", 2, Some(0), pad_char);
        let map = inverted_map(bigrams);

        assert_eq!(
            map.get("AB").expect("invalid inverted bigrams"),
            &[0].to_vec()
        );
        assert_eq!(
            map.get("BC").expect("invalid inverted bigrams"),
            &[1].to_vec()
        );

        // ABAB -> AB BA AB -> AB: [0, 2], BA: [1]
        let bigrams = sliding_window("ABAB", 2, Some(0), pad_char);
        let map = inverted_map(bigrams);

        assert_eq!(
            map.get("AB").expect("invalid inverted bigrams"),
            &[0, 2].to_vec()
        );
        assert_eq!(
            map.get("BA").expect("invalid inverted bigrams"),
            &[1].to_vec()
        );
    }

    #[test]
    fn test_create_occurence_matrix() {
        // [doc-id, doc-offsets pairs]
        let postings = [
            [(0, [-1].to_vec()), (1, [-1].to_vec())].to_vec(),
            [(0, [0].to_vec()), (1, [0, 6].to_vec())].to_vec(),
            [(1, [7].to_vec())].to_vec(),
        ]
        .to_vec();

        let mat = create_occurence_matrix(
            [(-1, &postings[0]), (0, &postings[1]), (1, &postings[2])].to_vec(),
        );

        assert_eq!(
            format!("{:?}", mat),
            "{0: {-1: [-1], 0: [0]}, 1: {-1: [-1], 0: [0, 6], 1: [7]}}"
        );
    }

    #[test]
    fn test_find_connected() {
        let doc_offsets = [[-1].to_vec(), [0, 6].to_vec(), [7].to_vec()].to_vec();
        let mut offsets: Map<Offset, &Vec<Offset>> = Map::new();
        offsets.insert(-1, &doc_offsets[0]);
        offsets.insert(0, &doc_offsets[1]);
        offsets.insert(1, &doc_offsets[2]);

        let connected = find_connected(offsets);
        assert_eq!(format!("{:?}", connected), "{-1: [[-1, 0]], 0: [[6, 7]]}");
    }
}

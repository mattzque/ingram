use std::collections::BTreeMap;

pub type Offset = i64;
pub type Id = u64;
pub type Confidence = f32;
/// Map type used for various indices. HashMap seems slower in benchmarks for me.
pub type Map<K, V> = BTreeMap<K, V>;
pub type Posting = (Id, Vec<Offset>);
pub type InvertedMapType = Map<String, Vec<Offset>>;

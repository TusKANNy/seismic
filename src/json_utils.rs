use std::{collections::HashMap, fs::File, io::BufReader};

use serde::Deserialize;

use serde_json::Deserializer;

use vectorium::FromF32;
use vectorium::ValueType;

#[derive(Debug, Deserialize)]
#[serde(untagged)] // This allows deserializing without tagging the enum variants
pub enum DocIdType {
    StringId(String),
    UsizeId(usize),
}

#[derive(Debug, Deserialize)]
pub struct JsonSparseVector {
    id: DocIdType,
    vector: HashMap<String, f32>,
    content: Option<String>,
}

impl JsonSparseVector {
    pub fn vector(&self) -> &HashMap<String, f32> {
        &self.vector
    }

    pub fn content(&self) -> Option<&str> {
        self.content.as_deref()
    }

    /// Returns the document ID as a string. Accepts both string and integer IDs in the input,
    /// but always stores them as strings internally.
    pub fn get_id_as_string(&self) -> String {
        match &self.id {
            DocIdType::StringId(s) => s.clone(),
            DocIdType::UsizeId(n) => n.to_string(),
        }
    }
}

pub fn extract_jsonl<V>(
    current_jsonl: JsonSparseVector,
) -> (String, Vec<String>, Vec<V>, Option<String>)
where
    V: ValueType + FromF32,
{
    let (coords, values): (Vec<_>, Vec<_>) = current_jsonl
        .vector()
        .iter()
        .map(|(s, y)| (s.to_string(), V::from_f32_saturating(*y)))
        .unzip();

    (
        current_jsonl.get_id_as_string(),
        coords,
        values,
        current_jsonl.content,
    )
}

pub fn read_queries(input_file: &String) -> Vec<(String, Vec<String>, Vec<f32>)> {
    let f = File::open(input_file).unwrap_or_else(|_| panic!("Unable to open {}", input_file));
    let reader = BufReader::new(f);
    let stream: serde_json::StreamDeserializer<
        serde_json::de::IoRead<BufReader<File>>,
        JsonSparseVector,
    > = Deserializer::from_reader(reader).into_iter();

    stream
        .into_iter()
        .map(|x| {
            let (id, tokens, values, _content) = extract_jsonl(x.unwrap());
            (id, tokens, values)
        })
        .collect()
}

use std::{collections::HashMap, fs::File, io::BufReader};

use serde::Deserialize;

//use qwt::SpaceUsage as SpaceUsageQwt;
use serde_json::Deserializer;

use crate::DataType;

#[derive(Debug, Deserialize)]
#[serde(untagged)] // This allows deserializing without tagging the enum variants
pub enum DocIdType {
    StringId(String),
    UsizeId(usize),
}

//TODO: change this name with a more meaningful one
#[derive(Debug, Deserialize)]
pub struct JsonFormat {
    id: DocIdType,
    vector: HashMap<String, f32>,
}

impl JsonFormat {
    pub fn vector(&self) -> &HashMap<String, f32> {
        &self.vector
    }

    //TODO: this allows to read docids either as a string or as an integer, but we always store them as
    // strings in memory.
    pub fn get_id_as_string(&self) -> String {
        match &self.id {
            DocIdType::StringId(s) => s.clone(),    // Clone the String
            DocIdType::UsizeId(n) => n.to_string(), // Convert usize to String
        }
    }
}

pub fn extract_jsonl<T>(current_jsonl: JsonFormat) -> (String, Vec<String>, Vec<T>)
where
    T: DataType,
{
    let (coords, values): (Vec<_>, Vec<_>) = current_jsonl
        .vector()
        .iter()
        .map(|(s, y)| (s.to_string(), T::from_f32(*y).unwrap()))
        .unzip();

    (current_jsonl.get_id_as_string(), coords, values)
}

pub fn read_queries(input_file: &String) -> Vec<(String, Vec<String>, Vec<f32>)> {
    let f = File::open(input_file).expect(&format!("Unable to open {}", input_file));
    let reader = BufReader::new(f);
    let stream: serde_json::StreamDeserializer<
        serde_json::de::IoRead<BufReader<File>>,
        JsonFormat,
    > = Deserializer::from_reader(reader).into_iter();

    let values = stream
        .into_iter()
        .map(|x| (extract_jsonl(x.unwrap())))
        .collect();
    values
}
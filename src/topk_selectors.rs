/// A TopKSelector could be either online or offline.
/// Here we store distances and we report the k smallest distances.
/// This means that for some metrics (such as dot product) you want to
/// store negative distances to get the closest vector.
///
/// We report the distances and ids of the top k items
/// The id is just the timestamp (from 0 to number of items inserted so far)
/// of the item. You must be able to remap those timestamps to the original ids
/// if needed.
/// We adopt this strategy for two reasons:
/// - The value of k is small so we can afford the remapping;
/// - The number of distances to be checked is large so we want to save the
///   time needed to create a vector of original ids and copy them.
///
/// An online selector, such as an implementation of a Heap,
/// updates the data structure after every `push`.
/// The current top k values can be reported efficiently after every push.
///
/// An offline selector (e.g., quickselect) may just collect every pushed
/// distances without doing anything. Then it can spend spends more time
/// (e.g., linear time) in computing the `topk` distances.
///
/// An online selector may be faster if a lot of distance are processed
/// at once.

pub trait OnlineTopKSelector {
    /// Creates a new empty data structure to compute top-`k` distances.
    fn new(k: usize) -> Self;

    /// Pushes a new item `distance` with the current timestamp.
    /// If the data structure has less than k distances, the current one is
    /// inserted.
    /// Otherwise, the current one replaces the largest distance
    /// stored so far, if it is smaller.
    fn push(&mut self, distance: f32);

    /// Pushes a new item `distance` with a specified `id` as its timestamp.
    /// If the data structure has less than k distances, the current one is
    /// inserted.
    /// Otherwise, the current one replaces the largest distance
    /// stored so far, if it is smaller.
    fn push_with_id(&mut self, distance: f32, id: usize);

    /// Pushes a slide of items `distances`.
    fn extend(&mut self, distances: &[f32]);

    /// Returns the top-k distances and their timestamps.
    /// The method returns these top-k distances as a
    /// sorted (by decreasing distance) vector of pairs.
    fn topk(&self) -> Vec<(f32, usize)>;
}

pub mod heap_faiss;
pub use heap_faiss::HeapFaiss;

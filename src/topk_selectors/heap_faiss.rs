use super::*;
use itertools::Itertools;

pub struct HeapFaiss {
    distances: Vec<f32>,
    ids: Vec<usize>,
    k: usize,
    timestamp: usize,
}

impl HeapFaiss {
    /// Adds a new `distance` and its associated `id` to the heap.
    ///
    /// This method is responsible for maintaining the heap property
    /// while inserting a new element. The heap property ensures that
    /// for any given node `I`, the value of `I` is greater than or equal
    /// to the values of its children. This property must hold recursively
    /// for all nodes in the binary tree.
    ///
    /// The `add` method starts by appending the new `distance` and `id` at
    /// the end of the `distances` and `ids` vectors respectively.
    /// It then repeatedly compares the newly added `distance` with its parent
    /// node, swapping them if necessary, until the heap property is restored.
    ///
    /// # Parameters
    /// - `distance` (`f32`): The distance value to be added to the heap.
    /// - `id` (`usize`): The identifier associated with the distance.
    ///
    /// # Examples
    ///
    /// Demonstrating `add` through `push`:
    /// ```
    /// use seismic::topk_selectors::{OnlineTopKSelector, HeapFaiss};
    ///
    /// let mut heap = HeapFaiss::new(3);
    /// heap.push(2.0);  // This calls `add` internally as the heap is not yet full
    /// heap.push(3.0);  // This also calls `add` internally
    ///
    /// let top_k = heap.topk();
    /// assert_eq!(top_k, vec![(2.0, 0), (3.0, 1)]);
    /// ```
    #[inline]
    fn add(&mut self, distance: f32, id: usize) {
        self.distances.push(distance);
        self.ids.push(id);

        let mut i = self.distances.len() - 1;
        let mut i_father;

        while i > 0 {
            i_father = ((i + 1) >> 1) - 1;
            if distance <= self.distances[i_father] {
                break;
            }
            self.distances[i] = self.distances[i_father];
            self.ids[i] = self.ids[i_father];
            i = i_father;
        }
        self.distances[i] = distance;
        self.ids[i] = id;
    }

    /// Replaces the largest distance in the heap with a new distance and id.
    ///
    /// This method maintains the heap property by placing the new element
    /// at the root and then repeatedly swapping it with one of its children
    /// until the heap property is restored.
    ///
    /// # Parameters
    /// - `distance` (`f32`): The new distance value to replace the largest
    ///   distance in the heap.
    /// - `id` (`usize`): The identifier associated with the new distance.
    ///
    /// # Examples
    /// ```
    /// use seismic::topk_selectors::{OnlineTopKSelector, HeapFaiss};
    ///
    /// let mut heap = HeapFaiss::new(3);
    /// heap.push(2.0);
    /// heap.push(3.0);
    /// heap.push(4.0);
    ///
    /// // Replace the top distance with a new distance
    /// heap.replace_top(1.0, 4);  // id is arbitrary here
    /// let top_k = heap.topk();
    /// assert_eq!(top_k, vec![(1.0, 4), (2.0, 0), (3.0, 1)]);
    /// ```
    #[inline]
    pub fn replace_top(&mut self, distance: f32, id: usize) {
        let k: usize = self.distances.len();
        let mut i = 0;
        let mut i1;
        let mut i2;

        loop {
            i2 = (i + 1) << 1;
            i1 = i2 - 1;
            if i1 >= k {
                break;
            }
            if (i2 == k) || (self.distances[i1] >= self.distances[i2]) {
                if distance >= self.distances[i1] {
                    break;
                }
                self.distances[i] = self.distances[i1];
                self.ids[i] = self.ids[i1];
                i = i1;
            } else {
                if distance >= self.distances[i2] {
                    break;
                }
                self.distances[i] = self.distances[i2];
                self.ids[i] = self.ids[i2];
                i = i2;
            }
        }
        self.distances[i] = distance;
        self.ids[i] = id;
    }

    #[inline]
    pub fn top(&self) -> f32 {
        self.distances[0]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.distances.len() == 0
    }
}

impl OnlineTopKSelector for HeapFaiss {
    /// Creates a new empty data structure to compute top `k` pairs.
    ///
    /// This method initializes a new `HeapFaiss` instance with specified
    /// capacity `k`.
    /// It allocates memory for the `distances` and `ids` vectors, which are
    /// used to store the distances and their associated ids, respectively.
    ///
    /// # Parameters
    /// - `k` (`usize`): The number of top distances to keep track of.
    /// # Examples
    /// ```
    /// use seismic::topk_selectors::{OnlineTopKSelector, HeapFaiss};
    ///
    /// let mut heap = HeapFaiss::new(2);
    ///
    /// heap.push(0.0);
    /// heap.push(1.0);
    /// heap.push(2.0);
    ///
    /// let top_k = heap.topk();
    /// ```
    #[inline]
    fn new(k: usize) -> Self {
        HeapFaiss {
            distances: Vec::<f32>::with_capacity(k),
            ids: Vec::<usize>::with_capacity(k),
            k,
            timestamp: 0,
        }
    }

    /// Pushes a new item `distance` with the current timestamp.
    /// If the data structure has less than k distances, the current one is
    /// inserted.
    /// Otherwise, the current one replaces the largest distance
    /// stored so far, if it is smaller.
    ///
    ///
    /// Inserts a new distance into the heap.
    ///
    /// This method decides whether to add a new distance to the heap or
    /// replace the largest distance with the new one, based on the current
    /// size of the heap and the value of the new distance. If the number of
    /// distances in the heap is less than `k`, the new distance is added to
    /// the heap. Otherwise, if the new distance is smaller than the largest
    /// distance in the heap, it replaces the largest distance.
    ///
    /// # Parameters
    /// - `distance` (`f32`): The distance value to be inserted into the heap.
    ///
    /// # Examples
    /// ```
    /// use seismic::topk_selectors::{OnlineTopKSelector, HeapFaiss};
    ///
    /// let mut heap = HeapFaiss::new(3);
    /// heap.push(2.0);
    /// heap.push(3.0);
    /// heap.push(4.0);
    ///
    /// // Now, push a smaller distance
    /// heap.push(1.0);
    ///
    /// // Check if the largest distance was replaced
    /// let top_k = heap.topk();
    /// assert_eq!(top_k, vec![(1.0, 3), (2.0, 0), (3.0, 1)]);
    /// ```
    #[inline]
    fn push(&mut self, distance: f32) {
        if self.timestamp < self.k {
            self.add(distance, self.timestamp);
            self.timestamp += 1;
            return;
        }

        if distance < self.top() {
            self.replace_top(distance, self.timestamp);
        }
        self.timestamp += 1;
    }

    #[inline]
    fn push_with_id(&mut self, distance: f32, id: usize) {
        if self.timestamp < self.k {
            self.add(distance, id);
            self.timestamp += 1;
            return;
        }

        if distance < self.top() {
            self.replace_top(distance, id);
        }
        self.timestamp += 1;
    }

    /// Pushes a slide of items `distances` with their corresponding timestamps.
    ///
    /// This method iterates through the given slice and calls the `push`
    /// method to insert each distance into the heap.
    ///
    /// # Parameters
    /// - `distances` (`&[f32]`): A slice of distance values to be inserted
    ///   into the heap.
    ///
    /// # Examples
    /// ```
    /// use seismic::topk_selectors::{OnlineTopKSelector, HeapFaiss};
    ///
    /// let mut heap = HeapFaiss::new(3);
    /// heap.extend(&[0.0, 1.0, 2.0]);
    ///
    /// // Now, the heap has three distances
    /// ```
    #[inline]
    fn extend(&mut self, distances: &[f32]) {
        let mut iter = distances.iter().enumerate();

        // Deal with the very first k items for the data structure
        // which must be always inserted
        while self.distances.len() < self.k {
            match iter.next() {
                Some((id, &distance)) => self.add(distance, self.timestamp + id),
                None => {
                    self.timestamp += distances.len();
                    return;
                }
            }
        }

        // Other distances
        for (id, &distance) in iter {
            if distance < self.top() {
                self.replace_top(distance, self.timestamp + id);
            }
        }

        self.timestamp += distances.len();
    }

    /// The `topk` function sorts a collection of pairs `(f32, usize)` based on
    /// the `f32` values, in ascending order, using an unstable sorting
    /// algorithm, and returns the sorted pairs in a `Vec<(f32, usize)>`.
    ///
    /// # Parameters
    /// - `&self`: A reference to the instance of the containing struct. This
    ///   function assumes that the struct has two fields: `distances` and
    ///   `ids`, both of which are slices or vectors.
    ///
    /// # Returns
    /// - A `Vec<(f32, usize)>` containing the sorted pairs.
    ///
    /// # Unstable Sorting
    /// This function uses the `sorted_unstable_by` method from the `itertools`
    /// crate for sorting. Unlike a stable sort, an unstable sort does not
    /// preserve the relative order of elements that compare as equal. This can
    /// lead to a performance advantage, as unstable sorting algorithm doesn't
    /// have the constraint of maintaining the order of equal elements.
    ///
    /// # Examples
    /// ```
    /// use seismic::topk_selectors::{OnlineTopKSelector, HeapFaiss};
    /// use itertools::Itertools;
    ///
    /// let mut heap = HeapFaiss::new(3);
    /// heap.extend(&[0.0, -1.0, 2.0]);
    ///
    /// // Now, get the top k distances along with their timestamps
    /// let top_k = heap.topk();
    /// assert_eq!(top_k, vec![(-1.0, 1), (0.0, 0), (2.0, 2)]);
    /// ```
    #[inline]
    fn topk(&self) -> Vec<(f32, usize)> {
        self.distances
            .iter()
            .zip(&self.ids)
            .map(|(d, i)| (*d, *i))
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================
    // 1. Initialization Tests
    // ========================

    /// Tests the default initialization of `HeapFaiss`.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` instance with a capacity of 10.
    /// 2. Checks if all properties are correctly initialized.
    ///
    /// Expected behavior:
    /// All properties should be correctly set to their initial values.
    #[test]
    fn test_default_initialization() {
        let heap = HeapFaiss::new(10);

        assert_eq!(heap.distances.len(), 0);
        assert_eq!(heap.ids.len(), 0);
        assert_eq!(heap.k, 10);
        assert_eq!(heap.timestamp, 0);
    }

    // ========================
    // 2. Push Behavior Tests
    // ========================

    /// Tests the behavior of `HeapFaiss` when distances are pushed in a non-sequential order.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 5.
    /// 2. Pushes distances in a non-sequential order using the `extend` method.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// All pushed distances should be present in the results in ascending order with their indices.
    #[test]
    fn test_non_sequential_push() {
        let mut h = HeapFaiss::new(5);

        h.extend(&[3.0, 5.0, 1.0, 4.0, 2.0]);

        let expected = vec![(1.0, 2), (2.0, 4), (3.0, 0), (4.0, 3), (5.0, 1)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the `HeapFaiss`'s ability to handle multiple distances using the `extend` method.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 4.
    /// 2. Pushes an array of four distances using the `extend` method.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// The distances pushed are `[4.0, 3.0, 2.0, 1.0]`. These are returned in
    /// ascending order with their corresponding indices.
    #[test]
    fn test_multi_distance_extend() {
        let mut h = HeapFaiss::new(4);

        h.extend(&[4.0, 3.0, 2.0, 1.0]);

        let expected = vec![(1.0, 3), (2.0, 2), (3.0, 1), (4.0, 0)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the behavior of `HeapFaiss` when mutating the heap after a retrieval.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 4.
    /// 2. Extends the heap with an array of regular positive values.
    /// 3. Retrieves the top K values without mutating them.
    /// 4. Pushes a new value into the heap.
    /// 5. Retrieves the top K values again and checks them against an expected result.
    ///
    /// Expected behavior:
    /// The results after the second retrieval should reflect the newly pushed value and be in the correct order.
    #[test]
    fn test_mutate_after_retrieval() {
        let mut h = HeapFaiss::new(4);
        h.extend(&[3.0, 2.0, 1.0]);

        let _ = h.topk();

        h.push(0.5);

        let expected = vec![(0.5, 3), (1.0, 2), (2.0, 1), (3.0, 0)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests if a newly pushed smaller distance displaces an existing larger one from the `HeapFaiss`.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 3.
    /// 2. Pushes an array of three distances.
    /// 3. Pushes a new smaller distance.
    /// 4. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// After pushing the distances `[3.0, 2.0, 4.0]`, the new distance `1.0` is pushed.
    /// Since `1.0` is smaller than the largest existing distance `4.0`, it should displace `4.0`
    /// in the top-k results.
    #[test]
    fn test_smaller_distance_displacement() {
        let mut h = HeapFaiss::new(3);

        h.extend(&[3.0, 2.0, 4.0]);

        assert_eq!(h.top(), 4.0);

        h.push(1.0);

        assert_eq!(h.top(), 3.0);

        let distances: Vec<_> = h.topk().iter().map(|&(d, _)| d).collect();
        assert!(distances.contains(&1.0));
        assert!(!distances.contains(&4.0));
    }

    /// Tests the behavior of `HeapFaiss` when pushing negative distances.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 4.
    /// 2. Pushes a mixture of negative and positive distances.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// All pushed distances, including negative values, should be present in the results
    /// in ascending order with their indices.
    #[test]
    fn test_negative_distances() {
        let mut h = HeapFaiss::new(4);

        h.extend(&[-1.0, 2.0, -3.0, 0.0]);

        let expected = vec![(-3.0, 2), (-1.0, 0), (0.0, 3), (2.0, 1)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the behavior of `HeapFaiss` when encountering duplicate distances.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 4.
    /// 2. Pushes an array with duplicate distances.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// All pushed distances, including duplicates, should be present in the results in ascending order
    /// with their indices.
    #[test]
    fn test_duplicate_distances() {
        let mut h = HeapFaiss::new(4);

        h.extend(&[1.0, 2.0, 1.0, 1.0]);

        let expected = vec![(1.0, 0), (1.0, 2), (1.0, 3), (2.0, 1)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the behavior of `HeapFaiss` when the number of distances pushed is equal to its capacity.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 3.
    /// 2. Pushes three distances.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// All pushed distances should be present in the results in ascending order with their indices.
    #[test]
    fn test_push_equal_to_k() {
        let mut h = HeapFaiss::new(3);

        h.extend(&[2.0, 1.0, 3.0]);

        let expected = vec![(1.0, 1), (2.0, 0), (3.0, 2)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the behavior of `HeapFaiss` when its specified size is zero.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 0.
    /// 2. Retrieves and verifies the results.
    ///
    /// Expected behavior:
    /// Since the capacity is zero, the results should be an empty vector.
    #[test]
    fn test_heap_zero_size() {
        let h = HeapFaiss::new(0);
        assert_eq!(h.topk(), Vec::<(f32, usize)>::new());
    }

    /// Tests the behavior of `HeapFaiss` when pushing a distance into a zero-sized heap.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 0.
    /// 2. Attempts to push a distance.
    /// 3. Retrieves and verifies the results.
    ///
    /// Expected behavior:
    /// Pushing a distance to a zero-sized heap should panic with an "index out of bounds" error.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_push_heap_zero_sized() {
        let mut h = HeapFaiss::new(0);

        h.push(1.0);

        assert_eq!(h.topk(), Vec::<(f32, usize)>::new());
    }

    /// Tests the behavior of `HeapFaiss` when its specified size is greater than the number of distances pushed.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 4.
    /// 2. Pushes three distances using the `extend` method.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// Even though the `HeapFaiss` capacity is 4, only three distances are pushed.
    /// Thus, the results should contain only those three distances in ascending order with their indices.
    #[test]
    fn test_heap_size_larger_than_distances() {
        let mut h = HeapFaiss::new(4);

        h.extend(&[1.0, 2.0, 3.0]);

        let expected = vec![(1.0, 0), (2.0, 1), (3.0, 2)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the behavior of `HeapFaiss` when its specified size is smaller than the number of distances pushed.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 2.
    /// 2. Pushes five distances using the `extend` method.
    /// 3. Retrieves and verifies the top-k results.
    ///
    /// Expected behavior:
    /// Only the two smallest distances out of the five pushed should be retained in the results, in
    /// this case `[(-1.0, 5), (1.0, 0)]`.
    #[test]
    fn test_heap_size_smaller_than_distances() {
        let mut h = HeapFaiss::new(2);

        h.extend(&[1.0, 3.0, 2.0, 4.0, 5.0, -1.0]);

        let expected = vec![(-1.0, 5), (1.0, 0)];
        assert_eq!(h.topk(), expected);
    }

    /// Tests the timestamping behavior of `HeapFaiss` after multiple pushes.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 5.
    /// 2. Pushes three values into the heap.
    /// 3. Checks if the timestamp of the last pushed item is correctly set.
    ///
    /// Expected behavior:
    /// The timestamp of the last pushed item should be 2.
    #[test]
    fn test_timestamp_after_pushes() {
        let mut h = HeapFaiss::new(5);

        h.push(3.0);
        h.push(2.0);
        h.push(4.0);

        assert_eq!(h.topk().last().unwrap().1, 2);
    }

    /// Helper function to check the integrity of the `HeapFaiss` structure.
    ///
    /// This function ensures that the distances stored within the heap maintain the max-heap property.
    /// Specifically, for each distance in the heap, it ensures that the distance is greater than or equal
    /// to the distances of its left and right children, if they exist.
    ///
    /// # Parameters
    /// * `heap` - A reference to the `HeapFaiss` instance being checked.
    ///
    /// # Returns
    /// * `true` if the `HeapFaiss` maintains the max-heap property.
    /// * `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut h = HeapFaiss::new(5);
    /// h.push(3.0);
    /// h.push(5.0);
    /// h.push(2.0);
    /// assert!(maintains_heap_property(&h)); // This will return true.
    /// ```
    fn maintains_heap_property(heap: &HeapFaiss) -> bool {
        for i in 0..heap.distances.len() {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < heap.distances.len() && heap.distances[i] < heap.distances[left] {
                return false;
            }
            if right < heap.distances.len() && heap.distances[i] < heap.distances[right] {
                return false;
            }
        }
        true
    }

    /// Tests the integrity of the `HeapFaiss` property after multiple pushes.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 5.
    /// 2. Pushes a series of values.
    /// 3. Checks if the max-heap property is maintained.
    ///
    /// Expected behavior:
    /// The heap should maintain the max-heap property after every push.
    #[test]
    fn test_heap_integrity_after_pushes() {
        let mut h = HeapFaiss::new(8);

        h.push(3.0);
        h.push(2.0);
        h.push(-2.0);
        h.push(4.0);
        h.push(1.0);
        h.push(-1.0);
        h.push(0.0);
        h.push(5.0);

        assert!(maintains_heap_property(&h));
    }

    /// Tests the behavior of `HeapFaiss` when it's partially filled.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 1000.
    /// 2. Pushes three values into the heap.
    /// 3. Checks the number of elements in the heap and the max-heap property.
    ///
    /// Expected behavior:
    /// The heap should contain only three elements and maintain the max-heap property.
    #[test]
    fn test_partial_fill_heap() {
        let mut h = HeapFaiss::new(1000);

        h.push(3.0);
        h.push(2.0);
        h.push(4.0);

        assert_eq!(h.topk().len(), 3);
        assert!(maintains_heap_property(&h));
    }

    // =======================================
    // 3. Boundary and Special Value Cases
    // =======================================

    /// Tests the behavior of `HeapFaiss` when encountering values near overflow.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 2.
    /// 2. Pushes the maximum finite positive value representable in `f32`.
    /// 3. Pushes the second largest finite positive value representable in `f32`.
    /// 4. Retrieves and checks for the presence of the maximum value in the results.
    ///
    /// Expected behavior:
    /// The results should contain the maximum positive finite value that was pushed.
    #[test]
    fn test_near_overflow() {
        let mut h = HeapFaiss::new(2);
        h.push(f32::MAX);
        h.push(f32::MAX - 1.0);

        assert!(h.topk().iter().any(|(d, _)| *d == f32::MAX));
    }

    /// Tests the behavior of `HeapFaiss` when encountering values near underflow.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 2.
    /// 2. Pushes the maximum finite negative value representable in `f32`.
    /// 3. Pushes the second largest finite negative value representable in `f32`.
    /// 4. Retrieves and checks for the presence of the maximum negative value in the results.
    ///
    /// Expected behavior:
    /// The results should contain the maximum negative finite value that was pushed.
    #[test]
    fn test_near_underflow() {
        let mut h = HeapFaiss::new(2);
        h.push(f32::MIN);
        h.push(f32::MIN + 1.0);

        assert!(h.topk().iter().any(|(d, _)| *d == f32::MIN));
    }

    // ====================
    // 4. Large Data Tests
    // ====================

    /// Tests the behavior of `HeapFaiss` when handling a large number of distances.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a large capacity.
    /// 2. Pushes a large number of sequential distances.
    /// 3. Checks that the first entry in the results is the smallest distance.
    ///
    /// Expected behavior:
    /// The `HeapFaiss` should handle a large number of distances without issues, and the first entry
    /// in the results should be the smallest pushed distance.
    #[test]
    fn test_push_large_data() {
        let n = 1000;
        let mut h = HeapFaiss::new(n);

        for i in 0..n {
            h.push(i as f32);
        }

        let expected_first_entry = (0.0, 0);
        assert_eq!(h.topk().first().unwrap(), &expected_first_entry);
    }

    /// Tests the behavior of `HeapFaiss` when handling a medium number of distances.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a mid capacity.
    /// 2. Pushes a medium number of sequential distances.
    /// 3. Checks that the first entry in the results is the smallest distance.
    ///
    /// Expected behavior:
    /// The `HeapFaiss` should handle a medium-sized number of distances without issues, and the first entry
    /// in the results should be the smallest pushed distance.
    #[test]
    fn test_push_medium_data() {
        let n = 100;
        let mut h = HeapFaiss::new(n);

        for i in 0..n {
            h.push(i as f32);
        }

        let expected_first_entry = (0.0, 0);
        assert_eq!(h.topk().first().unwrap(), &expected_first_entry);
    }

    /// Tests the behavior of `HeapFaiss` when handling a small number of distances.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a small capacity.
    /// 2. Pushes a small number of sequential distances.
    /// 3. Checks that the first entry in the results is the smallest distance.
    ///
    /// Expected behavior:
    /// The `HeapFaiss` should handle a small number of distances without issues, and the first entry
    /// in the results should be the smallest pushed distance.
    #[test]
    fn test_push_small_data() {
        let n = 10;
        let mut h = HeapFaiss::new(n);

        for i in 0..n {
            h.push(i as f32);
        }

        let expected_first_entry = (0.0, 0);
        assert_eq!(h.topk().first().unwrap(), &expected_first_entry);
    }

    /// Tests the behavior of `HeapFaiss` when extending with a large dataset.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 1000.
    /// 2. Extends the heap with 1000 sequential float values.
    /// 3. Checks if the smallest value in the heap is 0.0.
    ///
    /// Expected behavior:
    /// The smallest value in the heap should be 0.0 after extending with the large dataset.
    #[test]
    fn test_extend_data() {
        let n = 1000;
        let mut h = HeapFaiss::new(n);
        let data: Vec<f32> = (0..n).map(|x| x as f32).collect();

        h.extend(&data);
        assert_eq!(h.topk().first().unwrap().0, 0.0);
    }

    /// Tests the behavior of `HeapFaiss` when extending with a medium-sized dataset.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 100.
    /// 2. Extends the heap with 100 sequential float values.
    /// 3. Checks if the smallest value in the heap is 0.0.
    ///
    /// Expected behavior:
    /// The smallest value in the heap should be 0.0 after extending with the large dataset.
    #[test]
    fn test_extend_medium_data() {
        let n = 100;
        let mut h = HeapFaiss::new(n);
        let data: Vec<f32> = (0..n).map(|x| x as f32).collect();

        h.extend(&data);
        assert_eq!(h.topk().first().unwrap().0, 0.0);
    }

    /// Tests the behavior of `HeapFaiss` when extending with a small dataset.
    ///
    /// This test:
    /// 1. Initializes a `HeapFaiss` with a capacity of 10.
    /// 2. Extends the heap with 10 sequential float values.
    /// 3. Checks if the smallest value in the heap is 0.0.
    ///
    /// Expected behavior:
    /// The smallest value in the heap should be 0.0 after extending with the large dataset.
    #[test]
    fn test_extend_small_data() {
        let n = 10;
        let mut h = HeapFaiss::new(n);
        let data: Vec<f32> = (0..n).map(|x| x as f32).collect();

        h.extend(&data);
        assert_eq!(h.topk().first().unwrap().0, 0.0);
    }
}

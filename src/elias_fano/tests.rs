use super::*;

#[test]
fn test_select() {
    let v: Vec<usize> = vec![0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 1023];
    let ef = EliasFano::from(&v);
    for i in 0..v.len() {
        assert_eq!(ef.select(i), Some(v[i]));
    }
    assert_eq!(ef.select(v.len()), None);
}

#[test]
fn test_elias_fano_few_distinct_elements() {
    // Test for EliasFano with few distinct values and many duplicates
    // This reproduces a scenario that previously caused bugs
    let total_offsets = 902598;
    let increments = vec![(141207, 362350)];

    // Build offset array
    let mut offsets = vec![0; total_offsets + 1];
    let mut current_value = 0;
    let mut inc_idx = 0;

    for i in 1..=total_offsets {
        if inc_idx < increments.len() && i == increments[inc_idx].0 {
            current_value = increments[inc_idx].1;
            inc_idx += 1;
        }
        offsets[i] = current_value;
    }

    // Build EliasFano and verify all values
    let ef = EliasFano::from(&offsets);

    // Verify every position matches the original data
    for i in 0..offsets.len() {
        assert_eq!(
            ef.select(i),
            Some(offsets[i]),
            "EliasFano select({}) returned incorrect value",
            i
        );
    }

    // Verify out-of-bounds access returns None
    assert_eq!(ef.select(offsets.len()), None);
}

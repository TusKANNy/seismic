use super::*;

#[test]
fn test_select() {
    let arr = [0, 12, 33, 42, 55, 61, 62, 63, 128, 129, 254, 1023];
    let ef = EliasFano::from(&arr);
    for (i, v) in arr.into_iter().enumerate() {
        assert_eq!(ef.select(i), Some(v));
    }
    assert_eq!(ef.select(arr.len()), None);
}

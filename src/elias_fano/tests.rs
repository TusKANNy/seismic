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

use array::{SpanTrait, ArrayTrait};
use orion::numbers::{FixedTrait, FP16x16};
fn input() -> Array<FP16x16> {
    let mut arr = ArrayTrait::<FP16x16>::new();
    arr.append(FixedTrait::<FP16x16>::new(0, false));
    arr.append(FixedTrait::<FP16x16>::new(0, false));
    arr.append(FixedTrait::<FP16x16>::new(65536, false));
    arr.append(FixedTrait::<FP16x16>::new(0, false));
    arr.append(FixedTrait::<FP16x16>::new(0, false));
    arr.append(FixedTrait::<FP16x16>::new(0, false));
    arr
}
    
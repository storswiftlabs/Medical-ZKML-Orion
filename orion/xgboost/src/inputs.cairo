use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};

fn input() -> Tensor<FP16x16> {
    TensorTrait::new(
            array![1,6].span(),
            array![
                FP16x16 { mag: 61166, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 0, sign: false },
                ].span()
                // out 3
            
            // array![
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(65536, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(0, false)
            // ].span(),
            // // out 0

            // array![
            //     FixedTrait::<FP16x16>::new(15292, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(65536, false),
            //     FixedTrait::<FP16x16>::new(65536, false),
            //     FixedTrait::<FP16x16>::new(65536, false)
            // ].span(),
            // // out 1

            // array![
            //     FixedTrait::<FP16x16>::new(27307, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(65536, false),
            //     FixedTrait::<FP16x16>::new(65536, false),
            //     FixedTrait::<FP16x16>::new(0, false),
            //     FixedTrait::<FP16x16>::new(65536, false)
            // ].span()
            // // out 2
    )
}
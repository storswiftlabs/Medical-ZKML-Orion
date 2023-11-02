fn main() -> u8 {
    let _0_inputs = input();
    let _1_res = tree(_0_inputs);
    _1_res
}

use orion::numbers::FP16x16;
use orion::operators::ml::TreeClassifierTrait;
use core::array::{SpanTrait, ArrayTrait};
use decision_tree_ac::input::input;

fn tree(inputs: Array<FP16x16>) -> u8 {
    0
    // TODO
}

#[cfg(test)]
mod tests {
    use super::tree;
    use decision_tree_ac::input::input;
    #[test]
    #[available_gas(100000)]
    fn it_works() {
        let inputs = input();
        assert(tree(inputs) == 0, 'it works!');
    }
}

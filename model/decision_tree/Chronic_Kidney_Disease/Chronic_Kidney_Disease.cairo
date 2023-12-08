use core::integer::u32;
use dt::inputs::input;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};

fn main() -> u32 {
    let mut X = input();
    let class_ids: Span<usize> = array![0, 0, 0].span();
    let class_nodeids: Span<usize> = array![2, 3, 4].span();
    let class_treeids: Span<usize> = array![0, 0, 0].span();
    let class_weights: Span<FP16x16> = array![FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }].span();
    let classlabels: Span<usize> = array![0, 1].span();
    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 0].span();
    let nodes_featureids: Span<usize> = array![3, 17, 0, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![0, 0, 0, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF].span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![FP16x16 { mag: 8192, sign: false }, FP16x16 { mag: 21104, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }].span();
    let base_values: Option<Span<FP16x16>> = Option::Some(array![FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }].span());
    let post_transform = POST_TRANSFORM::SOFTMAX;

    let tree_ids: Span<usize> = array![0].span();
let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0);
let mut node_index: Felt252Dict<usize> = Default::default();
    node_index.insert(2089986280348253421170679821480865132823066470938446095505822317253594081284, 0);
    node_index.insert(2001140082530619239661729809084578298299223810202097622761632384561112390979, 1);
    node_index.insert(2592670241084192212354027440049085852792506518781954896144296316131790403900, 2);
    node_index.insert(2960591271376829378356567803618548672034867345123727178628869426548453833420, 3);
    node_index.insert(458933264452572171106695256465341160654132084710250671055261382009315664425, 4);

    let atts = TreeEnsembleAttributes {
        nodes_falsenodeids,
        nodes_featureids,
        nodes_missing_value_tracks_true,
        nodes_modes,
        nodes_nodeids,
        nodes_treeids,
        nodes_truenodeids,
        nodes_values
    };

    let mut ensemble: TreeEnsemble<FP16x16> = TreeEnsemble {
        atts, tree_ids, root_index, node_index
    };

    let mut classifier: TreeEnsembleClassifier<FP16x16> = TreeEnsembleClassifier {
        ensemble,
        class_ids,
        class_nodeids,
        class_treeids,
        class_weights,
        classlabels,
        base_values,
        post_transform
    };

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);
    let mut arr: Span<usize> = labels;
    let a: usize = *arr.pop_front().unwrap();
    a
}
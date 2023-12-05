use core::array::SpanTrait;
use decision_tree_ac::input::input;
use debug::PrintTrait;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::tensor::implementations::tensor_fp16x16::relative_eq;

fn main() {
    let mut X = input();
    let class_ids: Span<usize> = array![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].span();
    let class_nodeids: Span<usize> = array![2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 10, 10, 10, 10].span();
    let class_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let class_weights: Span<FP16x16> = array![FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }].span();
    let classlabels: Span<usize> = array![0, 1, 2, 3].span();
    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 6, 0, 8, 0, 10, 0, 0].span();
    let nodes_featureids: Span<usize> = array![2, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF].span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 5, 0, 7, 0, 9, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }].span();
    let base_values: Option<Span<FP16x16>> = Option::None;
    let post_transform = POST_TRANSFORM::NONE;

    let tree_ids: Span<usize> = array![0].span();
let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0);
let mut node_index: Felt252Dict<usize> = Default::default();
    node_index.insert(2089986280348253421170679821480865132823066470938446095505822317253594081284, 0);
    node_index.insert(2001140082530619239661729809084578298299223810202097622761632384561112390979, 1);
    node_index.insert(2592670241084192212354027440049085852792506518781954896144296316131790403900, 2);
    node_index.insert(2960591271376829378356567803618548672034867345123727178628869426548453833420, 3);
    node_index.insert(458933264452572171106695256465341160654132084710250671055261382009315664425, 4);
    node_index.insert(3344223123784052057366048933846905716067140384361791026153972616805110454637, 5);
    node_index.insert(658476905110174425295568215706634733332002869979287079110965040248935650599, 6);
    node_index.insert(2836212335642438363012490794290757623813171043187182819737087983331902926990, 7);
    node_index.insert(3496601277869056110810900082189273917786762659443522403285387602989271154262, 8);
    node_index.insert(1249294489531540970169611621067106471309281870082955806338234725206665112557, 9);
    node_index.insert(2161697998033672097816961828039488190903838124365465380011173778905747857792, 10);

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
    
    // ASSERT LABELS
    // labels.len().print();
    let mut arr: Span<usize> = labels;
    let a: usize = *arr.pop_front().unwrap();
    a.print();
    // assert(*labels[0] == 0, 'labels[0]');
    // assert(labels.len() == 1, 'len(labels)');

    // ASSERT SCORES
    @scores.get(0, 0).unwrap().mag.print();
    @scores.get(0, 1).unwrap().mag.print();
    @scores.get(0, 2).unwrap().mag.print();
    @scores.get(0, 3).unwrap().mag.print();
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 1]'
    );
    assert(
        relative_eq(@scores.get(0, 2).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 2]'
    );
    assert(
        relative_eq(@scores.get(0, 3).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 3]'
    );
}
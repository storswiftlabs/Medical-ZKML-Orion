use core::integer::u32;
use dt::inputs::input;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};

fn main() -> u32 {
    let mut X = input();
    let class_ids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let class_nodeids: Span<usize> = array![2, 3, 9, 12, 13, 14, 15, 16, 19, 20, 21, 23, 24].span();
    let class_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let class_weights: Span<FP16x16> = array![FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }].span();
    let classlabels: Span<usize> = array![0, 1].span();
    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 22, 17, 16, 15, 10, 0, 14, 13, 0, 0, 0, 0, 0, 21, 20, 0, 0, 0, 24, 0, 0].span();
    let nodes_featureids: Span<usize> = array![21, 11, 0, 0, 12, 20, 1, 0, 16, 0, 15, 20, 0, 0, 0, 0, 0, 15, 3, 0, 0, 0, 18, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF].span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 5, 6, 7, 8, 9, 0, 11, 12, 0, 0, 0, 0, 0, 18, 19, 0, 0, 0, 23, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![FP16x16 { mag: 8129, sign: false }, FP16x16 { mag: 231, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 6315, sign: false }, FP16x16 { mag: 23766, sign: false }, FP16x16 { mag: 15432, sign: false }, FP16x16 { mag: 18091, sign: false }, FP16x16 { mag: 26322, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 44327, sign: false }, FP16x16 { mag: 14542, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 34134, sign: false }, FP16x16 { mag: 7629, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 17654, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }].span();
    let base_values: Option<Span<FP16x16>> = Option::Some(array![FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }].span());
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
    node_index.insert(3344223123784052057366048933846905716067140384361791026153972616805110454637, 5);
    node_index.insert(658476905110174425295568215706634733332002869979287079110965040248935650599, 6);
    node_index.insert(2836212335642438363012490794290757623813171043187182819737087983331902926990, 7);
    node_index.insert(3496601277869056110810900082189273917786762659443522403285387602989271154262, 8);
    node_index.insert(1249294489531540970169611621067106471309281870082955806338234725206665112557, 9);
    node_index.insert(2161697998033672097816961828039488190903838124365465380011173778905747857792, 10);
    node_index.insert(1129815197211541481934112806673325772687763881719835256646064516195041515616, 11);
    node_index.insert(2592593088135949192377729543480191336537305484235681164569491942155715064163, 12);
    node_index.insert(578223957014284909949571568465953382377214912750427143720957054706073492593, 13);
    node_index.insert(1645617302026197421098102802983206579163506957138012501615708926120228167528, 14);
    node_index.insert(2809438816810155970395166036110536928593305127049404137239671320081144123490, 15);
    node_index.insert(2496308528011391755709310159103918074725328650411689040761791240500618770096, 16);
    node_index.insert(2003594778587446957576114348312422277631766150749194167061999666337236425714, 17);
    node_index.insert(2215681478480673835576618830034726157921200517935329010004363713426342305479, 18);
    node_index.insert(3185925835074464079989752015681272863271067691852543168049845807561733691707, 19);
    node_index.insert(1207265836470221457484062512091666004839070622130697586496866096347024057755, 20);
    node_index.insert(1870230949202979679764944800468118671928852128047695497376875566624821494262, 21);
    node_index.insert(618060852536781954395603948693216564334274573299243914053414488061601327758, 22);
    node_index.insert(232760707548494477255512699093366059519467428168757247456690480397246371463, 23);
    node_index.insert(1617386247965480308136742715422077429967341022950306068917456849194882895900, 24);

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
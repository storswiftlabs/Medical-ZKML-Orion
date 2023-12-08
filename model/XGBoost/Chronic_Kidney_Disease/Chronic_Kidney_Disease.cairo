use core::integer::u32;
use xgboost::inputs::input;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};

fn main() -> u32 {
    let mut X = input();
    let class_ids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let class_nodeids: Span<usize> = array![2, 3, 4, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2].span();
    let class_treeids: Span<usize> = array![0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9].span();
    let class_weights: Span<FP16x16> = array![FP16x16 { mag: 34695, sign: false }, FP16x16 { mag: 9830, sign: false }, FP16x16 { mag: 37665, sign: true }, FP16x16 { mag: 28844, sign: true }, FP16x16 { mag: 28037, sign: false }, FP16x16 { mag: 24270, sign: false }, FP16x16 { mag: 24925, sign: true }, FP16x16 { mag: 20172, sign: false }, FP16x16 { mag: 22947, sign: true }, FP16x16 { mag: 19635, sign: false }, FP16x16 { mag: 20691, sign: true }, FP16x16 { mag: 19338, sign: true }, FP16x16 { mag: 18285, sign: false }, FP16x16 { mag: 15390, sign: false }, FP16x16 { mag: 19021, sign: true }, FP16x16 { mag: 14415, sign: false }, FP16x16 { mag: 17966, sign: true }, FP16x16 { mag: 14777, sign: false }, FP16x16 { mag: 16121, sign: true }, FP16x16 { mag: 12117, sign: false }, FP16x16 { mag: 16189, sign: true }].span();
    let classlabels: Span<usize> = array![0, 1].span();
    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0].span();
    let nodes_featureids: Span<usize> = array![15, 2, 0, 0, 0, 3, 0, 0, 14, 0, 0, 15, 0, 0, 14, 0, 0, 3, 0, 0, 14, 0, 0, 15, 0, 0, 17, 0, 0, 14, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF].span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![FP16x16 { mag: 45875, sign: false }, FP16x16 { mag: 57344, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 8192, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 43244, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 45875, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 43244, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 8192, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 44359, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 45875, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 25547, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 44359, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }].span();
    let base_values: Option<Span<FP16x16>> = Option::None;
    let post_transform = POST_TRANSFORM::SOFTMAX;

    let tree_ids: Span<usize> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].span();
let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0);
    root_index.insert(1, 5);
    root_index.insert(2, 8);
    root_index.insert(3, 11);
    root_index.insert(4, 14);
    root_index.insert(5, 17);
    root_index.insert(6, 20);
    root_index.insert(7, 23);
    root_index.insert(8, 26);
    root_index.insert(9, 29);
let mut node_index: Felt252Dict<usize> = Default::default();
    node_index.insert(2089986280348253421170679821480865132823066470938446095505822317253594081284, 0);
    node_index.insert(2001140082530619239661729809084578298299223810202097622761632384561112390979, 1);
    node_index.insert(2592670241084192212354027440049085852792506518781954896144296316131790403900, 2);
    node_index.insert(2960591271376829378356567803618548672034867345123727178628869426548453833420, 3);
    node_index.insert(458933264452572171106695256465341160654132084710250671055261382009315664425, 4);
    node_index.insert(1089549915800264549621536909767699778745926517555586332772759280702396009108, 5);
    node_index.insert(1321142004022994845681377299801403567378503530250467610343381590909832171180, 6);
    node_index.insert(2592987851775965742543459319508348457290966253241455514226127639100457844774, 7);
    node_index.insert(1637368371864026355245122316446106576874611007407245016652355316950184561542, 8);
    node_index.insert(1207699383798263883125605407307435965808923448511613904826718551574712750645, 9);
    node_index.insert(1180550645873507273865212362837104046225859416703538577277065670066180087996, 10);
    node_index.insert(936823097115478672163131070534991867793647843312823827742596382032679996195, 11);
    node_index.insert(2908682032041418908903105681227249033483541201006723240850136728317167492227, 12);
    node_index.insert(576657123605396437968823113955952586959670965011232700393892413073919304299, 13);
    node_index.insert(469486474782544164430568959439120883383782181399389907385047779197726806430, 14);
    node_index.insert(3512521406437956009189089258567111789473785799907488469636118378769715425964, 15);
    node_index.insert(2556139128341700567231916301725351155453738182692001295135848448652163014397, 16);
    node_index.insert(2941083907689010536497253969578701440794094793277200004061830176674600429738, 17);
    node_index.insert(3515557115945123685249720924176918246289668839127088842764552624387741006658, 18);
    node_index.insert(1086529665842980980708131000626441572702884934518178946087373814825190924452, 19);
    node_index.insert(2741690337285522037147443857948052150995543108052651970979313688522374979162, 20);
    node_index.insert(2650761223990278311391161549562769924185646525933398576366599729646470331982, 21);
    node_index.insert(2394106477146207452133044987332160354989937730594050309156799249985649638789, 22);
    node_index.insert(2258442912665439649622769515993460039756024697697714582745734598954638194578, 23);
    node_index.insert(1923650700608380821616803627552990459031020321822263486178231314533355655733, 24);
    node_index.insert(2986518118017342969503780420947472437589402926077471462482848861272477228312, 25);
    node_index.insert(2743794648056839147566190792738700325779538550063233531691573479295033948774, 26);
    node_index.insert(2798268708043007987823290469469057887013592827991152425130480624165644530309, 27);
    node_index.insert(2656358495835759095543325181783754425097697418395968068218877742551382200798, 28);
    node_index.insert(3149011590233272225803080114059308917528748800879621812239443987136907759492, 29);
    node_index.insert(1256253793249097778491586851395915710675714820964284887367516531463100834885, 30);
    node_index.insert(1367204028253064270272780398308197116331885826019160095546179574252301715156, 31);

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
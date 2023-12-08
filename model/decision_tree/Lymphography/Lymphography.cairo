use core::integer::u32;
use dt::inputs::input;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};

fn main() -> u32 {
    let mut X = input();
    let class_ids: Span<usize> = array![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].span();
    let class_nodeids: Span<usize> = array![3, 3, 3, 3, 6, 6, 6, 6, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 14, 14, 14, 14, 16, 16, 16, 16, 17, 17, 17, 17, 19, 19, 19, 19, 21, 21, 21, 21, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 34, 34, 34, 34, 35, 35, 35, 35, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42].span();
    let class_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let class_weights: Span<FP16x16> = array![FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 65536, sign: false }].span();
    let classlabels: Span<usize> = array![0, 1, 2, 3].span();
    let nodes_falsenodeids: Span<usize> = array![26, 11, 4, 0, 10, 7, 0, 9, 0, 0, 0, 25, 18, 15, 0, 17, 0, 0, 20, 0, 22, 0, 24, 0, 0, 0, 36, 33, 32, 31, 0, 0, 0, 35, 0, 0, 42, 41, 40, 0, 0, 0, 0].span();
    let nodes_featureids: Span<usize> = array![1, 14, 0, 0, 5, 17, 0, 16, 0, 0, 0, 8, 7, 15, 0, 9, 0, 0, 10, 0, 15, 0, 14, 0, 0, 0, 17, 12, 12, 17, 0, 0, 0, 17, 0, 0, 8, 13, 10, 0, 0, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::BRANCH_LEQ, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF].span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 3, 0, 5, 6, 0, 8, 0, 0, 0, 12, 13, 14, 0, 16, 0, 0, 19, 0, 21, 0, 23, 0, 0, 0, 27, 28, 29, 30, 0, 0, 0, 34, 0, 0, 37, 38, 39, 0, 0, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 16384, sign: false }, FP16x16 { mag: 10922, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 4681, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 49152, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 10922, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 49152, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 49152, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 23405, sign: false }, FP16x16 { mag: 54613, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 4681, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 4681, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 16384, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 49152, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }].span();
    let base_values: Option<Span<FP16x16>> = Option::Some(array![FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }, FP16x16 { mag: 65536, sign: false }].span());
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
    node_index.insert(654822874782506608656472905579051041410086644071534146326024101025575400153, 25);
    node_index.insert(525638101901638132526332140778087078272370083489998903571807698910013602668, 26);
    node_index.insert(3091640181556387972179279087539287892670640556085669903494551919685982442095, 27);
    node_index.insert(1425411460578159050163131982087304445715005458700346341117759372943452688022, 28);
    node_index.insert(1722933265299553894839124723076027659619615015638971980461286818493531809034, 29);
    node_index.insert(3325117385742592388671007840076299062858228097051060057749225651290693960897, 30);
    node_index.insert(1869273998012404873272699831805499731567895666937555882116307079956228100456, 31);
    node_index.insert(257262395234910825879033951801423835835630270967846664413154594520703929530, 32);
    node_index.insert(2891500475385583315757684141371327604925143655360011721762142660942782195029, 33);
    node_index.insert(1257459981124043271342269816753070228024611695909553991758648317372015085782, 34);
    node_index.insert(3573101724490615587655146760489247477770015274618159524231872921394794809579, 35);
    node_index.insert(2951401777594449283985541406642940553317465718696638438535370997641527993378, 36);
    node_index.insert(2436860863451320452900512817385686838091627966322316039332239784330434600829, 37);
    node_index.insert(3257977356974702770994741663931928753019715185508521958836925918758890988390, 38);
    node_index.insert(2741853283805093821434776875305720302351684616683152528499335618682018880592, 39);
    node_index.insert(514567459251558911686762246500770717674979116530125263461114578537254680672, 40);
    node_index.insert(2119374930171040799805795099091470687208894498354655018353474015395489390434, 41);
    node_index.insert(3338470191188327918255138125570464269857839379813971679216902484398948556964, 42);

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
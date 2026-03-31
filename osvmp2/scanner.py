import os
import sys
import warnings
import psutil
import shutil
import h5py
import itertools
from operator import itemgetter
import numpy as np
import pyscf
from pyscf import gto, scf, lib
from pyscf.df import addons
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc import df as pbc_df
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from osvmp2.mm.solvation import get_eps
from numpy.linalg import eigh, multi_dot
from osvmp2.__config__ import inputs, ngpu
from osvmp2 import hf_grad, mp2_ene
from osvmp2.hf_ene import scf_parallel
if inputs["qm_atoms"] is not None:
    from osvmp2.mm import qmmm
from osvmp2.get_mol_special import get_aux_molpro, get_aux
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size 
nnode = nrank//comm_shm.size # number of nodes

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)
if ngpu:
    import cupy
    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]

def get_name(xyz):
    if '/' in xyz:
        rev_count = np.arange(1,len(xyz), dtype='i')*(-1)
        for i in rev_count:
            if xyz[i] == '/': break
        xyz_name = xyz[i+1:-4]
    else:
        xyz_name = xyz[:-4]
    return xyz_name

def read_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        natom = int(lines[0])
        coord = ""
        for l in lines[2:2+natom]:
            coord += l
    return natom, coord

def get_atoms(mol):
   atom_list = []
   for ai in mol._atom:
      atom_list.append(ai[0])
   return atom_list

def rmsd(g_osv, log):
   #benzidine(def2-tzvp)
   #FC
   #g_ri = [-0.00121129142, 0.000742865499, 0.0218564817, -0.00953330485, 0.00192245719, -0.00401905595, -4.82842077e-06, -1.13224723e-06, -0.0519754759, 0.00953338739, -0.00192294798, -0.00401910848, 0.0012111912, -0.000742538625, 0.0218560638, -2.28315356e-07, 1.04696004e-07, 0.0280044113, -8.89982366e-06, -2.34402189e-06, 0.0519475527, -0.00953840878, -0.00192463423, 0.00412363788, -0.00121375214, -0.000743577557, -0.0219111342, 8.29941569e-06, 2.25251872e-06, -0.0280082637, 0.00120930208, 0.000742154136, -0.0218600904, 0.00953169903, 0.00192237149, 0.0040239576, 3.07085775e-08, -3.47195551e-08, -0.0506349974, -1.07441751e-06, -2.62196168e-07, 0.0506354604, 0.00103346407, -0.000426933004, -0.00311512807, -0.00021984602, -0.000351810101, 0.00504651465, 0.000219711604, 0.000351631737, 0.00504659867, -0.00103344816, 0.000426931635, -0.00311521166, -0.000202591408, 0.000356975487, -0.00506387152, 0.0010339155, 0.000427002079, 0.00311350474, -0.00103367544, -0.000426976869, 0.00311499338, 0.00022033357, -0.000351550305, -0.00504678314, 0.0233334269, -0.00609275789, -0.00449765089, -0.0233334313, 0.00609274783, -0.00449765669, -0.0233334718, -0.00609276313, 0.00449760332, 0.0233334909, 0.00609276859, 0.00449764784]
   #no_FC
   #g_ri = [-0.0012102775302640367, 0.0007424577374575136, 0.021857046199658114, -0.009531009794304879, 0.0019223740773623987, -0.004019710650385977, -2.8051078305871627e-14, 1.422758276887733e-14, -0.051967486517517814, 0.009531009794342182, -0.0019223740773348652, -0.0040197106506251745, 0.0012102775301734425, -0.000742457737513913, 0.02185704619983664, -5.379793815697654e-15, -1.369382506304415e-14, 0.02800281609225852, 6.921721496747774e-14, 8.649282631529188e-14, 0.0519674865170664, -0.009531009794692125, -0.001922374077448552, 0.004019710650768726, -0.001210277529876791, -0.0007424577373877916, -0.021857046199619923, -7.286662592753679e-14, -3.822815545020952e-14, -0.028002816092484117, 0.001210277529962056, 0.0007424577374575136, -0.02185704619973272, 0.00953100979454824, 0.0019223740774041431, 0.00401971065077672, 4.3225305301741154e-14, 1.0155402368619768e-14, -0.05063418355113214, 1.781994690697175e-14, -9.869709216570044e-15, 0.050634183551027334, 0.0010338212163882865, -0.0004270436229866226, -0.0031153618033568087, -0.00021936799103805527, -0.0003518518059064668, 0.005046842869256052, 0.0002193679910238444, 0.0003518518058965858, 0.005046842869295243, -0.001033821216400721, 0.0004270436230076058, -0.003115361803328831, -0.00021936799106692106, 0.0003518518058781561, -0.005046842869280033, 0.0010338212163536475, 0.0004270436229916186, 0.0031153618033008534, -0.001033821216400721, -0.00042704362299550436, 0.003115361803328387, 0.00021936799105981564, -0.00035185180590668885, -0.005046842869283696, 0.02333255658018052, -0.006092530520781625, -0.004498086767621867, -0.023332556580194286, 0.006092530520781736, -0.004498086767624088, -0.02333255658017741, -0.006092530520771855, 0.0044980867676280845, 0.023332556580187624, 0.006092530520777295, 0.004498086767650289]
   #Frozen(SVP)
   #g_ri = [0.002566310833861607, 0.00042636072051660534, 0.02283756819292382, -0.009859595966688062, 0.001009166810707729, -0.003411080475694328, 1.4801735581844184e-15, -1.1465533029659737e-13, -0.0533546983348997, 0.009859595966592138, -0.001009166810563844, -0.003411080475605288, -0.0025663108337745655, -0.00042636072057744556, 0.022837568192934476, -1.3904183091530187e-14, 1.6439976261186553e-14, 0.021608716661022598, 1.1784342490543664e-14, 7.338651780990253e-14, 0.05335469833480866, -0.009859595966801749, -0.001009166810753026, 0.0034110804757749857, 0.002566310833936214, -0.00042636072050017404, -0.02283756819280569, 5.6433102548630876e-14, 1.5199470913655755e-14, -0.021608716661243754, -0.0025663108339895047, 0.0004263607205015063, -0.022837568192835, 0.009859595966736023, 0.0010091668106873009, 0.00341108047573796, 5.603204071825773e-16, 4.168195815030488e-15, -0.05305803956154875, -2.4872370330865368e-14, -7.110343536826866e-15, 0.053058039561571846, 0.006320336834353402, -0.0018707395596759824, -0.006478982317609683, 0.00541944317641585, -0.001951471420315154, 0.0080515637790326, -0.005419443176400751, 0.0019514714203124894, 0.008051563778995408, -0.006320336834380047, 0.0018707395596705423, -0.006478982317587478, 0.005419443176463812, 0.0019514714203331396, -0.008051563779058357, 0.006320336834360951, 0.001870739559679202, 0.0064789823175881445, -0.00632033683435429, -0.0018707395596713194, 0.006478982317566384, -0.005419443176440719, -0.0019514714203362482, -0.008051563779023163, 0.02076218570193289, -0.005426090674686246, -0.0061658355759188765, -0.02076218570193733, 0.005426090674691797, -0.006165835575916656, -0.02076218570195021, -0.005426090674696793, 0.006165835575931311, 0.02076218570195021, 0.005426090674698125, 0.0061658355759237615]
   #Frozen(TZVP)
   #g_ri = [0.00011404831810679639, 0.00043778130258598225, 0.021112060349066653, -0.00870279957789144, 0.0016928288235988909, -0.0034226746573267075, 4.854546018162264e-13, 2.6829881769412e-14, -0.05148458430581371, 0.008702799577593012, -0.001692828823385728, -0.003422674657550584, -0.00011404831836259177, -0.000437781303271656, 0.02111206034914126, 6.04468540371604e-14, 1.6203670133411843e-13, 0.02831199144516816, -2.1215044415298086e-13, 7.46363616147165e-14, 0.05148458430594649, -0.008702799577012144, -0.001692828824416015, 0.0034226746567982413, 0.00011404831856864917, -0.00043778130242477786, -0.021112060348366768, 6.448234344572905e-13, 1.718562773969278e-13, -0.028311991444921247, -0.00011404831916195235, 0.0004377813020481902, -0.021112060348752237, 0.008702799576992604, 0.0016928288241397915, 0.0034226746564445243, 4.8077028079793584e-14, 7.10614429687547e-14, -0.052808098283474436, -6.118010557800563e-14, -2.438193147814155e-14, 0.052808098283714244, 0.0015309473342743907, -0.0005630631406459141, -0.0034038867050740773, 0.0002766111258512005, -0.000494099645072521, 0.00529136338850722, -0.0002766111257614945, 0.0004940996451198165, 0.005291363388588821, -0.0015309473341997837, 0.0005630631409204723, -0.003403886705047432, 0.00027661112554033807, 0.0004940996454002589, -0.0052913633885958156, 0.0015309473342046687, 0.0005630631408123365, 0.003403886704983261, -0.0015309473340061608, -0.0005630631405622033, 0.0034038867051084942, -0.000276611125634485, -0.0004940996453582924, -0.005291363388470249, 0.023239031611241412, -0.006068100295633871, -0.0046370220807356866, -0.023239031611308913, 0.006068100295613665, -0.004637022080808961, -0.02323903161123919, -0.006068100295595569, 0.004637022080739239, 0.02323903161123031, 0.006068100295626655, 0.004637022080771658]
   #benzene(ccpvtz)
   #g_ri = [0.005081305276623205, -0.0006811561073099826, 0.008868431493402884, -0.00022427953533465939, 2.4635832649777534e-05, -0.00028995002245668644, 0.00015894006990002651, -2.1665531034498198e-06, -8.006526207449127e-05, -0.005182612997098213, -0.0002571443473714796, 0.008753604188863395, 0.00038039676012857626, -7.81906393815257e-06, -0.00014359839420394205, -0.010269467234831353, 0.00041868624800833754, -1.0957202940899136e-05, 0.00014444609562946908, -1.6183730679464325e-05, 0.00019290492395551695, -0.00501679707374314, 0.0006785176640976831, -0.008868339735118802, -0.00019446466776162907, -7.56390078607283e-06, 0.00028960168060887526, 0.005150142827733184, 0.00026091205194384015, -0.008799611459786583, -0.0002802998960245162, 1.2226221308975216e-05, -1.541657193710605e-05, 0.0102526903744522, -0.00042294431473945127, 0.00010339636142926018]
   #benzene(svp)
   #g_ri = [-0.002828651748680855, -0.001633121828089834, 1.811851741889348e-16, -0.0028286517487039475, 0.0016331218280933868, -1.469156888097237e-16, -1.0506523824557504e-14, 0.0032662462995194375, -6.126055949346416e-16, 0.0028286517487137175, 0.001633121828099604, 3.0015040132587154e-16, 0.0028286517486781904, -0.0016331218280800641, -2.7991691090632583e-16, 3.605924875938955e-15, -0.003266246299531872, -4.1080996146916377e-16, 0.007080767952142519, 0.004088083251805763, -1.1838572086048195e-16, 0.007080767952154954, -0.0040880832518077614, 1.7445400150620834e-16, -8.873767747474916e-16, -0.008176166950760066, 3.871235576301098e-16, -0.007080767952135414, -0.004088083251804653, 6.739433967672929e-17, -0.007080767952134526, 0.004088083251806429, 1.4800625419278666e-16, 2.145596222134705e-15, 0.008176166950748964, 3.103201484596963e-16]
   #frozen
   #g_ri = [-0.0023756482465442375, -0.0013715801305771436, -1.1312126364279116e-16, -0.002375648246570883, 0.001371580130624217, 1.8216537159598434e-16, -8.428175384887737e-15, 0.0027431629123153556, -8.332229384850203e-16, 0.002375648246556672, 0.0013715801305824726, 2.1011440352356218e-16, 0.0023756482465309148, -0.001371580130604677, -3.096905792351737e-16, 3.122714158168398e-15, -0.0027431629123046974, -1.6034307391186577e-16, 0.007790288609400342, 0.004497725194415381, -6.809964414005694e-17, 0.007790288609418106, -0.004497725194432256, 4.1032949758194864e-16, 2.4122923276771722e-15, -0.008995450836613905, 2.2557734721097087e-16, -0.007790288609395901, -0.004497725194421154, 1.2544674561665443e-17, -0.007790288609381246, 0.0044977251944093855, 2.789152978812232e-16, 7.347923822446175e-16, 0.008995450836613017, 1.6483090705976167e-16]
  #Baker 02
   #Frozen
   #g_ri = [-1.3728081333471517e-15, 2.987620755848719e-16, 0.09894121676219125, -3.919097289018115e-15, -1.6785216834067305e-16, -0.09894121676663659, 2.8967193758844144e-15, 1.0372549537213498e-17, 0.05178116647127773, 2.3951860464474502e-15, -1.4128245678166814e-16, -0.05178116646684083]
   #ethanol
   #g_ri = [0.004233667447182121, 0.001086298744468639, -9.494141584021065e-15, 0.009645663869072285, 0.010538965318753313, -1.6201276431537792e-13, 0.005258410647947187, -0.0020351578738733167, 2.9306765347847374e-13, -0.008086999958723862, 0.006657753650294307, -1.7236835873554734e-13, -0.0019654529724117964, -0.0015312031879611832, 1.0561755463267541e-14, 0.0005909329712878164, -0.0008019519083153348, -0.002301338027790978, 0.0005909329712847633, -0.0008019519083060089, 0.0023013380278038564, -0.0051335774872920265, -0.006556376417808485, -0.0034152769726869536, -0.005133577487197366, -0.006556376417784726, 0.0034152769727151533]
   #water2(with DIIS)
   #g_ri = [0.07851618928702031, 0.0013507633869102031, -0.026771640647547112, -0.05702012277320012, -0.001640107028346438, -0.027860560349078467, -0.022502203368449436, -0.0003469852817897731, 0.05528287173380653, 0.07511001160830982, 0.000886536199405652, -0.027154556882560188, -0.053212512393799205, -1.4284583033364467e-05, -0.031964122114773064, -0.020891362359848675, -0.00023592269350102002, 0.05846800826012899]
   #water3
   #g_ri = [-0.0004912235197913617, -0.0002518007543876877, 0.00030591259583667707, 0.0013897255958941201, 0.00017689243349575112, 0.0003275687145600781, -0.0002911314758660133, 0.0006182084783633024, -8.736603302317292e-05, 0.0008662280829230262, -0.0006098972955705939, -0.0002113150777498518, -0.000991223556655374, 0.0004281995469257449, -0.00021791029254947247, -0.0005878062306876597, -0.0004102503684377812, 0.0009119488015401078, -5.109878076337004e-06, 0.0011733150348653965, 0.00015690573271109898, 4.237733176593039e-06, -0.000676252517026299, -0.001017541210099171, 0.00010630325021754317, -0.00044841455749877746, -0.00016820323115940283]
   #g_ri = [-0.0004843239671492583, -0.0002574907794734571, 0.000305890304143297, 0.001376390081041512, 0.00017414399055909469, 0.0003294585984521259, -0.00029583741108885064, 0.0006133613336903043, -9.334019784801306e-05, 0.0008675492400489482, -0.0006087603366746741, -0.00020679424395231294, -0.0009830762276304972, 0.0004313829200928154, -0.0002168934283166557, -0.0005853370315215267, -0.00040914891307708423, 0.0008938485466090107, -1.581496070157673e-05, 0.0011779582716373582, 0.00015379001755988497, 1.4561024928694266e-05, -0.0006715530837266837, -0.001005976699744976, 0.00010588925326082688, -0.0004498934023272616, -0.00015998289681196098]
   #water2
   #g_ri = [0.07851631420971827, 0.0013500430258001628, -0.026769596029470488, -0.057020936333118666, -0.0016397346939528573, -0.027862495792515984, -0.02250156935504316, -0.00034657563379129896, 0.05528327345910222, 0.07510634576881026, 0.000885287796542289, -0.02715265617981011, -0.05321122735936212, -1.3496610304856993e-05, -0.03196667080503146, -0.02088892693097133, -0.00023552388464441465, 0.05846814534773426]
   #water(tzvp)
   #g_ri = [-1.0032108894775238e-05, 1.0792251748564946e-15, -0.010817787593916606, -0.004285117589770682, -2.504316392334048e-16, 0.005412493475202451, 0.004295149698664957, -8.287935356233257e-16, 0.0054052941192368475]
   #water5
   #g_ri = [0.024926903668930134, 0.01050202007847112, 0.015912804681751602, 0.014358655154276256, -0.021665066727988957, -0.017265363927020827, -0.01580685365143175, -0.020315661906328053, 0.01773215035852549, -0.023200592170526835, 0.008474714670072103, -0.019124580663733903, -0.0028665315771068123, 0.02571626141113592, 0.0177321540388653, -0.0037997803681992515, -0.013183231826065178, -0.014577048631230438, -0.010702511570005457, 0.0006798366256909327, 0.0170700967010462, -0.003059701736432441, 0.01025255199117292, -0.017066986408427187, 0.007906972270842472, 0.0063189283993778655, 0.017428188624084573, 0.011360568335342602, -0.006131792103821698, -0.015464511353967758, -0.021870144249125145, 0.006145003330399107, -0.0007159212025063533, -0.0005002999158939536, 0.022752696871737088, -0.0004974578814966324, 0.02144068554980927, 0.0075532767738693, 0.00021662013750672893, 0.013822593464914101, -0.018023988721246775, 0.0009451114000801897, -0.012009963208035157, -0.019075548867921732, -0.0023252558735790407]
   #04(frozen, TZVP)
   g_ri = [-0.09811819558485624, 0.055957944876023014, -0.03761122209975465, 0.0822374794731413, -0.03977176482127387, 0.03399606165936664, 0.0044437064904145895, -0.015335164145310554, 0.00888817814981491, 0.011437009611656945, -0.0008510159094812231, -0.005273017709743089]
   #20(frozen, TZVP)
   #g_ri = [7.002944664648251e-15, -1.6215770303330773e-14, 1.69608332551232e-14, -0.002084013450206257, 0.0020840134502133623, -0.0020840134502009278, 0.002084013450192046, -0.0020840134501982632, -0.0020840134502053687, -0.002084013450202704, -0.0020840134502053687, 0.002084013450192934, 0.002084013450211586, 0.0020840134501884933, 0.0020840134501938223, 0.006045124869100915, 0.0018233702469894197, 0.006045124869095142, -0.0018233702469920843, -0.006045124869098251, 0.006045124869106244, 0.006045124869092922, -0.0060451248690978066, -0.0018233702469920843, -0.006045124869105356, -0.0018233702469881985, 0.006045124869108465, 0.0018233702469917512, 0.006045124869112017, 0.0060451248691022474, -0.006045124869094032, 0.006045124869100249, -0.001823370246995637, 0.006045124869110241, -0.001823370246990308, -0.006045124869094698, 0.006045124869104912, 0.006045124869100915, 0.0018233702469968582, -0.0018233702469987456, 0.006045124869101803, -0.0060451248691022474, 0.0018233702469870883, -0.0060451248691024695, -0.006045124869110463, -0.006045124869102025, 0.0018233702469923063, -0.006045124869108687, -0.006045124869104246, -0.006045124869110907, 0.001823370247000966]
   #loc_df
   #g_ri = [-0.09816245011648483, 0.05597144472967219, -0.03756604398492591, 0.08228680383759901, -0.039801402148018195, 0.03392676809583062, 0.004430240959585996, -0.015331444117448356, 0.008901024336527234, 0.011445405312682011, -0.0008385984657741652, -0.005261748447590753]
   #porphycene
   #g_ri = [0.012755490419931625, 0.00661754412062221, 0.00991578790698977, -0.012735150475903456, 0.006765184332667928, 0.009927915363702411, 0.04101010516239574, 0.02516048691051065, -0.0033512805229216625, -0.041046414909167245, 0.02513842973846625, -0.003391841327865741, 0.014947464167044844, -0.022959111040798064, -0.02056225459458985, -0.014982552356565204, -0.02302645297030992, -0.020530367260476523, 0.01262558797966662, -0.025233301177690848, -0.025148507633472594, -0.01261466926119681, -0.025404201329776388, -0.025034965394651387, -0.032843324754602055, 0.017520715549025567, 0.012502393006858181, 0.03300675890755267, 0.01739935932541492, 0.012562018246990103, 0.009071565830247152, 0.035076392810989176, 0.0104085951914068, -0.009041767890758834, 0.03528966301234604, 0.010292628173278029, -0.008152768298977264, 0.03762529982095586, 0.002738173972848923, 0.008169324178558313, 0.037573160162677866, 0.002684801370168133, -0.06229760357284783, 0.004944854889527228, 0.006127583332521169, 0.06229414345022746, 0.004948602172524907, 0.0061922102938440915, -0.08711051284913829, 0.01361227461031822, -0.011569132728939835, 0.08714749219848095, 0.013474432370268019, -0.011529755775461004, 0.029150774138782864, -0.05060841342676703, 0.0034483185701417174, -0.029208244998502053, -0.05043460211062456, 0.0034872783110357908, 0.02863155506178261, -0.049664126693212296, -0.0007125683010922867, -0.02872094307248707, -0.04950005348663744, -0.0007479636056700878, -0.0035286286687359336, 0.02583183882408946, 0.002182318767470104, 0.0036962775503130985, 0.025789967388794466, 0.0022165405080980527, -0.0010494527543377652, -0.010789141252438794, 0.009326808728363112, 0.0009808899549509364, -0.010961396242048282, 0.009366179136948594, 0.001978080089263947, 0.0048079963097675815, 0.002530458304482952, -0.00198732210197243, 0.004908508942121692, 0.0025451332546102545, -0.005273865863265836, 0.0025412685856045503, 0.0019291122516745807, 0.005263648308393876, 0.002527263847502015, 0.0018794035907910844, -0.005230234803302647, 0.004410687802790303, 0.003255613106763422, 0.005159680079077855, 0.004398057955219192, 0.003250173681841273, -0.0023868215119868808, -0.003356888411532921, -0.0015644387564915374, 0.00241527594861779, -0.0033820835045996755, -0.0015825237968348782, 0.005280632820660332, -0.010436758380798494, 0.0003625267441399599, -0.005311261423281727, -0.010513416069996673, 0.00034731995244688396, -0.0098921923851778, -0.005055282707870479, -0.0018808332521689786, 0.009828985710852223, -0.005036760674874241, -0.001872858815883116]
   #Eigen
   #g_ri = [0.010330632295223352, 0.022567756290751362, -0.013112731582837489, -0.006928970659131561, -0.015138747732223123, 0.014284092939734983, -0.0006697702033168218, -0.0009920599565843569, -0.000578924567190775, 0.00839331789403186, -0.014734476630200666, 0.010551714417214642, -0.001886848942773911, 0.0016723702574101829, 0.003417709389598844, -0.002116265855965249, 0.00925849987096683, -0.013418350835802784, 0.0018263164441497115, 0.0065684220707992735, 0.0014472172283102047, -0.007107428786040981, -0.015607902862558198, -0.0026809797994857276, -0.006085740463904643, 0.006506448422774369, -0.00026981656205554705, -0.004752086544185463, 0.0012463988169426066, 0.003064842648182242, -0.0014811423438936266, -0.0005726428879750112, -0.0033604913065801156, 0.004070593694517788, -0.00015407883693902003, -0.0011238643314603891, 0.006407393471526079, -0.000619986823636065, 0.0017795823627360097]
   
   dev = [abs(abs(g_osv[i])-abs(g_ri[i])) for i in range(len(g_ri))]
   g_rmsd = np.sqrt(np.mean((np.asarray(dev))**2))
   log.info("gradient RMSD: %.4E"%g_rmsd)
   return g_rmsd


def strl2list(strl, dtype='i'):
    if strl is None:
        return None
    else:
        if dtype == "i":
            formater = int
        elif dtype == "f":
            formater = float
        else:
            raise NotImplementedError
        return [formater(i) for i in strl.replace("[","").replace("]","").split(',')]

def num_from_environ(val, dtype):
    if val is None:
        return None
    else:
        if dtype == "i":
            formater = int
        elif dtype == "f":
            formater = float
        else:
            raise NotImplementedError
        return formater(val)

int_storage_dic = {
    "incore": 0,
    "outcore": 1,
    "direct": 2,
    "gpu_incore": 3,
    0: "incore",
    1: "outcore",
    2: "direct",
    3: "gpu_incore",
}

def get_int_storage(int_storage):
    if int_storage in {"0", "1", "2", "3"}:
        return int(int_storage)
    else:
        return int_storage_dic[int_storage.lower()]

method_dic = {
    "rhf": 0,
    "hf": 0,
    "cmbeosvmp2": 1,
    "mbeosvmp2": 2,
    "osvmp2": 3,
    "oriosvmp2": 4,
}

def get_method(method):
    return method_dic[method.replace("-",'').replace("_",'').lower()]
    


class FetchParameters():
    '''
    A class to collect input parameters from run.sh
    '''
    def __init__(self, mol):
        '''self.molecule = get_name(sys.argv[1])
        self.use_ecp = bool(int(os.environ.get("use_ecp", 0)))
        self.cal_mode = os.environ.get('cal_mode', 'energy')
        self.save_pene = bool(int(os.environ.get("save_pene", 0)))
        self.basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()'''

        for key, value in inputs.items():
            setattr(self, key, value)
        
        self.use_gpu = bool(self.ngpu)
        
        if "ml" in self.cal_mode:
            self.ml_test = True
            if "mp2int" in self.cal_mode:
                self.ml_mp2int = True
            else:
                self.ml_mp2int = False
            self.cal_mode = 'energy'
            if os.path.isfile("ml_features.hdf5"):
                '''with h5py.File("ml_features.hdf5", "r") as f:
                    keys = [ki for ki in f.keys()]'''
                #if self.molecule in keys:
                with h5py.File("ml_features.hdf5", "r") as f:
                    if self.molecule in f:
                        if irank == 0:
                            print("Dataset for %s exists"%self.molecule)
                        
                        sys.exit()
        else:
            self.ml_mp2int = False
            self.ml_test = False
        #self.nosv_ml = num_from_environ(os.environ.get("nosv_ml"), dtype="i")

        if "opt" in self.cal_mode:
            self.chkfile_init = self.chkfile_hf = self.chkfile_loc = None
            self.chkfile_save = self.chkfile_ialp_mp2 = self.chkfile_ialp_hf = None
        else:
            '''self.chkfile_init = os.environ.get("chkfile_init", None)
            self.chkfile_hf = os.environ.get("chkfile_hf", None)
            self.chkfile_loc = os.environ.get("chkfile_loc", None)
            self.chkfile_ialp_hf = os.environ.get("chkfile_ialp_hf", None)
            self.chkfile_ialp_mp2 = os.environ.get("chkfile_ialp_mp2", None)
            self.chkfile_fitratio_hf = os.environ.get("chkfile_fitratio_hf", None)
            self.chkfile_fitratio_mp2 = os.environ.get("chkfile_fitratio_mp2", None)
            self.chkfile_ti = os.environ.get("chkfile_ti", None)
            self.chkfile_qcp = os.environ.get("chkfile_qcp", None)
            self.chkfile_qmat = os.environ.get("chkfile_qmat", None)
            self.chkfile_qao = os.environ.get("chkfile_qao", None)
            self.chkfile_imup = os.environ.get("chkfile_imup", None)
            self.chkfile_save = os.environ.get("chkfile_save", None) '''
            if self.chkfile_save is not None:
                os.makedirs(self.chkfile_save, exist_ok=True)
        if self.cal_mode == "energy":
            self.cal_grad = False
        else:
            self.cal_grad = True

        if self.cal_mode == "chk":
            self.get_chk = True
        else:
            self.get_chk = False

        '''if irank == 0:
            self.verbose = int(os.environ.get("verbose", 4))
        else:
            self.verbose = 0
        self.max_memory = int(os.environ.get("max_memory", -1))'''

        if irank != 0: self.verbose = 0

        if self.max_memory == -1:
            self.max_memory = psutil.virtual_memory().total*0.9*1e-6

        mol.max_memory = self.max_memory
        mol.name = self.molecule

        #self.qm_atoms = strl2list(os.environ.get("qm_atoms", None))
        if self.qm_atoms is None: #Non-qmmm case
            self.mol_total = self.mol = mol
            #self.mol_mm = None
        else: #QM-MM
            #Backup the total mol
            self.mol_total = mol.copy()
            '''self.qm_region = strl2list(os.environ.get("qm_region", None))
            self.nonwater_region = strl2list(os.environ.get("nonwater_region", None))
            #self.qm_atoms = strl2list(os.environ.get("qm_atoms", None))
            self.qm_center = int(os.environ.get("qm_center", 2))
            self.cg_residue = os.environ.get("cg_residue", "CG1")
            self.nwater_qm = int(os.environ.get("nwater_qm", 20))'''
            
            #Build MM mol
            #Bohr to Angstrom
            self.coords_non_ghost = []
            self.index_non_ghost = []
            self.index_ghost = []
            self.coords_all = []
            for ia, (ia_sym, co_i) in enumerate(mol._atom):
                ico = np.asarray(co_i)*lib.param.BOHR
                self.coords_all.append((ia_sym, ico))
                is_ghost = False
                for isym in [":", "-", "_"]:
                    if isym in ia_sym:
                        is_ghost = True
                if is_ghost:
                    self.index_ghost.append(ia)
                else:
                    self.index_non_ghost.append(ia)
                    self.coords_non_ghost.append([ia_sym, ico])
            

            self.qm_region_full, self.mm_region, self.qm_coords, self.mm_coords = \
                             qmmm.region_qmmm(self.coords_all, qm_region=self.qm_region,
                                              qm_atoms=self.qm_atoms, qm_center=self.qm_center, 
                                              nwater_qm=self.nwater_qm)
            self.qm_region = [ia for ia in self.qm_region_full if ia not in self.index_ghost]
            


            #Build QM mol
            self.mol = gto.M()
            self.mol.atom = self.qm_coords
            self.mol.unit = 'Angstrom'
            self.mol.basis = self.basis
            #self.mol.charge = int(os.environ.get("charge", 0))
            #self.mol.spin = int(os.environ.get("spin", 0))
            self.mol.charge = self.charge
            self.mol.spin = self.spin
            self.mol.build(verbose=0)
            self.mol.name = self.mol_total.name
            self.mol.max_memory = self.mol_total.max_memory
            self.mol.verbose = self.verbose
            #print(len(self.mol._atom));sys.exit()
        
        '''self.loc_fit = bool(float(os.environ.get("loc_fit", 0)))
        self.use_gpu = bool(int(os.environ.get("ngpu", 0)))
        self.double_buffer = bool(int(os.environ.get("double_buffer", 0)))'''
        if self.double_buffer:
            assert self.use_gpu
            assert nrank % 2 == 0 and nrank_per_gpu % 2 == 0
        if self.use_gpu:
            
            #self.gpu_memory = float(os.environ.get("gpu_memory", 1100)) #MB
            mempool = cupy.get_default_memory_pool()
            mempool.set_limit(size=self.gpu_memory * 1e6)
            
            #self.ngpu_per_node = min(cupy.cuda.runtime.getDeviceCount(), nrank_shm)

            if nrank_shm % ngpu_shm != 0:
                raise NotImplementedError(f"Number of MPI processes ({nrank_shm}) is not multiples of GPUs ({ngpu_shm}).")

            '''self.gpu_dtype = np.float64
            if irank_shm < self.ngpu_shm:
                self.gpu_id = irank_shm
                cupy.cuda.runtime.setDevice(self.gpu_id)
            else:
                self.gpu_id = None
                #self.use_gpu = False'''

        #self.outcore = bool(int(os.environ.get("outcore", 0)))
        '''self.int_storage = get_int_storage(os.environ.get("int_storage", "incore"))
        self.shell_tol = float(os.environ.get("shell_tol", 1e-10))
        self.fit_tol = float(os.environ.get("fit_tol", 1e-6))
        self.bfit_tol = float(os.environ.get("bfit_tol", 1e-2))
        self.max_cycle = int(os.environ.get("max_cycle", 30))
        self.local_type = int(os.environ.get("local_type", 1))
        self.pop_method = os.environ.get("pop_method", "low_melow")
        self.pop_hf = bool(int(os.environ.get("pop_hf", 0)))
        self.pop_uremp2 = bool(int(os.environ.get("pop_urmp2", 0)))
        self.pop_remp2 = bool(int(os.environ.get("pop_remp2", 1)))
        self.charge_method = os.environ.get("charge_method", "meta_lowdin")
        self.charge_method_mp2 = os.environ.get("charge_method_mp2", self.charge_method)
        self.use_cposv = bool(int(os.environ.get("use_cposv", 1))) 
        self.use_cpl = bool(int(os.environ.get("use_cpl", 0))) 
        self.cposv_tol = float(os.environ.get("cposv_tol", 1e-10))
        self.osv_tol = float(os.environ.get("osv_tol", 1e-4))
        self.nosv_id = num_from_environ(os.environ.get("nosv_id"), "i")
        self.svd_method = int(os.environ.get("svd_method", 1)) #0: exact, 1: rsvd, 2: idsvd
        self.threeb_tol = float(os.environ.get("threeb_tol", '0.2'))
        self.remo_tol = float(os.environ.get("remo_tol", 1e-2))
        self.disc_tol = float(os.environ.get("disc_tol", 1e-7))
        self.loc_tol = float(os.environ.get("loc_tol", 1e-6))
        self.use_frozen = bool(float(os.environ.get("use_frozen", 1)))
        self.use_sl = bool(float(os.environ.get("use_sl", 1)))
        self.method = get_method(os.environ.get("method", "cmbeosvmp2"))
        self.basis_molpro = bool(int(os.environ.get("basis_molpro", 0)))
        self.solvent = os.environ.get("solvent", None) #Only PTE scheme is implemented
        self.use_df_hf = bool(int(os.environ.get("use_df_hf", 1)))'''
        if not self.use_df_hf:
            if (not self.use_gpu) or self.int_storage != 2:
                raise NotImplementedError("Non-DF HF is implemented only with the integral-direct scheme on GPUs")
                self.use_df_hf = True
        #self.fully_direct = int(os.environ.get("fully_direct", 1))
        assert self.fully_direct in {0, 1, 2}

        self.method = get_method(self.method)

        if self.int_storage != 2:
            self.fully_direct = 0
        try:
            self.solvent = float(self.solvent)
        except (ValueError, TypeError) as e:
            pass
        
        self.sol_eps = get_eps(self.solvent)
        
    def get_auxbasis(self):
        if self.mol.pbc: # in the PBC calculations, hf and mp2 share the same ao integrals
            self.with_df = pbc_df.GDF(self.mol)
            self.with_df.build(with_j3c=False)
            self.with_df.df_builder = _RSGDFBuilder(self.with_df.cell, self.with_df.auxcell, self.with_df.kpts)
            self.with_df.df_builder.mesh = self.with_df.mesh
            self.with_df.df_builder.linear_dep_threshold = self.with_df.linear_dep_threshold
            self.with_df.df_builder.build()
            self.with_df.auxmol = self.with_df.auxcell
            self.with_df.df_builder.use_gpu = self.with_df.use_gpu = self.use_gpu
            self.auxmol_hf = self.auxmol_mp2 = self.with_df.auxcell
            self.auxbasis_hf = self.auxbasis_mp2 = self.with_df.auxcell.basis
            self.naux_hf = self.naux_mp2 = self.with_df.auxcell.nao_nr()
        else:
            atom_list = [ico[0] for ico in self.mol._atom]
            '''if self.basis_molpro:
                self.auxbasis_hf = get_aux_molpro(self.mol, self.basis, mp2fit=False)
                self.auxbasis_mp2 = get_aux_molpro(self.mol, self.basis, mp2fit=True)
            if self.use_ecp or "Be" in atom_list:
                self.auxbasis_hf = get_aux(self.mol, mp2fit=False)
                self.auxbasis_mp2 = get_aux(self.mol, mp2fit=True)
            elif self.mol.basis == '631+g**':
                self.auxbasis_mp2 = self.auxbasis_hf = 'heavy-aug-cc-pvdz-jkfit'
            else:
                self.auxbasis_hf = addons.make_auxbasis(self.mol)
                self.auxbasis_mp2 = addons.make_auxbasis(self.mol, mp2fit=True)
            
            self.auxbasis_hf = os.environ.get("auxbasis_hf", self.auxbasis_hf)
            self.auxbasis_mp2 = os.environ.get("auxbasis_mp2", self.auxbasis_mp2)'''

            if self.auxbasis_hf is None:
                if self.basis_molpro:
                    self.auxbasis_hf = get_aux_molpro(self.mol, self.basis, mp2fit=False)
                elif self.use_ecp or "Be" in atom_list:
                    self.auxbasis_hf = get_aux(self.mol, mp2fit=False)
                elif self.mol.basis == '631+g**':
                    self.auxbasis_hf = 'heavy-aug-cc-pvdz-jkfit'
                else:
                    self.auxbasis_hf = addons.make_auxbasis(self.mol)
            
            if self.auxbasis_mp2 is None:
                if self.basis_molpro:
                    self.auxbasis_mp2 = get_aux_molpro(self.mol, self.basis, mp2fit=True)
                elif self.use_ecp or "Be" in atom_list:
                    self.auxbasis_mp2 = get_aux(self.mol, mp2fit=True)
                elif self.mol.basis == '631+g**':
                    self.auxbasis_mp2 = 'heavy-aug-cc-pvdz-jkfit'
                else:
                    self.auxbasis_mp2 = addons.make_auxbasis(self.mol, mp2fit=True)
            
            self.auxmol_hf = addons.make_auxmol(self.mol, self.auxbasis_hf)
            self.auxmol_mp2 = addons.make_auxmol(self.mol, self.auxbasis_mp2)
            self.naux_hf =  self.auxmol_hf.nao_nr()
            self.naux_mp2 = self.auxmol_mp2.nao_nr()

    def get_mpi_para(self):
        self.nrank = comm.Get_size()
        self.nrank_shm = comm_shm.size
        self.nnode = self.nrank//self.nrank_shm
        self.rank_list = range(nrank)
        self.shm_ranklist = range(self.nrank_shm)
        self.rank_slice = [range(i, i+nrank//nnode) for i in range(0, nrank, nrank//nnode)]
        win_pid, pid_list = get_shared(comm_shm.size, dtype='i')
        pid_list[irank_shm] = os.getpid()
        comm_shm.Barrier()
        self.pid_list = np.copy(pid_list)
        comm_shm.Barrier()
        free_win(win_pid)
        self.mol.pid_list = self.pid_list

    

    def extra_para(self):
        self.get_auxbasis()
        self.get_mpi_para()
        if not self.mol.pbc:
            get_c_mol(self.mol, self.auxmol_hf)
            get_c_mol(self.mol, self.auxmol_mp2)

def gradient(mol):
    def clear_tmp(dir_now="."):
        file_list = os.listdir(dir_now)
        for fname in file_list:
            if "tmp" not in fname:
                continue
            full_path = os.path.join(dir_now, fname)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)  # Delete files/symlinks
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
    def read_para(mol):
        mf = FetchParameters(mol)
        mf.extra_para()
        return mf
    def print_info(mf, log):
        msg = 'Basic settings:\n'
        int_scheme = int_storage_dic[mf.int_storage]
        if int_scheme == "direct":
            if mf.fully_direct == 0:
                int_scheme = "direct with cached ialp"
            elif mf.fully_direct == 1:
                int_scheme = "fully direct"
            elif mf.fully_direct == 1:
                int_scheme = "fully direct with host memory"
        msg += f'Integral generation: {int_scheme}\n'
        if mf.loc_fit:
            msg += 'Local density fitting: %.1E/%.1E(block)\n'%(mf.fit_tol, mf.bfit_tol)
        else:
            msg += 'Local density fitting: OFF\n'
        msg += "Molecue to be computed: %s\n"%mol.name
        msg += "Number of CPU nodes: %d, cores: %d, threads/core: %s\n"%(nnode, nrank, os.environ.get("OMP_NUM_THREADS", 1))
        msg += "Maximum memory per node: %.2f MB (%.2f MB used)\n"%(mf.mol.max_memory, psutil.virtual_memory()[3]*1e-6)#lib.current_memory()[0])
        if mf.use_gpu:
            msg += "Number of GPUs per node: %d\n"%ngpu_shm
            msg += "Maximum memory per GPU: %.2f MB\n"%mf.gpu_memory
        msg += "Basis set: %s\n"%mf.mol.basis
        msg += "Auxilary basis for HF: %s\n"%mf.auxmol_hf.basis
        msg += "Auxilary basis for MP2: %s\n"%mf.auxmol_mp2.basis
        if mf.mol.ecp is not None:
            msg += "ecp: %s\n"%mf.mol.ecp
        nao = mf.mol.nao_nr()
        nocc = mf.mol.nelectron // 2
        nvir = nao - nocc
        msg += "Number of AOs: %d\n"%nao
        msg += "Number of occupied MOs: %d\n"%nocc
        msg += "Number of unoccupied MOs: %d\n"%nvir
        msg += "Number of auxilary basis for RHF: %d\n"%mf.naux_hf
        msg += "Number of auxilary basis for MP2: %d\n"%mf.naux_mp2
        if mf.solvent is not None:
            msg += "Cosmo solvation model with solvent "
            if type(mf.solvent) == str:
                msg += "%s (dielectric constant = %.4f)\n"%(mf.solvent, mf.sol_eps)
            else:
                msg += "dielectric constant %.4f\n"%mf.sol_eps
        if hasattr(mol, 'md_step'):
        #if mf.cal_mode == 'md':
            if mol.md_step == 1:
                with open("sim_parameters.out", 'w') as f:
                    f.write(msg)
                with open("traj_qm.xyz", 'w') as f:
                    pass
        log.info(msg)
    def info_time(mf, hfe, g, e_mm=None, t_tot=None, log=None):
        msg = '\n' + "-"*43 + '\n'
        msg += "Electronic energy (Eh):"
        log.info(msg)
        msg_list = []
        etot = hfe
        msg_list.append(['RHF energy :', '%.10f'%hfe])

        cal_mp2 = mf.method in {1, 2, 3, 4}
        
        if cal_mp2:
            etot += mf.ene_mp2
            msg_list += [['MP2 correlation energy:', '%.10f'%mf.ene_mp2]]
        if e_mm != 0:
            etot += e_mm
            msg_list.append(['MM energy :', '%.10f'%e_mm])
        msg_list += [['', '-'*len('%.10f'%hfe)],  ['Total energy:', '%.10f'%(etot)]]
        if irank == 0:
            with open("etot_record.log", "a") as fetot:
                fetot.write("%s    %15.8f\n"%(mf.mol.name, etot))
        print_align(msg_list, align='lr', indent=0, log=log)
        
        if mf.cal_grad: 
            #print gradient
            msg = "-"*50
            msg += "Energy gradient (Eh/Bohr):"
            log.info(msg)
            msg_list = [["Atom", "X", "Y", "Z"]]
            for ia, (gx, gy, gz) in enumerate(g.reshape(-1, 3)):
                atm = mf.mol_total.atom_symbol(ia)
                msg_list.append([atm, '%.8f'%gx, '%.8f'%gy, '%.8f'%gz])
            print_align(msg_list, align='crrr', align_1='cccc', indent=0, log=log)
        log.info(('-'*76))
        log.info("Summary of timings:")
        time_list = [["RHF energy", mf.t_hf]]
        if cal_mp2:
            time_list += [["localization", mf.t_loc], ["MP2 feri", mf.t_feri_mp2],
                          ["OSV-based matrices", mf.t_osv_gen], ["Residual iterations", mf.t_res]]
        if mf.cal_grad: 
            if cal_mp2:
                t_grad_mp2 = mf.t_loc + mf.t_feri_mp2 + mf.t_osv_gen + mf.t_res + \
                            mf.t_dr + mf.t_dk_yi + mf.t_dferi_mp2 + mf.t_zvec
                time_list += [["OSV DM and dR", mf.t_dr], ["dK and Yi", mf.t_dk_yi], ["derivative feri", mf.t_dferi_mp2],
                            ["Z vector", mf.t_zvec], ["MP2 energy and gradient", t_grad_mp2]]
            time_list.append(["RHF energy and gradient", mf.t_grad_hf+mf.t_hf])
        elif cal_mp2:
            t_ene_mp2 = mf.t_loc + mf.t_feri_mp2 + mf.t_osv_gen + mf.t_res
            time_list.append(["MP2 energy", t_ene_mp2])
        if cal_mp2:
            time_list.append(["RHF + MP2", t_tot])
        
        time_list = get_max_rank_time_list(time_list)
        print_time(time_list, log, left_align=True)
        log.info(('-'*76))

        

    def kernel():
        def get_gtot(qm_region, mm_region, gqm, gmm, gmm_qm):
            gtot = np.copy(gmm).reshape(-1, 3)
            gtot[qm_region] += gqm.reshape(-1,3)
            gtot[mm_region] += gmm_qm.reshape(-1,3)
            return gtot
        
        if irank == 0:
            clear_tmp()
        comm.Barrier()
        tt = get_current_time()
        mf = read_para(mol)
        log = lib.logger.Logger(sys.stdout, mf.verbose)
        if irank == 0:
            print_info(mf, log)
        
        if mf.qm_atoms is not None:
            win_emm, emm_node = get_shared(1)
            win_fmm, fmm_node = get_shared((mol.natm*3))
            win_charge, charge_node = get_shared(len(mf.mm_region))
            
            if irank_shm == 0:
                GRAD_MM = qmmm.mm_gradient(mf.coords_non_ghost, qm_region=mf.qm_region,
                                           nonwater_region=mf.nonwater_region, 
                                           cg_residue=mf.cg_residue, log=log)
                #print_time(["Initialization", get_elapsed_time(tt)], log)
                emm_node[0], fmm, charge_node[:] = GRAD_MM.kernel()
                fmm_node.reshape(-1, 3)[mf.index_non_ghost] = fmm.reshape(-1, 3)

            comm_shm.Barrier()
            mf.mol_mm = pyscf.qmmm.mm_mole.create_mm_mol(mf.mm_coords, charges=np.copy(charge_node))
            f_mm = np.copy(fmm_node)
            e_mm = emm_node[0]
            g_mm = -f_mm.reshape(-1, 3) #gradient = - force
            comm_shm.Barrier()
            for win_i in [win_emm, win_fmm, win_charge]:
                free_win(win_i)
        else:
            mf.mol_mm = None
            e_mm, g_mm = 0.0, None
        #Computation of HF enengy
        if mol.pbc:
            hf = pbc_scf.RHF(mol).density_fit()
        else:
            hf = scf.RHF(mol).density_fit()
        hfe = scf_parallel(hf, mf)

        if mf.method == 0: #HF
            e_qm = hfe
            if mf.cal_grad:
                hf.verbose = mf.verbose
                g = hf_grad.kernel(hf).reshape(-1, 3)
                if mf.mol_mm is not None:
                    g = get_gtot(mf.qm_region_full, mf.mm_region, g, g_mm, hf.mol_mm.grad)
            else:
                g = None
            t_tot = get_elapsed_time(tt)
            info_time(hf, hfe, g, e_mm=e_mm, t_tot=t_tot, log=log)
        elif mf.method in {1, 2, 3, 4}:
            mp2 = mp2_ene.OSVLMP2(hf, mf)
            mp2e = mp2.kernel()
            e_qm = hfe + mp2e
            if mf.cal_grad:
                g = compute_gradient_mbe(mp2, log=log).reshape(-1, 3)
                if mf.mol_mm is not None:
                    g = get_gtot(mf.qm_region_full, mf.mm_region, g, g_mm, hf.mol_mm.grad)
            else:
                g = None
            t_tot = get_elapsed_time(tt)
            info_time(mp2, hfe, g, e_mm=e_mm, t_tot=t_tot, log=log)

        e = e_qm + e_mm
        if mf.cal_grad:
            if mf.shared_disk:
                if irank == 0:
                    clear_tmp()
            free_all_win()

            '''if g_mm is not None:                
                mf.mol = mf.mol_total'''
            if irank == 0:
                if ("opt" in mf.cal_mode):
                    def save_eg(mol, ene, grad):
                        g_list = []
                        for ia in range(mol.natm):
                            gx, gy, gz = grad[ia]
                            g_list.append([mol.atom[ia][0], "%.8f"%gx, "%.8f"%gy, "%.8f"%gz])
                        g_msg = print_align(g_list, align='lrrr', printout=False)
                        with open("opt_eg.xyz", 'a') as f:
                            f.write("Step %d\nEnergy(Eh): %.8f\nGradient(Eh/Bohr):\n%s\n"%(mol.opt_cycle, ene, g_msg))
                    save_eg(mf.mol_total, e, g)
                    with open("opt_traj.xyz", 'a') as f:
                        f.write(get_coords_from_mol(mf.mol_total, info="Step %d     Energy: %.9f"%(mol.opt_cycle, e)))
                if hasattr(mol, 'md_step') and g_mm is not None:
                    ranges = []
                    for key, group in itertools.groupby(enumerate(mf.qm_region), lambda i: i[0] - i[1]):
                        group = list(map(itemgetter(1), group))
                        ranges.append((group[0], group[-1]))
                    with open("traj_qm.xyz", 'a') as f:
                        f.write(get_coords_from_mol(mf.mol, info="Step %d Energy: %.9f QM region: %s"%(mol.md_step, e_qm, ranges)))
        else:
            g = None
            MPI.Finalize()
            sys.exit()
        return e, g
    return kernel()
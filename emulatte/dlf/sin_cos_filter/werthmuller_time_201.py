# Copyright 2021 Waseda Geophysics Laboratory
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

base = np.array([
 0.00041894212344838373,  0.00044633335854972402,  0.00047551548484672926,
 0.00050660559422163276,  0.00053972843424726375,  0.00057501690873069998,
 0.00061261261098342196,  0.00065266639195768169,  0.00069533896552871567,
 0.00074080155335146157,  0.00078923657187923680,  0.00084083836430100917,
 0.00089581398033412136,  0.00095438400700135508,  0.00101678345372577619,
 0.00108326269529477768,  0.00115408847647690454,  0.00122954498232244599,
 0.00130993497844232023,  0.00139558102584056635,  0.00148682677517490941,
 0.00158403834563853522,  0.00168760579399579989,  0.00179794467966628863,
 0.00191549773213705774,  0.00204073662739346837,  0.00217416388049645418,
 0.00231631486190011365,  0.00246775994559996631,  0.00262910679773126687,
 0.00280100281480022547,  0.00298413772133144118,  0.00317924633735448044,
 0.00338711152683399954,  0.00360856733887387083,  0.00384450235429919423,
 0.00409586325104425476,  0.00436365860265232851,  0.00464896292512865811,
 0.00495292098838438864,  0.00527675240957093446,  0.00562175654673531099,
 0.00598931771243197460,  0.00638091072821058130,  0.00679810684226669123,
 0.00724258003399981792,  0.00771611373077548800,  0.00822060796384200751,
 0.00875808699211472544,  0.00933070742441784867,  0.00994076687277394011,
 0.01059071317146193479,  0.01128315419883481677,  0.01202086834130642462,
 0.01280681564149370281,  0.01364414967524573384,  0.01453623020521557785,
 0.01548663666174684810,  0.01649918250516621207,  0.01757793052711013188,
 0.01872720915228142172,  0.01995162980604581540,  0.02125610541755505387,
 0.02264587013263949511,  0.02412650031556731398,  0.02570393692393855775,
 0.02738450934649269003,  0.02917496079947733828,  0.03108247538348008279,
 0.03311470690928752791,  0.03527980960843404845,  0.03758647085166469393,
 0.04004394600659374331,  0.04266209557442330463,  0.04545142475473155919,
 0.04842312559708220054,  0.05158912190858661040,  0.05496211709760797509,
 0.05855564514557817790,  0.06238412491144922772,  0.06646291798667318396,
 0.07080839033285123152,  0.07543797794936972501,  0.08037025683451147484,
 0.08562501752075796224,  0.09122334448335193968,  0.09718770074074291332,
 0.10354201798637166498,  0.11031179261344217779,  0.11752418801797677450,
 0.12520814359063983656,  0.13339449083465509949,  0.14211607707573359338,
 0.15140789726039260987,  0.16130723437150060184,  0.17185380902445776541,
 0.18308993884426019716,  0.19506070826294111265,  0.20781414941869166824,
 0.22140143488251246007,  0.23587708298570000953,  0.25129917657203681980,
 0.26772959605241636316,  0.28523426769702386885,  0.30388342816133417923,
 0.32375190630732292307,  0.34491942345068687947,  0.36747091323879960134,
 0.39149686244289844117,  0.41709367403191427437,  0.44436405398476119810,
 0.47341742339315123766,  0.50437035750847702964,  0.53734705349441758138,
 0.57247982876210079528,  0.60990965188737000791,  0.64978670824043416321,
 0.69227100259746254807,  0.73753300115207653942,  0.78575431550277474724,
 0.83712843136076386941,  0.89186148490209371520,  0.95017308987917958873,
 1.01229721881044953946,  1.07848314178385562556,  1.14899642664114520763,
 1.22412000455609071281,  1.30415530528225609430,  1.38942346662542437485,
 1.48026662299364586417,  1.57704927819513796017,  1.68015976799233834882,
 1.79001181828052446932,  1.90704620514312983559,  2.03173252344463550756,
 2.16457107105744928077,  2.30609485628311761118,  2.45687173652257895284,
 2.61750669677674618896,  2.78864427711981699431,  2.97097115888540086459,
 3.16521891994244164437,  3.37216697011633570469,  3.59264567853346905579,
 3.82753970543751886879,  4.07779155184625086150,  4.34440534129165989441,
 4.62845084881747403927,  4.93106779340021805780,  5.25347041101695388932,
 5.59695232670889453175,  5.96289174518984665951,  6.35275698082652695575,
 6.76811234917954784862,  7.21062444374456745777,  7.68206882307877769733,
 8.18433713514444072246,  8.71944470745565958225,  9.28953863348332298244,
 9.89690638776472120242, 10.54398500428541041174, 11.23337085496120657524,
11.96783006745597610632, 12.75030962413621793416, 13.58394918669652717824,
14.47209369390148836487, 15.41830678299210966031, 16.42638508860932944344,
17.50037347660841291486, 18.64458127388926911294, 19.86359955936389454223,
21.16231958544036473313, 22.54595240393853572414, 24.02004977518555861593,
25.59052644418794386638, 27.26368387326219888678, 29.04623552635027650126,
30.94533380647150977438, 32.96859875439682241449, 35.12414862369684698251,
37.42063245484502687077, 39.86726477907773613651, 42.47386259125889296229,
45.25088474010121331048, 48.20947389379483638550, 51.36150124942922445825,
54.71961416560207425164, 58.29728690933919921235, 62.10887472094472627759,
66.16967141371431182506, 70.49597073962775084510, 75.10513176724701622788,
80.01564853414659239661, 85.24722425335225750587, 90.82085037153834150558,
96.75889079620085908573,103.08517162976365000304,109.82507677067211204758,
117.00564976506775849430,124.65570231772045417529,132.80592989761206013100
])


sin = np.array([
-10.5854278667137382541341,  6.0117434517566659124554, 30.7082118602062728029978,
  0.9224790035930224840754,-84.3615720108925870590610, 88.8428357938806101401497,
-40.8188456303967797111909,-12.8869366188287219898712, 13.6749201778017006603250,
 57.4760733364941245326918,-54.2349847019922606250475, -6.2305187043144130143446,
 -5.9509651707401456377511, 32.0844562242976039101450,-14.7168713811895859322476,
 -3.4163383937442688420560,  4.2516056281902807612028,  3.5754092974691746853466,
 -4.2764745305055873458855, -4.7215787437425840167293,  1.3869979387800974723888,
 12.4388107217449928043607,-12.6248960553096196690603,  1.3955973144233049243468,
  3.3808366117204418088704, -1.5722060018254477853361,  1.7522399735463016767767,
 -3.1677367774064073557838,  1.3431768203571168296406,  2.4014379768246598700898,
 -3.7238538948594706035067,  1.3825370068860765027097,  2.3707359410624571083304,
 -4.9765313063374883029155,  5.5449548570156945714871, -4.6109964146746573732116,
  3.1049578886204440841823, -1.6898948379286762300211,  0.6362841589000829412015,
  0.0426987685294207622300, -0.4347365467243858860868,  0.6368230354563706452353,
 -0.7224826131661619132984,  0.7394864563227565579240, -0.7169451781639669674107,
  0.6726142831266894717146, -0.6176286708977171135970,  0.5593059097508875643356,
 -0.5024540699856938630319,  0.4502874089397652657141, -0.4047754538104190169534,
  0.3671613115280718542266, -0.3380205285617136623522,  0.3176297781124143293496,
 -0.3057816119259073039949,  0.3020389980482354186364, -0.3053441609083717978024,
  0.3142980886112791139020, -0.3267373735781202714890,  0.3402924254351268329444,
 -0.3520946252143826171732,  0.3597019585892426984941, -0.3607855810890421843773,
  0.3541927146746004506639, -0.3392379786805311270292,  0.3166736607814501502922,
 -0.2873974274459754063571,  0.2534475603967050183485, -0.2162670184446815468959,
  0.1780946341788273712403, -0.1399599618236394282800,  0.1037417000579121673098,
 -0.0698007699076020615703,  0.0397434070416963006189, -0.0133708141460441558984,
 -0.0078357131470088541014,  0.0247260423377510668574, -0.0360220042274702681451,
  0.0434415445036169106663, -0.0458613927897562334568,  0.0458260233168823263861,
 -0.0419851825907811629368,  0.0375007624708102380029, -0.0303776796625639493310,
  0.0243782535257392324324, -0.0166342211982785162516,  0.0117308099255555211421,
 -0.0057381237952221671172,  0.0043095720206511008502, -0.0020739394653950895576,
  0.0059176970960035059008, -0.0084576203647986347711,  0.0180594885777317394615,
 -0.0247444955334739716268,  0.0388962047179583839118, -0.0474221293069358867944,
  0.0636009793297599712592, -0.0706716935299713383545,  0.0859645892005550249504,
 -0.0883054176089236864122,  0.1004782179839346784034, -0.0958262050788941111001,
  0.1042253091548364235530, -0.0919681624529280622449,  0.0978666481330021514884,
 -0.0788970410693913409839,  0.0849190134250027706120, -0.0605776251457353018259,
  0.0694756539315366100640, -0.0402938194565015186943,  0.0540785127310986976634,
 -0.0193008297290368502352,  0.0393482804312092343535,  0.0026936404805013493963,
  0.0249849758213136902141,  0.0264145157421220716509,  0.0106940014662946832752,
  0.0523924404836165411026, -0.0034582279250759996592,  0.0807215904861955102501,
 -0.0168715254005838422025,  0.1108709278811071813342, -0.0283946013620199055882,
  0.1417823844712950698277, -0.0367826872084439510102,  0.1723434566015649982429,
 -0.0413674394857953889315,  0.2015930806009429698822, -0.0422243218207893691862,
  0.2281228806324831215324, -0.0398127635631776721770,  0.2491558680928359637008,
 -0.0349681470077667783114,  0.2602532515173308635958, -0.0298709191072443168424,
  0.2558726370812025030510, -0.0293867851417255986080,  0.2300459202413440706891,
 -0.0412692920947139926802,  0.1767261801988514180461, -0.0742272132303283677102,
  0.0911933942928719321053, -0.1339937388071893997754, -0.0249142309084846372280,
 -0.2171490769319003311111, -0.1543110637898739423424, -0.3016747752296284335038,
 -0.2542534890248850931727, -0.3374749720204660574652, -0.2559324215943050706557,
 -0.2522003999598131929183, -0.0916548751698880587169,  0.0002451393700228310184,
  0.2283221119647683150511,  0.3362578192882395566876,  0.5084884452971621948691,
  0.4621192616960526078351,  0.3840452228983644022975,  0.0568684192434416313500,
 -0.2677832927682805563840, -0.6112250685837133934442, -0.6897421427754512057717,
 -0.4181849148408994798487,  0.1263384453832862019951,  0.7677969933286705739306,
  0.8133586540368122896894,  0.3497498419022973537018, -0.7763162464677908491950,
 -0.9667265002837666099111, -0.3800926825624162397332,  1.2352548663725366751009,
  0.7911020096402959778104, -0.5236339553176779793020, -1.6264734992347238407007,
  1.0134569769895656055070,  1.1291318119601305713928, -0.9113589815793819992606,
 -1.3006195500286237276555,  2.5437196295797908085490, -1.7016966234274755187528,
 -0.0776742177440578884795,  1.4192400602165931022824, -1.8684033121929977117048,
  1.6758717454715628125683, -1.2388127812781593029001,  0.8131829006721006392056,
 -0.4933343800117624411428,  0.2829501175469676943486, -0.1551452064443097078605,
  0.0815348128308123892838, -0.0409110626331862237137,  0.0194298797681715443297,
 -0.0086269212400996320850,  0.0035250876049227738637, -0.0013000749017179353672,
  0.0004223264749396243530, -0.0001170415122752015808,  0.0000264673718167320011,
 -0.0000045620673308466905,  0.0000005308801493992684, -0.0000000311711012463033
])

cos = np.array([
-4.72606792172843448440744, 6.23661538086159339400183, 0.26648912256460921543777,
 4.40436092661943057180451,-8.56202960987701189310428, 1.49089154852534444550827,
-2.14204380329299892693484, 4.33356316840370414666950,-2.27115568976068171735960,
 2.66547674454966454504756,-4.54298422856987915707805, 3.28666986211343292723086,
 4.86276552895523206387907,-8.59562831421796325059859, 3.11072493430328478325464,
-0.72766333313570685792371, 1.27863193122447604821446, 0.17068394224154379013036,
-0.00148611758998264120724,-1.27778851941428750471630, 0.82375867736726193779617,
-0.05888951167428058131037, 0.10106273215719745184327,-0.37682296930243519561543,
 0.55071522924744586990897,-0.40828938176791873537752, 0.08326001158722882988794,
-0.11136732438573784809144, 0.46453631944130485686983,-0.54702368240454124492800,
 0.17836914845354909231467, 0.21849970882974559249767,-0.27038673010542418895241,
 0.04291444385932342842072, 0.18214635342350729563243,-0.22859668495287682743466,
 0.11728989196996701671605, 0.04585753517862816569517,-0.17348848423403515073282,
 0.24200524099915338149813,-0.26326458020394005732356, 0.26344377866058815707007,
-0.25439041122812322548441, 0.24147596972327298181504,-0.21679033057219335156596,
 0.17822347031873575495808,-0.12070933087660178084644, 0.05238591854051308921436,
 0.02101937706239160458255,-0.08205899349405729181761, 0.12174901259148468435356,
-0.12430574359808710394049, 0.08925326785323473011147,-0.01197542777363710259086,
-0.09411589519093271904993, 0.22265398194596541792123,-0.35150360884884301881925,
 0.47239160833206639278359,-0.56480093827255828387024, 0.62813784106864789524849,
-0.64982699034791369108177, 0.63938818272915676210744,-0.59031431141267853845989,
 0.51784449809989663293663,-0.41591840447847611139665, 0.30113615810048788290487,
-0.16535020559502128234541, 0.02679993636106233270699, 0.12310503588476454095435,
-0.26200705707159444024512, 0.39795419170440932132848,-0.50503519579049371834856,
 0.59390809659357879368713,-0.63896170178207334355847, 0.65785111602890267157306,
-0.62792728962009125570631, 0.57556090228459599877908,-0.48060128641587018805126,
 0.37727735962318537099591,-0.24550623603968940722453, 0.12521061644655290256978,
 0.00703177740641665847965,-0.10815149626779060887394, 0.20819809772003816661368,
-0.26262381495535164210864, 0.31018203676490591069737,-0.30499657985194800202677,
 0.29530102924083428961310,-0.23276948248933260199323, 0.17498349992870620983609,
-0.06925875872550875389511,-0.01774782242250545413742, 0.14472958299289198635940,
-0.23542299383470216556091, 0.35582134804536375005668,-0.41987316367190891108763,
 0.50385609740901737829688,-0.51386452931419712708561, 0.54137953923067938521996,
-0.48691388727531159652528, 0.46081934786046152563443,-0.35734112989938288107439,
 0.30586731559020269299864,-0.19010488720335599888145, 0.15589290897694907389592,
-0.07020445040876448217215, 0.09184838567402123232775,-0.06483189596786528841044,
 0.15822828562151658449153,-0.18878598830919135487250, 0.33604227692355315149086,
-0.38980982347956333322614, 0.54526054121735834101514,-0.57142422944320925282113,
 0.68772424667623632821289,-0.65067011376938788114188, 0.70911305866405682785114,
-0.61040813588848497062145, 0.62788208850171312036537,-0.49656214892485078804540,
 0.50123993747646156116105,-0.36329150774067292539726, 0.36631702048445086150608,
-0.22792781292939248705132, 0.22070701585875895922584,-0.07991139568514277202738,
 0.05281082785888737557434, 0.08014109082419015872567,-0.12429579444936976473368,
 0.22183269712945757889955,-0.27316377621978038536810, 0.30172136884110439813966,
-0.35762099392743945669437, 0.29066011417621101520226,-0.36413009950945862680527,
 0.18804499156607193399715,-0.30607439778397710350433, 0.02152540349678754016960,
-0.21580009366041408291892,-0.16125877503491309328254,-0.12924534858400021919422,
-0.30593257200305973286802,-0.06711990205851831037265,-0.36565454081990866885832,
-0.02127600308257220421138,-0.31230701600651583627055, 0.04396032717243000820240,
-0.14637503124218115280186, 0.16813762551198666028718, 0.08828124399461252658217,
 0.34773424558719895349768, 0.28449757309180573328078, 0.48659494372138956119755,
 0.27264728529928439204610, 0.40986212363109325584887,-0.08698123731244092327053,
 0.03795164113179277931565,-0.66077177979890056036538,-0.31279670195259245968344,
-0.85202386246316264006850, 0.01866986149615887380371,-0.15101179873555470312141,
 0.94535795028540015039198, 0.48237742850089643242839, 0.74707322200902592790328,
-0.56978674508970694745358,-0.70559538059460191750105,-0.96144958113485412543042,
 0.35460012779402438898302, 1.12821598248763255156746, 0.59490192814013553856967,
-0.72672939160915106526772,-1.50724901486331130939789, 1.07831281511605814316113,
 0.94863213815872671208496,-0.33549432044316018775021,-1.94326837641417937696531,
 2.37188720793334839598288,-0.53650939688982679509621,-1.40893057152890599503792,
 1.97461106105452555148361,-1.33493527171821857457701, 0.34625020886659119145889,
 0.39226641533747286239020,-0.73064695273103796857583, 0.76768497869450857962192,
-0.65013526748650019015940, 0.48529767269521839612167,-0.33041241789769365544416,
 0.20787690551474463651616,-0.12114095930860087346748, 0.06513372979887556424305,
-0.03204278230870652444118, 0.01424888656339123003958,-0.00563469653412903594136,
 0.00193897106710304062589,-0.00056365452715281295915, 0.00013264504130607146612,
-0.00002364110105322926388, 0.00000282978680927989219,-0.00000017016052966761765
])

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
        9.1898135789795544e-07, 1.0560236258145801e-06, 1.2135021985967225e-06,
        1.3944646218148563e-06, 1.6024129035298632e-06, 1.8413712856029165e-06,
        2.1159641213409353e-06, 2.4315053665758195e-06, 2.7941014160203776e-06,
        3.2107692750032131e-06, 3.6895723534572921e-06, 4.2397765100647990e-06,
        4.8720293663446341e-06, 5.5985663607919089e-06, 6.4334475306554683e-06,
        7.3928296035848751e-06, 8.4952786646994725e-06, 9.7621294498526205e-06,
        1.1217898218180574e-05, 1.2890757193898812e-05, 1.4813079758803961e-05,
        1.7022066946117209e-05, 1.9560467359658073e-05, 2.2477404450316962e-05,
        2.5829327159387539e-05, 2.9681102325729859e-05, 3.4107269997169212e-05,
        3.9193485939077079e-05, 4.5038179255745865e-05, 5.1754457203059970e-05,
        5.9472294054641452e-05, 6.8341046380689574e-05, 7.8532343415514477e-05,
        9.0243408448527221e-05, 1.0370087551467150e-04, 1.1916517524538292e-04,
        1.3693557475562244e-04, 1.5735596909953090e-04, 1.8082153637169376e-04,
        2.0778638524439160e-04, 2.3877234293696414e-04, 2.7437905368319086e-04,
        3.1529558312353735e-04, 3.6231375319194681e-04, 4.1634346555562428e-04,
        4.7843031014898881e-04, 5.4977579956440249e-04, 6.3176062087820603e-04,
        7.2597135488438745e-04, 8.3423117980991972e-04, 9.5863515369401899e-04,
        1.1015907582204575e-03, 1.2658644886123726e-03, 1.4546353912032167e-03,
        1.6715565847497888e-03, 1.9208259560556996e-03, 2.2072673980160704e-03,
        2.5364241622125167e-03, 2.9146661326298548e-03, 3.3493130964692705e-03,
        3.8487763976105395e-03, 4.4227217140193290e-03, 5.0822561091888648e-03,
        5.8401429774594740e-03, 6.7110490428654943e-03, 7.7118281914628686e-03,
        8.8618476298971925e-03, 1.0183362682074693e-02, 1.1701947477049383e-02,
        1.3446989862853833e-02, 1.5452260123909515e-02, 1.7756564507909106e-02,
        2.0404496209307014e-02, 2.3447298342546899e-02, 2.6943855605395223e-02,
        3.0961833823176868e-02, 3.5578989426519304e-02, 4.0884674203786789e-02,
        4.6981564448367090e-02, 5.3987647963491189e-02, 6.2038507377358290e-02,
        7.1289943955573878e-02, 8.1920992687256264e-02, 9.4137386993145250e-02,
        1.0817554010518914e-01, 1.2430712016577938e-01, 1.4284430758454120e-01,
        1.6414583639372404e-01, 1.8862393651527715e-01, 2.1675231128725500e-01,
        2.4907530463166816e-01, 2.8621843526798946e-01, 3.2890050183176844e-01,
        3.7794749315816500e-01, 4.3430857292399505e-01, 4.9907444798513595e-01,
        5.7349847587571534e-01, 6.5902091994412060e-01, 7.5729682151433553e-01,
        8.7022802845825153e-01, 1.0000000000000000e+00, 1.1491241000036050e+00,
        1.3204861972090955e+00, 1.5174025129350848e+00, 1.7436837970197381e+00,
        2.0037090739411751e+00, 2.3025103862617100e+00, 2.6458701753619409e+00,
        3.0404331839891712e+00, 3.4938350461726517e+00, 4.0148500529942019e+00,
        4.6135609537963891e+00, 5.3015540788430497e+00, 6.0921435594709612e+00,
        7.0006289848698273e+00, 8.0445914816976902e+00, 9.2442339463025291e+00,
        1.0622772013767669e+01, 1.2206883329864255e+01, 1.4027223820279268e+01,
        1.6119020948027551e+01, 1.8522755439841418e+01, 2.1284944674394648e+01,
        2.4459042892590269e+01, 2.8106475650897377e+01, 3.2297828536610695e+01,
        3.7114213149203508e+01, 4.2648836782420439e+01, 4.9008806183799550e+01,
        5.6317200298209784e+01, 6.4715452107403038e+01, 7.4366085659245940e+01,
        8.5455861253972017e+01, 9.8199389653503559e+01, 1.1284328525648564e+02,
        1.2967093861180905e+02, 1.4900800062891784e+02, 1.7122868461604187e+02,
        1.9676300810421012e+02, 2.2610511460175258e+02, 2.5982283632295099e+02,
        2.9856868294999509e+02, 3.4309246908417492e+02, 3.9425582475436693e+02,
        4.5304886979204105e+02, 5.2060937475742980e+02, 5.9824477922157064e+02,
        6.8745749350484289e+02, 7.8997397351448706e+02, 9.0777813134110693e+02,
        1.0431497281803040e+03, 1.1987084925641964e+03, 1.3774648176845108e+03,
        1.5828780189083436e+03, 1.8189232788935387e+03, 2.0901685758341446e+03,
        2.4018630835612289e+03, 2.7600387542291814e+03, 3.1716270494286800e+03,
        3.6445930787218194e+03, 4.1880897414655765e+03, 4.8126348548959668e+03,
        5.5303146962783048e+03, 6.3550178980975252e+03, 7.3027042226581152e+03,
        8.3917134174545281e+03, 9.6431201283206228e+03, 1.1081141738683076e+04,
        1.2733607027476588e+04, 1.4632494715248606e+04, 1.6814552320467552e+04,
        1.9322007302220827e+04, 2.2203384251427578e+04, 2.5514443944955918e+04,
        2.9319262435339933e+04, 3.3691471058779483e+04, 3.8715681358217458e+04,
        4.4489122496788041e+04, 5.1123522849071662e+04, 5.8747272182953282e+04,
        6.7507906274902969e+04, 7.7574962041275547e+04, 8.9143258438494697e+04,
        1.0243666662452392e+05, 1.1771244234227551e+05, 1.3526620436579350e+05,
        1.5543765535274608e+05, 1.7861715581389511e+05, 2.0525327841984577e+05,
        2.3586148883699448e+05, 2.7103412108532194e+05, 3.1145184046243853e+05,
        3.5789681586586579e+05, 4.1126785642601945e+05, 4.7259780537596118e+05,
        5.4307352776633098e+05, 6.2405887883026747e+05, 7.1712109748508944e+05,
        8.2406113574115187e+05, 9.4694851095649926e+05, 1.0881613554026424e+06
        ])


cos = np.array([
        4.8963534801291350e-04, -3.2447354678906376e-03,
        1.0952238450063470e-02, -2.5330877509368774e-02,
        4.5603964620893764e-02, -6.8751965018998398e-02,
        9.1055170868309943e-02, -1.0955571014254684e-01,
        1.2271886285618731e-01, -1.3032033375840532e-01,
        1.3300188933965829e-01, -1.3178555117707785e-01,
        1.2775414321103981e-01, -1.2185305954486332e-01,
        1.1484395399555812e-01, -1.0727994394369871e-01,
        9.9560982839136489e-02, -9.1939517526987313e-02,
        8.4590736600088562e-02, -7.7600019560680286e-02,
        7.1031733586497212e-02, -6.4890821187381778e-02,
        5.9196016139512468e-02, -5.3915848643058928e-02,
        4.9055780940812956e-02, -4.4567058676679240e-02,
        4.0457261282427506e-02, -3.6666941211080724e-02,
        3.3213645139589902e-02, -3.0027761601860024e-02,
        2.7141909473138912e-02, -2.4473692903024750e-02,
        2.2075499247215292e-02, -1.9847468573710642e-02,
        1.7867265714909791e-02, -1.6011026177709484e-02,
        1.4388810260485966e-02, -1.2844078700958564e-02,
        1.1528724140086060e-02, -1.0242160186713188e-02,
        9.1906684982280187e-03, -8.1147007847875829e-03,
        7.2916108124280446e-03, -6.3832919299231621e-03,
        5.7602923199031935e-03, -4.9801561547513692e-03,
        4.5359177425499482e-03, -3.8467889983387665e-03,
        3.5670373081260649e-03, -2.9327308045782176e-03,
        2.8105929470807834e-03, -2.1944294961424278e-03,
        2.2311101317977000e-03, -1.5941611249750095e-03,
        1.8000275448786861e-03, -1.0989791755701920e-03,
        1.4951672246892860e-03, -6.7966542661269437e-04,
        1.3003578629701799e-03, -3.0965377717395037e-04,
        1.2052335925762474e-03, 3.6106423804137471e-05,
        1.2052407466848229e-03, 3.8238957499821198e-04,
        1.3018966393838437e-03, 7.5508986898323120e-04,
        1.5033588524742210e-03, 1.1828448709658736e-03,
        1.8253815899842722e-03, 1.6989785802421903e-03,
        2.2927582125052783e-03, 2.3438821839617966e-03,
        2.9413764369870584e-03, 3.1679810467363173e-03,
        3.8210431939779545e-03, 4.2354668717297693e-03,
        4.9992627735887547e-03, 5.6289926876045100e-03,
        6.5661551724297195e-03, 7.4555038861999377e-03,
        8.6406325432364806e-03, 9.8532339747283270e-03,
        1.1377691235455435e-02, 1.2999446350215346e-02,
        1.4975945575967134e-02, 1.7117326788278663e-02,
        1.9682681677977296e-02, 2.2477549053870567e-02,
        2.5789277329789900e-02, 2.9383250822409085e-02,
        3.3599553377482397e-02, 3.8111654527735343e-02,
        4.3330483453611145e-02, 4.8751291534594260e-02,
        5.4854616146235546e-02, 6.0801696080654326e-02,
        6.7092281900802533e-02, 7.2264672303858538e-02,
        7.6684426491380325e-02, 7.7748740946138262e-02,
        7.5380953371380677e-02, 6.5034557293637621e-02,
        4.5881721095444840e-02, 1.0745068396029680e-02,
        -4.1063368584660725e-02, -1.1762555750610600e-01,
        -2.1286042283609111e-01, -3.2473691853928877e-01,
        -4.1889667952194926e-01, -4.5782693031141991e-01,
        -3.5587281978710816e-01, -6.6666111275991896e-02,
        4.1398275206290663e-01, 8.2532976945066749e-01,
        7.4155935075441382e-01, -2.2362177591765958e-01,
        -1.2468909757950963e+00, -5.6409324151001772e-01,
        1.6313369576910255e+00, 1.3693670120559179e-01,
        -1.9009211808491244e+00, 2.1011582166679363e+00,
        -1.4539662980149537e+00, 8.1038085852727904e-01,
        -4.1770604353721019e-01, 2.1941406706811120e-01,
        -1.2468136830432722e-01, 7.8269403871874657e-02,
        -5.3876922626623532e-02, 3.9875972155938078e-02,
        -3.1099121841349150e-02, 2.5148873022191463e-02,
        -2.0846754019080382e-02, 1.7576952842787749e-02,
        -1.4997498936892097e-02, 1.2906795835689719e-02,
        -1.1178868977228673e-02, 9.7305065956261819e-03,
        -8.5038781802097181e-03, 7.4569676118292445e-03,
        -6.5580949021930646e-03, 5.7826495893214484e-03,
        -5.1110604027691945e-03, 4.5274795600497757e-03,
        -4.0188948307658011e-03, 3.5745070845087227e-03,
        -3.1852787667663190e-03, 2.8435964846781139e-03,
        -2.5430124541388067e-03, 2.2780422183555945e-03,
        -2.0440036868883257e-03, 1.8368872847487388e-03,
        -1.6532500255010323e-03, 1.4901283117434534e-03,
        -1.3449656159356862e-03, 1.2155521334433954e-03,
        -1.0999741706045096e-03, 9.9657152340340712e-04,
        -9.0390146341796690e-04, 8.2070821214142624e-04,
        -7.4589699233026084e-04, 6.7851191397762147e-04,
        -6.1771708408859378e-04, 5.6278043567672292e-04,
        -5.1305986066984307e-04, 4.6799130039041709e-04,
        -4.2707850066888483e-04, 3.8988418417160815e-04,
        -3.5602243103677837e-04, 3.2515209207027528e-04,
        -2.9697108796156711e-04, 2.7121146954596444e-04,
        -2.4763512919344816e-04, 2.2603006942123759e-04,
        -2.0620715318264257e-04, 1.8799727229799711e-04,
        -1.7124887513989931e-04, 1.5582580442025662e-04,
        -1.4160541245686331e-04, 1.2847693068685222e-04,
        -1.1634006683069419e-04, 1.0510380008256007e-04,
        -9.4685353393630364e-05, 8.5009334765365045e-05,
        -7.6007041962903359e-05, 6.7615916583788370e-05,
        -5.9779122420642347e-05, 5.2445208782622779e-05,
        -4.5567833380983288e-05, 3.9105728752033442e-05,
        -3.3023834075077836e-05, 2.7297600234901939e-05,
        -2.1922306831363493e-05, 1.6926318238944308e-05,
        -1.2382788632218085e-05, 8.4107903563777150e-06,
        -5.1548458966523642e-06, 2.7349863491621416e-06,
        -1.1775367755450835e-06, 3.6617415171458042e-07,
        -6.1920666642164617e-08])

sin = np.array([
        -5.8602704468975832e-10, 4.8048608865691248e-09,
        -1.9771446411637653e-08, 5.5269671219669613e-08,
        -1.1943308954933238e-07, 2.1463221286467290e-07,
        -3.3618689168303137e-07, 4.7460375561491547e-07,
        -6.2010393774690030e-07, 7.6676687291580134e-07,
        -9.1382097135454670e-07, 1.0643058622431453e-06,
        -1.2227765259769829e-06, 1.3937672563527502e-06,
        -1.5810499935024131e-06, 1.7877429029287643e-06,
        -2.0163554732931835e-06, 2.2692361668040396e-06,
        -2.5484456753753459e-06, 2.8562454180065290e-06,
        -3.1946379472885138e-06, 3.5660627245815181e-06,
        -3.9725086879521357e-06, 4.4166880900049553e-06,
        -4.9004832484318760e-06, 5.4270061911978838e-06,
        -5.9978954164211254e-06, 6.6168927621162747e-06,
        -7.2851116079354270e-06, 8.0072717151050409e-06,
        -8.7834623074537534e-06, 9.6200272666869681e-06,
        -1.0515203893806995e-05, 1.1478097778214146e-05,
        -1.2503644637963446e-05, 1.3605711023452051e-05,
        -1.4773445946482213e-05, 1.6028992588028293e-05,
        -1.7351362255724533e-05, 1.8777046268328975e-05,
        -2.0267260460262344e-05, 2.1883367382220654e-05,
        -2.3555377916627267e-05, 2.5387811572432832e-05,
        -2.7256008695400006e-05, 2.9339419633021675e-05,
        -3.1417660496931868e-05, 3.3800354252141579e-05,
        -3.6099620853251731e-05, 3.8851364451479363e-05,
        -4.1374814496544362e-05, 4.4599458677506889e-05,
        -4.7332670414243943e-05, 5.1188885345556143e-05,
        -5.4081411276216648e-05, 5.8817223325521029e-05,
        -6.1748595688031771e-05, 6.7759540724914998e-05,
        -7.0477668174613142e-05, 7.8405578637928987e-05,
        -8.0416390943704734e-05, 9.1318327863896956e-05,
        -9.1689586953474345e-05, 1.0732811343723065e-04,
        -1.0434243728703050e-04, 1.2768616500544454e-04,
        -1.1822963938719640e-04, 1.5431866320930474e-04,
        -1.3280640984775213e-04, 1.9025153589598097e-04,
        -1.4674319163166278e-04, 2.4032678429752795e-04,
        -1.5722591368834836e-04, 3.1241839074880459e-04,
        -1.5869843609971545e-04, 4.1950672592888611e-04,
        -1.4061987157747521e-04, 5.8323092045218519e-04,
        -8.3489051752646859e-05, 8.3998754956665807e-04,
        4.8166642275560894e-05, 1.2514134018618409e-03,
        3.1809381315641577e-04, 1.9223958368384385e-03,
        8.3977376473052937e-04, 3.0319286374562394e-03,
        1.8135564849093052e-03, 4.8856220017935168e-03,
        3.5901028334771145e-03, 8.0038676866840078e-03,
        6.7764383041831900e-03, 1.3266048500372144e-02,
        1.2406252926137538e-02, 2.2134422870406390e-02,
        2.2191911862939541e-02, 3.6963935749189471e-02,
        3.8829525301567067e-02, 6.1308824849773652e-02,
        6.6140947825550744e-02, 9.9797901699075597e-02,
        1.0824048046010233e-01, 1.5616018092254505e-01,
        1.6533327222825650e-01, 2.2564430322919368e-01,
        2.2062815143611000e-01, 2.7477185256445674e-01,
        2.1146243753121693e-01, 2.0743923168031805e-01,
        2.9273136481935752e-03, -1.1875521395824359e-01,
        -4.8782254186329055e-01, -5.6383218637505039e-01,
        -7.0493717238569553e-01, -4.5783079000773212e-02,
        4.9572887642641605e-01, 1.2793243232592335e+00,
        7.6984445533323119e-04, -1.0809022247360809e+00,
        -9.7826142685914264e-01, 2.4412168561816161e+00,
        -1.5742192933752286e+00, 1.0474779812704657e-01,
        6.8117592802985916e-01, -8.1735467756558944e-01,
        6.8894183180797697e-01, -5.2159909808793825e-01,
        3.8460793473043292e-01, -2.8421923628404017e-01,
        2.1211949229810861e-01, -1.5993330303166020e-01,
        1.2163508069999608e-01, -9.3164674694277416e-02,
        7.1777818840364910e-02, -5.5578938642792224e-02,
        4.3227895023256115e-02, -3.3758552896064629e-02,
        2.6463811011765734e-02, -2.0820205680643536e-02,
        1.6436812641415149e-02, -1.3019659797407497e-02,
        1.0346416487448091e-02, -8.2481070073694342e-03,
        6.5957434607522691e-03, -5.2904671538314026e-03,
        4.2562264095930237e-03, -3.4343067314751316e-03,
        2.7792243881857633e-03, -2.2556298932160214e-03,
        1.8359634980842480e-03, -1.4986732799921144e-03,
        1.2268558973329326e-03, -1.0072161412856525e-03,
        8.2926783836384434e-04, -6.8471812797056032e-04,
        5.6699154764304634e-04, -4.7086106657177131e-04,
        3.9216119658083232e-04, -3.2756429146830335e-04,
        2.7440564055551230e-04, -2.3054635199842104e-04,
        1.9426558644629048e-04, -1.6417564894421281e-04,
        1.3915493002405042e-04, -1.1829482010512426e-04,
        1.0085758994667589e-04, -8.6242897629566575e-05,
        7.3961097310559017e-05, -6.3611923024107272e-05,
        5.4867429384991716e-05, -4.7458310810428852e-05,
        4.1162907677974120e-05, -3.5798353789658158e-05,
        3.1213433790926005e-05, -2.7282808824825689e-05,
        2.3902339183058118e-05, -2.0985288286859604e-05,
        1.8459236223596745e-05, -1.6263565819420051e-05,
        1.4347411857118876e-05, -1.2667986114133353e-05,
        1.1189208624850740e-05, -9.8805899457079486e-06,
        8.7163210286869178e-06, -7.6745372470670224e-06,
        6.7367316776180222e-06, -5.8873002689814021e-06,
        5.1132081051238435e-06, -4.4037712527232031e-06,
        3.7505516230790782e-06, -3.1473609977624753e-06,
        2.5903624227147574e-06, -2.0782393943789611e-06,
        1.6123579385552092e-06, -1.1967503151651560e-06,
        8.3762259929749212e-07, -5.4201864582008082e-07,
        3.1540800169690699e-07, -1.5849870094735947e-07,
        6.4517602920670347e-08, -1.8937067758737377e-08,
        3.0164202265996216e-09
        ])


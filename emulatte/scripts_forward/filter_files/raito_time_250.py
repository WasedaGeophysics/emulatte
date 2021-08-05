import numpy as np

base = np.array([
        2.304028217899656e-03, 1.213938942054682e-02, 2.983233534919805e-02,
        5.538401370728252e-02, 8.879088715435302e-02, 1.300478163872114e-01,
        1.791483507585312e-01, 2.360847833031769e-01, 3.008481663046538e-01,
        3.734283174761970e-01, 4.538138234189262e-01, 5.419920422281592e-01,
        6.379491058696161e-01, 7.416699225598293e-01, 8.531381792484112e-01,
        9.723363442460027e-01, 1.099245670018774e+00, 1.233846196160044e+00,
        1.376116752544362e+00, 1.526034962666792e+00, 1.683577247168792e+00,
        1.848718827551252e+00, 2.021433730074804e+00, 2.201694789847267e+00,
        2.389473655097760e+00, 2.584740791637088e+00, 2.787465487503852e+00,
        2.997615857795502e+00, 3.215158849683760e+00, 3.440060247613651e+00,
        3.672284678685237e+00, 3.911795618217404e+00, 4.158555395492698e+00,
        4.412525199682399e+00, 4.673665085950887e+00, 4.941933981738302e+00,
        5.217289693220636e+00, 5.499688911946102e+00, 5.789087221646788e+00,
        6.085439105224616e+00, 6.388697951910366e+00, 6.698816064594752e+00,
        7.015744667330303e+00, 7.339433913003006e+00, 7.669832891172327e+00,
        8.006889636078519e+00, 8.350551134815930e+00, 8.700763335670963e+00,
        9.057471156623450e+00, 9.420618494010029e+00, 9.790148231348297e+00,
        1.016600224832015e+01, 1.054812142991307e+01, 1.093644567571789e+01,
        1.133091390938142e+01, 1.173146408821274e+01, 1.213803321294132e+01,
        1.255055733762573e+01, 1.296897157971110e+01, 1.339321013023406e+01,
        1.382320626417320e+01, 1.425889235094381e+01, 1.470019986503485e+01,
        1.514705939678683e+01, 1.559940066330866e+01, 1.605715251953191e+01,
        1.652024296940058e+01, 1.698859917719483e+01, 1.746214747898669e+01,
        1.794081339422605e+01, 1.842452163745514e+01, 1.891319613014951e+01,
        1.940676001268385e+01, 1.990513565642060e+01, 2.040824467591947e+01,
        2.091600794126617e+01, 2.142834559051797e+01, 2.194517704226465e+01,
        2.246642100830240e+01, 2.299199550641906e+01, 2.352181787328836e+01,
        2.405580477747143e+01, 2.459387223252326e+01, 2.513593561020230e+01,
        2.568190965378088e+01, 2.623170849145462e+01, 2.678524564984845e+01,
        2.734243406761733e+01, 2.790318610913940e+01, 2.846741357829949e+01,
        2.903502773236076e+01, 2.960593929592233e+01, 3.018005847496064e+01,
        3.075729497095249e+01, 3.133755799507730e+01, 3.192075628249657e+01,
        3.250679810670817e+01, 3.309559129397325e+01, 3.368704323781353e+01,
        3.428106091357655e+01, 3.487755089306685e+01, 3.547641935924046e+01,
        3.607757212096061e+01, 3.668091462781236e+01, 3.728635198497357e+01,
        3.789378896814022e+01, 3.850313003850341e+01, 3.911427935777589e+01,
        3.972714080326567e+01, 4.034161798299442e+01, 4.095761425085815e+01,
        4.157503272182785e+01, 4.219377628718783e+01, 4.281374762980906e+01,
        4.343484923945549e+01, 4.405698342812054e+01, 4.468005234539173e+01,
        4.530395799384073e+01, 4.592860224443659e+01, 4.655388685197964e+01,
        4.717971347055371e+01, 4.780598366899411e+01, 4.843259894636911e+01,
        4.905946074747232e+01, 4.968647047832366e+01, 5.031352952167634e+01,
        5.094053925252768e+01, 5.156740105363089e+01, 5.219401633100589e+01,
        5.282028652944629e+01, 5.344611314802036e+01, 5.407139775556341e+01,
        5.469604200615927e+01, 5.531994765460827e+01, 5.594301657187946e+01,
        5.656515076054451e+01, 5.718625237019094e+01, 5.780622371281217e+01,
        5.842496727817215e+01, 5.904238574914185e+01, 5.965838201700558e+01,
        6.027285919673433e+01, 6.088572064222411e+01, 6.149686996149659e+01,
        6.210621103185978e+01, 6.271364801502643e+01, 6.331908537218764e+01,
        6.392242787903939e+01, 6.452358064075955e+01, 6.512244910693315e+01,
        6.571893908642345e+01, 6.631295676218647e+01, 6.690440870602674e+01,
        6.749320189329183e+01, 6.807924371750343e+01, 6.866244200492270e+01,
        6.924270502904750e+01, 6.981994152503935e+01, 7.039406070407767e+01,
        7.096497226763924e+01, 7.153258642170050e+01, 7.209681389086060e+01,
        7.265756593238267e+01, 7.321475435015155e+01, 7.376829150854537e+01,
        7.431809034621912e+01, 7.486406438979770e+01, 7.540612776747673e+01,
        7.594419522252858e+01, 7.647818212671163e+01, 7.700800449358094e+01,
        7.753357899169760e+01, 7.805482295773535e+01, 7.857165440948202e+01,
        7.908399205873383e+01, 7.959175532408052e+01, 8.009486434357940e+01,
        8.059323998731614e+01, 8.108680386985048e+01, 8.157547836254486e+01,
        8.205918660577395e+01, 8.253785252101332e+01, 8.301140082280517e+01,
        8.347975703059942e+01, 8.394284748046809e+01, 8.440059933669133e+01,
        8.485294060321317e+01, 8.529980013496515e+01, 8.574110764905619e+01,
        8.617679373582681e+01, 8.660678986976595e+01, 8.703102842028889e+01,
        8.744944266237428e+01, 8.786196678705868e+01, 8.826853591178727e+01,
        8.866908609061858e+01, 8.906355432428211e+01, 8.945187857008693e+01,
        8.983399775167985e+01, 9.020985176865170e+01, 9.057938150598997e+01,
        9.094252884337655e+01, 9.129923666432904e+01, 9.164944886518407e+01,
        9.199311036392149e+01, 9.233016710882768e+01, 9.266056608699699e+01,
        9.298425533266970e+01, 9.330118393540525e+01, 9.361130204808964e+01,
        9.391456089477538e+01, 9.421091277835322e+01, 9.450031108805391e+01,
        9.478271030677936e+01, 9.505806601826170e+01, 9.532633491404911e+01,
        9.558747480031761e+01, 9.584144460450730e+01, 9.608820438178259e+01,
        9.632771532131477e+01, 9.655993975238636e+01, 9.678484115031624e+01,
        9.700238414220451e+01, 9.721253451249615e+01, 9.741525920836291e+01,
        9.761052634490224e+01, 9.779830521015273e+01, 9.797856626992520e+01,
        9.815128117244875e+01, 9.831642275283122e+01, 9.847396503733322e+01,
        9.862388324745564e+01, 9.876615380383996e+01, 9.890075432998123e+01,
        9.902766365575400e+01, 9.914686182075158e+01, 9.925833007744018e+01,
        9.936205089413039e+01, 9.945800795777184e+01, 9.954618617658107e+01,
        9.962657168252380e+01, 9.969915183369534e+01, 9.976391521669683e+01,
        9.982085164924146e+01, 9.986995218361278e+01, 9.991120911284565e+01,
        9.994461598629272e+01, 9.997016766465080e+01, 9.998786061057945e+01,
        9.999769597178209e+01
        ])


j0 = np.array([
        9.828579942414525e-02, 9.966080429905441e-02, 9.984209343228008e-02,
        9.979508762493577e-02, 9.957103943139099e-02, 9.911765682374263e-02,
        9.834800932787176e-02, 9.715391691777801e-02, 9.541095246887138e-02,
        9.298218686048622e-02, 8.972240747412025e-02, 8.548345258665230e-02,
        8.012099829186260e-02, 7.350299282566608e-02, 6.551981114392411e-02,
        5.609605702566009e-02, 4.520375513610055e-02, 3.287644666836109e-02,
        1.922343240309277e-02, 4.443107386016678e-03, -1.116597753031233e-02,
        -2.719797900700406e-02, -4.313842347400363e-02, -5.837064684305756e-02,
        -7.219191555354526e-02, -8.384081517002559e-02, -9.253687366563521e-02,
        -9.753242434473859e-02, -9.817539708195525e-02, -9.398009128937253e-02,
        -8.470112593333556e-02, -7.040384974711862e-02, -5.152277414605046e-02,
        -2.889837756639459e-02, -3.782285254712953e-03, 2.219827929617858e-02,
        4.712343337575472e-02, 6.891345481895743e-02, 8.550810754541499e-02,
        9.508187386687045e-02, 9.627660099055381e-02, 8.842556147585838e-02,
        7.173717076761407e-02, 4.740413295316667e-02, 1.760606883550404e-02,
        -1.461818483256700e-02, -4.563805840662560e-02, -7.162807573580893e-02,
        -8.906013768687035e-02, -9.523368425962901e-02, -8.876661817168936e-02,
        -6.995940652450684e-02, -4.094793297678034e-02, -5.580518841227993e-03,
        3.100849601524601e-02, 6.310523709745319e-02, 8.532151560723773e-02,
        9.356306250956714e-02, 8.588679313166836e-02, 6.306322327762040e-02,
        2.868747494898507e-02, -1.125079225427065e-02, -4.932817179389668e-02,
        -7.805156832352808e-02, -9.138791573108145e-02, -8.614665334819208e-02,
        -6.288711015527471e-02, -2.608409410107041e-02, 1.657829942470613e-02,
        5.572283087152113e-02, 8.233584501275383e-02, 8.992354867226386e-02,
        7.624299486871268e-02, 4.412470305876092e-02, 1.080824506688354e-03,
        -4.232201666832165e-02, -7.501548342349848e-02, -8.830920349739070e-02,
        -7.834085773575893e-02, -4.737872015442737e-02, -3.510192933249428e-03,
        4.133134184375355e-02, 7.456709447124694e-02, 8.656064973594962e-02,
        7.355796043332832e-02, 3.904783095846685e-02, -6.984560802562902e-03,
        -5.082758516249672e-02, -7.910931036274961e-02, -8.296250029073596e-02,
        -6.096749345261541e-02, -1.985142552102767e-02, 2.741394719895245e-02,
        6.563529221040448e-02, 8.231858262757298e-02, 7.188263670218092e-02,
        3.770257865969490e-02, -8.818296855075182e-03, -5.196153666321151e-02,
        -7.701651985601977e-02, -7.540644105054180e-02, -4.779044172236651e-02,
        -3.958557143498125e-03, 4.054929426112576e-02, 6.996848106710708e-02,
        7.400792519363880e-02, 5.161416240095949e-02, 1.135401234356790e-02,
        -3.178736782086535e-02, -6.202615650462178e-02, -6.869045472102359e-02,
        -5.014645100957605e-02, -1.421850553412135e-02, 2.503984428089361e-02,
        5.291250080873955e-02, 5.968383226874570e-02, 4.414684940638246e-02,
        1.362495913571569e-02, -1.936530213122743e-02, -4.231106517113638e-02,
        -4.756028242249998e-02, -3.489513166365216e-02, -1.094198561370729e-02,
        1.408693189118120e-02, 3.075496185953878e-02, 3.398062065793993e-02,
        2.446288883875845e-02, 7.627108110447284e-03, -9.161683482539764e-03,
        -1.974611325604035e-02, -2.135393084441232e-02, -1.507710120807284e-02,
        -4.744797681247362e-03, 5.068864465480888e-03, 1.093855577574561e-02,
        1.164390998369695e-02, 8.141360302926686e-03, 2.699975533222753e-03,
        -2.265352821907431e-03, -5.131137925881854e-03, -5.462100191792395e-03,
        -3.851957023208594e-03, -1.418809992663650e-03, 7.541265829592288e-04,
        2.001759905357183e-03, 2.189786250887842e-03, 1.596339131953578e-03,
        6.797384448420897e-04, -1.437324993960939e-04, -6.323828291728057e-04,
        -7.432392048466919e-04, -5.765107790494071e-04, -2.890356656682678e-04,
        -1.944940410702885e-05, 1.522519860557339e-04, 2.092761337904987e-04,
        1.789463538281116e-04, 1.060072161103936e-04, 3.073321950030855e-05,
        -2.249118751235755e-05, -4.650519986591545e-05, -4.639905217229209e-05,
        -3.261608179523543e-05, -1.529046938427277e-05, -1.174689595989863e-06,
        6.973292960412407e-06, 9.447490237110817e-06, 8.111620363497764e-06,
        5.105391769014650e-06, 2.066765028989830e-06, -1.250288488437870e-07,
        -1.251981301724389e-06, -1.505676700462523e-06, -1.233938333526242e-06,
        -7.642426544307674e-07, -3.197530949939827e-07, -7.313865614290079e-09,
        1.549449630304627e-07, 1.984335661937886e-07, 1.706333769568400e-07,
        1.140833519785820e-07, 5.744037406752442e-08, 1.492031097621379e-08,
        -1.006161144258089e-08, -2.023956089868063e-08, -2.063179518189517e-08,
        -1.609096164994456e-08, -1.017694463807695e-08, -4.949811001680341e-09,
        -1.248777959171183e-09, 8.728137852245139e-10, 1.749810955865940e-09,
        1.824235890933766e-09, 1.487678420965584e-09, 1.019149725942864e-09,
        5.819638134928672e-10, 2.486066257288662e-10, 3.272523361123246e-11,
        -8.293189979914553e-11, -1.267410246477002e-10, -1.265771034198689e-10,
        -1.044761547801668e-10, -7.533787573763742e-11, -4.776198459515831e-11,
        -2.569347277708291e-11, -1.010482071101596e-11, -3.578684783943035e-13,
        4.850607868957370e-12, 6.918874159831228e-12, 7.052571556989883e-12,
        6.161847276664374e-12, 4.863372125023393e-12, 3.531001481683074e-12,
        2.360347305396982e-12, 1.428800788995454e-12, 7.433185599659140e-13,
        2.745553066301596e-13, -2.098349182041270e-14, -1.879887773701259e-13,
        -2.657932135214407e-13, -2.858783390063469e-13, -2.717092285758947e-13,
        -2.397216276673238e-13, -2.007321471181883e-13, -1.613625168724535e-13,
        -1.252804906964138e-13, -9.418837129963084e-14, -6.856045030987623e-14,
        -4.816359887705532e-14, -3.240602925991836e-14, -2.055819710503736e-14,
        -1.188341199101648e-14, -5.707721576091140e-15, -1.450981843542954e-15,
        1.365432395209046e-15, 3.123864504638975e-15, 4.121866100342093e-15,
        4.585861107079069e-15, 4.684667080650126e-15, 4.541694239535037e-15,
        4.245341571541214e-15, 3.857513329130626e-15, 3.420392540040210e-15,
        2.961703683447929e-15, 2.498723385207422e-15, 2.041287610021522e-15,
        1.594016314934538e-15, 1.157945084997330e-15, 7.317399990792992e-16,
        3.130496644491803e-16
        ])

j1 = np.array([
        2.264536560054172e-04, 1.209880745203144e-03, 2.979406724794770e-03,
        5.532710649971882e-03, 8.864308053259733e-03, 1.296319710623620e-02,
        1.780982284772187e-02, 2.337241195808794e-02, 2.960275741818770e-02,
        3.643158260142829e-02, 4.376367613793942e-02, 5.147306415053947e-02,
        5.939857379853884e-02, 6.734023756968888e-02, 7.505708468708876e-02,
        8.226695430287602e-02, 8.864903116801308e-02, 9.384983089742700e-02,
        9.749332834391396e-02, 9.919580653515324e-02, 9.858578279435161e-02,
        9.532902392011004e-02, 8.915818185924701e-02, 7.990596585311784e-02,
        6.754003546287068e-02, 5.219699397353384e-02, 3.421205481553292e-02,
        1.414024712723985e-02, -7.235447937571780e-03, -2.891372726338310e-02,
        -4.970660048725934e-02, -6.829611400481622e-02, -8.331729740635771e-02,
        -9.346569353385716e-02, -9.762433655718721e-02, -9.500106825641769e-02,
        -8.526305101465055e-02, -6.865183721070975e-02, -4.606011450377025e-02,
        -1.905105050329428e-02, 1.019626497049600e-02, 3.902592571951180e-02,
        6.453025474311476e-02, 8.384997836576848e-02, 9.452470585221250e-02,
        9.485493296547227e-02, 8.422536159457099e-02, 6.333325120631475e-02,
        3.426749398914974e-02, 3.961236285176257e-04, -3.395735510465855e-02,
        -6.403643767923183e-02, -8.532287256313327e-02, -9.426931611747878e-02,
        -8.897653775828618e-02, -6.968483261755813e-02, -3.895867912757926e-02,
        -1.479661936165647e-03, 3.657596439232996e-02, 6.851930496702448e-02,
        8.835066560961846e-02, 9.197068516810246e-02, 7.815197557528629e-02,
        4.903272900467932e-02, 9.960471784309071e-03, -3.136685702172082e-02,
        -6.636237216772427e-02, -8.735829297028211e-02, -8.938448994063974e-02,
        -7.149442585415221e-02, -3.726721735561030e-02, 5.730491795557686e-03,
        4.750407135695894e-02, 7.792750427942979e-02, 8.926411136105283e-02,
        7.827626535401916e-02, 4.732435497209352e-02, 4.062571775865735e-03,
        -4.031550950052659e-02, -7.392562048112651e-02, -8.743006492135540e-02,
        -7.677319373787246e-02, -4.458316414636109e-02, 2.765086772610481e-04,
        4.501464964393829e-02, 7.652400338662849e-02, 8.529155484079984e-02,
        6.842424678367455e-02, 3.078809303374962e-02, -1.622044184770864e-02,
        -5.802721276535564e-02, -8.140523200658625e-02, -7.876650068515292e-02,
        -5.079610705273682e-02, -6.464426659687245e-03, 3.972164751137570e-02,
        7.243319297361611e-02, 8.067931222300397e-02, 6.165081313066673e-02,
        2.185140663632107e-02, -2.499154887085311e-02, -6.261844683055963e-02,
        -7.794431010695425e-02, -6.575343980793361e-02, -3.064034259091100e-02,
        1.459246507229322e-02, 5.357297668218915e-02, 7.236576488076765e-02,
        6.460757706058180e-02, 3.381151037633848e-02, -8.067372365292499e-03,
        -4.526933205673900e-02, -6.426983982347337e-02, -5.885576964594783e-02,
        -3.225542223630932e-02, 4.465869970868479e-03, 3.701528943082866e-02,
        5.355275786627092e-02, 4.912379535978623e-02, 2.712023837370525e-02,
        -2.657750694835797e-03, -2.833939608907179e-02, -4.078908680211588e-02,
        -3.688993914832463e-02, -2.010627889571185e-02, 1.633926574800961e-03,
        1.958298601884326e-02, 2.767955121500291e-02, 2.450814096137264e-02,
        1.312947695613619e-02, -8.340392684647124e-04, -1.180619364221551e-02,
        -1.638758731520054e-02, -1.423539982833650e-02, -7.574570212796236e-03,
        1.919439041905531e-04, 6.031088751423789e-03, 8.335725014368661e-03,
        7.183013729941636e-03, 3.883533661096683e-03, 1.741820284529615e-04,
        -2.538012438712803e-03, -3.600905268897543e-03, -3.138484088050669e-03,
        -1.775760846176198e-03, -2.600561437291729e-04, 8.458015478968546e-04,
        1.304949732365462e-03, 1.182747710449159e-03, 7.213106215246146e-04,
        1.933521583072995e-04, -2.036160122524840e-04, -3.883401342914031e-04,
        -3.807298487331901e-04, -2.568979346726202e-04, -1.017628873339358e-04,
        2.297041387331429e-05, 9.018085696994396e-05, 1.023261209187176e-04,
        7.847018443166041e-05, 4.112219203390683e-05, 7.479148860015968e-06,
        -1.382581590484248e-05, -2.175324984508623e-05, -1.984109376707791e-05,
        -1.305273326959776e-05, -5.588359318753452e-06, 6.411624221945432e-08,
        3.101913941171539e-06, 3.866979939535994e-06, 3.205519253196622e-06,
        1.976858129619115e-06, 7.934527835440294e-07, -3.929949442197726e-08,
        -4.624585308593031e-07, -5.599432195801955e-07, -4.643834766556799e-07,
        -2.957219782186684e-07, -1.341898942958265e-07, -1.814321555630797e-08,
        4.527261794723360e-08, 6.624226631297206e-08, 6.075798563374456e-08,
        4.345098655011839e-08, 2.449075818125060e-08, 9.273105637468503e-09,
        -4.818905855324972e-10, -5.256884119366565e-09, -6.474926938537987e-09,
        -5.664009858906954e-09, -4.035717565382576e-09, -2.363817754408762e-09,
        -1.032776219741178e-09, -1.532150228770800e-10, 3.196668332997566e-10,
        4.946669618032692e-10, 4.871982295505074e-10, 3.909509736056888e-10,
        2.694249647428633e-10, 1.585171846222275e-10, 7.362151881962121e-11,
        1.725267762451350e-11, -1.472736534370282e-11, -2.884213916134782e-11,
        -3.152236549415545e-11, -2.795830829462268e-11, -2.179643662266030e-11,
        -1.528606233265268e-11, -9.605949153673284e-12, -5.212435968161844e-12,
        -2.132770572779098e-12, -1.793783856789497e-13, 9.111403382120695e-13,
        1.398646593940594e-12, 1.501752178862154e-12, 1.386089640295040e-12,
        1.166805274496697e-12, 9.174050664562203e-13, 6.802360163690123e-13,
        4.761362000443448e-13, 3.122328080860736e-13, 1.876992905002597e-13,
        9.769578565863485e-14, 3.587172482675328e-14, -4.178116275859639e-15,
        -2.816700430060989e-14, -4.081777121429044e-14, -4.582504766234897e-14,
        -4.595247598197403e-14, -4.318601671917824e-14, -3.889756594659202e-14,
        -3.399506361410507e-14, -2.904894619484099e-14, -2.439269865349170e-14,
        -2.019937700300260e-14, -1.653775239345791e-14, -1.341216466810146e-14,
        -1.078991197449261e-14, -8.619437411340996e-15, -6.841928726484080e-15,
        -5.398339264138097e-15, -4.233318534950602e-15, -3.297122612968905e-15,
        -2.546252563608400e-15, -1.943329460947742e-15, -1.456541205363109e-15,
        -1.058874199108483e-15, -7.272598152921200e-16, -4.417201435824073e-16,
        -1.847984204751173e-16
        ])

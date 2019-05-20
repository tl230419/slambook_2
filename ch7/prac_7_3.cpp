#include <iostream>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int features_num = 1000; 		// 最大特征点数量
float scale_factor = 1.2f; 		// 金字塔之间尺度参数
int levels_num = 8;			// 金字塔层数
int default_fast_threshold = 20;	// fast角点检测时候的阈值
int min_fast_threshold = 7;		// 最小fast检测阀值
int EDGE_THRESHOLD = 19;		// 过滤边缘效应的阀值
int PATCH_SIZE = 31;
int HALF_PATH_SIZE = 15;

// 定义一个四叉树节点类型的类，类里面定义了一些成员及函数
class extractor_node
{
public:
	extractor_node() : b_no_more(false) {}
	// 分配节点函数
	void divide_node(extractor_node &node1, extractor_node &node2, extractor_node &node3, extractor_node &node4);
	std::vector<cv::KeyPoint> vkeys;	// 节点keypoints容器
	cv::Point2i UL, UR, BL, BR;		// 二维整数点类型数据u的上下左右像素
	std::list<extractor_node>::iterator lit;// 节点类型列表迭代器
	bool b_no_more = false;			// 确认是否含有i个特征点	
};

// 分配节点函数
void divide_node(extractor_node &node1, extractor_node &node2, extractor_node &node3, extractor_node &node4)
{
  /*
   * 
   * -----------------------------------------------------------------------
   *	/                               /                               /
   *	/                               /				/
   *	/                               /				/
   *	/                               /				/
   *	/             n1                /		n2		/
   *	/                               /				/
   *	/                               /				/
   *	/                               /				/
   *	/                               / 				/
   *	/-------------------------------/-------------------------------	 
   *	/                               / 				/
   *	/                               / 				/
   *	/                               / 				/
   *	/               n3              / 		n4		/
   *	/                               / 				/
   *	/                               / 				/
   *	/                               / 				/
   *	/                               / 				/
   *	/                               /				/
   *---------------------------------------------------------------------------
   * 
   */
	const int half_x = ceil(static_cast<float>(UR.x - UL.x) / 2);
	const int half_y = ceil(static_cast<float>(BR.y - UL.y) / 2);

	// 矩阵切四块
	node1.UL = UL;
	node1.UR = cv::Point2i(UL.x + half_y, UL.y);
	node1.BL = cv::Point2i(UL.x, UL.y + half_);
	node1.BR = cv::Point2i(UL.x + half_x, UL.y + half_y);
	node1.vkeys.reserve(vkeys.size());

	node2.UL = node1.UR;
	node2.UR = UR;
	node2.BL = node1.BR;
	node2.BR = cv::Point2i(UR.x, UL.y + half_y);
	node2.vkeys.reserve(vkeys.size());

	node3.UL = node1.BL;
	node3.UR = node1.BR;
	node3.BL = BL;
	node3.BR = cv::Point2i(node1.BR.x, BL.y);
	node3.vkeys.reserve(vkeys.size());

	node4.UL = node3.UR;
	node4.UR = node2.BR;
	node4.BL = node3.BR;
	node4.BR = BR;
	node4.vkeys.reserve(vkeys.size());

	for (size_t i = 0; i < vkeys.size(); i++)
	{
		const cv::KeyPoint &kp = vkeys[i];
		if (kp.pt.x < node1.UR.x)
		{
			if (kp.pt.y < node1.BR.y)
			{
				node1.vkeys.push_back(kp);
			}
			else
			{
				node3.vkeys.push_back(kp);
			}
		}
		else if (kp.pt.y < node1.BR.y)
		{
			node2.vkeys.push_back(kp);
		}
		else
		{
			node4.vkeys.push_back(kp);
		}
	}

	if (node1.vkeys.size() == 1)
	{
		node1.b_no_more = true;
	}
	if (node2.vkeys.size() == 1)
	{
		node2.b_no_more = true;
	}
	if (node3.vkeys.size() == 1)
	{
		node3.b_no_more = true;
	}
	if (node4.vkeys.size() == 1)
	{
		node4.b_no_more = true;
	}
}

// 计算描述子的pattern, 高斯分布，也可以使用其他定义的pattern
static int bit_pattern_31[256 * 4] = 
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

int main(int argc, char** argv)
{
	// 读取图像
	Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (img.empty())
	{
		cout << "No picture was found ..." << endl;
		cout << "Usage: prac_7_3 img" << endl;
		return 1;
	}
	else
	{
		cout << "Img loaded successed!" << endl;
	}

	vector<int> features_num_per_level;	// 每层特征点数
	vector<int> umax;			// 存储特征方向，每个v对应的最大u
	vector<float> vec_scale_factor;		// 存储每层尺度因子

	/******************** 构建图像金字塔 *************************/
	
	// 初始化每层金字塔对应的尺度因子
	vec_scale_factor.resize(levels_num);
	vec_scale_factor[0] = 1.0f;
	for (int i = 1; i < levels_num; i++)
	{
		vec_scale_factor[i] = vec_scale_factor[i-1] * scale_factor;
	}

	std::vector<cv::Mat> vec_image_pyramid(levels_num); // 图像金字塔容器

	for (int level = 0; level < levels_num; level++)
	{
		float scale = 1.0f / vec_scale_factor[level];
		cv::size sz(cvRound((float)img.cols * scale), 
			cvRound((float)img.rows * scale));

		if (level == 0)
		{
			vec_image_pyramid[level] = img;
		}
		else
		{
			// 双线性插值新图像
			resize(vec_image_pyramid[level-1], vec_image_pyramid[level], 
				sz, 0, 0, CV_INTER_LINEAR);
		}

		// 金字塔构建过程可视化
		cout << "正在构建第" << level+1 << "层图像金字塔" << endl;
		imshow("img", vec_image_pyramid[level]);
		waitKey(100);
	}

	cout << "*******************************************" << endl;

	/**************** 四叉树划分特征点 *****************/

	// 搜索每一层图像上的特征点
	std::vector<std::vector<cv::KeyPoint>> all_keypoints; // 所有图层图像上特征点容器
	all_keypoints.resize(levels_num);

	const float border_width = 30;	// 设置栅格子大小

	for (int level = 0; level < levels_num; level++)
	{
		// 得到每一层图像进行特征检测区域上下两坐标
		const int min_border_x = EDGE_THRESHOLD - 3; // 边缘阈值滤掉
		const int min_border_y = min_border_x;
		const int max_border_x = vec_image_pyramid[level].cols - EDGE_THRESHOLD + 3;
		const int max_border_y = vec_image_pyramid[level].rows - EDGE_THRESHOLD + 3;

		// 每层待分配关键点数量
		std::vector<cv::KeyPoint> vec_to_distribute_keys;
		vec_to_distribute_keys.reserve(features_num * 10);
		// 计算总面积长宽
		const float width = max_border_x - min_border_x;
		const float height = max_border_y - min_border_y;
		// 记录划分完后格子行和列
		const int cols = width / border_width;
		const int rows = height / border_width;	
		// 重新计算格子大小
		const int width_cell = ceil(width / cols);
		const int height_cell = ceil(height / rows);
		cout << "第" << level+1 << "层图像切割成" << rows << "行" << cols "列，";

		// 开始对每个格子进行检测
		for (int i = 0; i < rows; i++)
		{
			const float ini_y = min_border_y + i * height_cell; // 格子高度下坐标
			float max_y = ini_y + height_cell + 6; // 格子高度上坐标

			if (ini_y >= max_border_y - 3)
			{
				continue;
			}

			if (max_y >= max_border_y)
			{
				max_y = max_border_y;
			}

			for (int j = 0; j < cols; j++)
			{
				const float ini_x = min_border_x + j * width_cell;
				float max_x = ini_x + width_cell + 6;

				if (ini_x >= max_border_x - 6)  // 一般认为相片宽度比高度大
				{
					continue;
				}

				if (max_x > max_border_x)
				{
					max_x = max_border_x;
				}

				std::vector<cv::KeyPoint> vec_keys_cell; // 用FAST特征检测并存储每个格子的特征点
				cv::FAST(vec_image_pyramid[level].rowRange(ini_x, max_y).colRange(ini_x, max_x), 
					vec_keys_cell, default_fast_threshold, true);

				// 如果fast检测空，降低阈值继续检测
				if (vec_keys_cell.empty())
				{
					cv::FAST(vec_image_pyramid[level].rowRange(ini_x, max_y).colRange(ini_x, max_x), 
					vec_keys_cell, min_fast_threshold, true);

				}
				// 计算特征点位置
				if (!vec_keys_cell.empty())
				{
					// 迭代法遍历每个格子中特征点容器的特征点
					for (std::vector<cv::KeyPoint>::iterator vit = vec_keys_cell.begin();
						vit != vec_keys_cell.end(); vit++) 
					{
						// 记录特征点在图像中的绝对坐标
						(*vit).pt.x += j * width_cell;
						(*vit).pt.y += i * height_cell;
						vec_to_distribute_keys.push_back(*vit);
					}
				}
			}
		}

		cout < "这层图像共有" << vec_to_distribute_keys.size() << "个特征点" << endl;

		std::vector<cv::KeyPoint> &keypoints = all_keypoints[level];
		keypoints.reserve(features_num);

		/*************开始四叉树划分*************/
		// 初始化几个节点，不难发现，由于长和宽比较接近，所以一般初始节点为1
		const int init_node_num = round(static_cast<float>(max_border_x - min_border_x) / 
			/ (max_border_y - min_border_y));
		cout << "初始时有" << init_node_num << "个节点";
		// 节点间间隔
		const float interval_x = static_cast<float>(max_border_x - min_border_x) / init_node_num;
		cout << "节点间间隔为" << interval_x << ", ";
		
		/**************四叉树设计****************/
		// 定义节点类型的初始化节点容器
		std::vector<extractor_node*> init_nodes;
		init_nodes.resize(init_node_num);
		// 划分之后的节点列表
		std::list<extractor_node> list_node;
		// 处理初始节点
		for (int i = 0; i < init_node_num; i++)
		{
			// 定义四叉树节点变量ni
			extractor_node ni;
			ni.UL = cv::Point2i(interval_x * static_cast<float>(i), 0);
			ni.UR = cv::Point2i(interval_x * static_cast<float>(i+1), 0);
			ni.BL = cv::Point2i(ni.UL.x, max_border_y-min_border_y);
			ni.BR = cv::Point2i(ni.UR.x, max_border_y-min_border_y);
			ni.vkeys.reserve(vec_to_distribute_keys.size());

			list_node.push_back(ni);
			init_nodes[i] = &list_node.back(); // 返回list_node最后的元素值
		}

		// 将点分配给初始节点
		for (size_t i = 0; i < vec_to_distribute_keys.size(); i++)
		{
			const cv::KeyPoint &kp = vec_to_distribute_keys[i];
			init_nodes[kp.pt.x/interval_x]->vkeys.push_back(kp);
		}

		// 设计节点迭代器
		std::list<extractor_node>::iterator lit = list_node.begin();
		// 遍历节点列表
		while (lit != list_node.end())
		{
			// 只含有一个特征点，就不再划分
			if (lit->vkeys.size() == 1)
			{
				lit->b_no_more = true;
				lit++;
			}
			else if (lit->vkeys.empty())
			{
				lit = list_nodes.erase(lit); // 如果这个节点没有特征点就删了
			}
			else
			{
				lit++;
			}
		}

		// 完结标志定义
		bool is_finish = false;
		// 迭代次数
		int iteration = 0;

		// 定义新数据类型节点及其所包含的特征数
		std::vector<std::pair<int, extractor_node*>> keys_size_and_node;
		keys_size_and_node.resize(list_nodes.size() * 4);

		while (!is_finish)
		{
			iteration++;
			// 初始化节点个数，用于判断节点是否再次进行了划分
			int pre_size = list_nodes.size();

			lit = list_nodes.begin();
			// 定义节点分解次数
			int to_expand_num = 0;
			keys_size_and_node.clear();

			while (lit != list_nodes.end())
			{
				if (lit->b_no_more)
				{
					lit++;
					continue;
				}
				else
				{
					// 超过一个特征点就继续划分
					extractor_node n1, n2, n3, n4;
					lit->divide_node(n1, n2, n3, n4);

					// 对划分后的节点进行判断，判断是否含有特征点，含有特征点则添加特征点
					if (n1.vkeys.size() > 0)
					{
						list_node.push_front(n1);
						if (n1.vkeys.size() > 1)
						{
							to_expand_num++;
							keys_size_and_node.push_back(
								std::make_pair(n1.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}
					
					if (n2.vkeys.size() > 0)
					{
						list_node.push_front(n2);
						if (n2.vkeys.size() > 1)
						{
							to_expand_num++;
							keys_size_and_node.push_back(
								std::make_pair(n2.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}

					if (n3.vkeys.size() > 0)
					{
						list_node.push_front(n3);
						if (n3.vkeys.size() > 1)
						{
							to_expand_num++;
							keys_size_and_node.push_back(
								std::make_pair(n3.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}
					
					if (n4.vkeys.size() > 0)
					{
						list_node.push_front(n4);
						if (n4.vkeys.size() > 1)
						{
							to_expand_num++;
							keys_size_and_node.push_back(
								std::make_pair(n4.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}
					
					lit = list_nodes.erase(lit);
					continue;
				}
			}

			// 给每层分配特征点数先估计
			features_num_per_level.resize(levels_num);
			float factor = 1.0f / scale_factor;
			// 构造等比数列
			float desired_features_per_scale = features_num * (1 - factor)
				/ (1 - (float)pow((double)factor, (double)levels_num));
			int sum_features = 0;
			for (int level = 0; level < levels_num; level++)
			{
				features_num_per_level[level] = cvRound(desired_features_per_scale);
				sum_features += features_num_per_level[level];
				desired_features_per_scale *= factor;
			}
			features_num_per_level[levels_num-1] = std::max(features_num-sum_features, 0);

			// 当节点个数大于需分配的特征数或者所有的节点只有一个特征点（节点不能划分）的时候，则结束。
			if ((int)list_nodes.size() >= features_num_per_level[level]
				|| (int)list_nodes.size() == pre_size)
			{
				is_finish = true;
			}
			else if (((int)list_nodes.size() + to_expand_num * 3) > features_num_per_level[level])
			{
				// 节点展开次数乘以3用于表明下一次的节点分解可能超过特征数，即为最后一次分解
				while (!is_finish)
				{
					pre_size = list_nodes.size();

					std::vector<std::pair<int, extractor_node*>> pre_size_and_node = keys_size_and_node;
					keys_size_and_node.clear();

					sort(pre_size_and_node.begin(), pre_size_and_node.end());
					for (int j = pre_size_and_node.size()-1; j >= 0; j--)
					{
						extractor_node n1, n2, n3, n4;
						pre_size_and_node[j].second->divide_node(n1, n2, n3, n4);

						// 划分之后进一步判断
						if (n1.vkeys.size() > 0)
						{
							list_nodes.push_front(n1);
							if (n1.vkeys.size() > 1)
							{
								keys_size_and_node.push_back(std::make_pair(n1.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}

						if (n2.vkeys.size() > 0)
						{
							list_nodes.push_front(n2);
							if (n2.vkeys.size() > 1)
							{
								keys_size_and_node.push_back(std::make_pair(n2.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}

						if (n3.vkeys.size() > 0)
						{
							list_nodes.push_front(n3);
							if (n3.vkeys.size() > 1)
							{
								keys_size_and_node.push_back(std::make_pair(n3.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}

						if (n4.vkeys.size() > 0)
						{
							list_nodes.push_front(n4);
							if (n4.vkeys.size() > 1)
							{
								keys_size_and_node.push_back(std::make_pair(n4.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}

						list_nodes.erase(pre_size_and_node[j].second->lit);
						if ((int)list_nodes.size() >= features_num_per_level[level])
						{
							break;
						}
					}

					if ((int)list_nodes.size() >= features_num_per_level[level]
						|| (int)list_nodes.size() == pre_size)
					{
						is_finish = true;
					}
				}
			}
		}

		// 用KeyPoints数据类型.response进行挑选保留每个节点下最好的特征点
		std::vector<cv::KeyPoint> result_keys;
		result_keys.reserve(features_num_per_level[level]);
		for (std::list<extractor_node>::iterator lit = list_nodes.begin(); lit = list_nodes.end(); lit++)
		{
			std::vector<cv::KeyPoint> &node_keys = lit->vkeys;
			cv::KeyPoint* keypoint = &node_keys[0];
			float max_response = keypoint->response;

			for (size_t k = 1; k < node_keys.size(); k++)
			{
				if (node_keys[k].response > max_response)
				{
					keypoint = &node_keys[k];
					max_response = node_keys[k].response;
				}
			}
			
			result_keys.push_back(*keypoint);
		}

		keypoints = result_keys;

		const int scaled_path_size = PATCH_SIZE * vec_scale_factor[level];
		// 换算特征点的真实值
		const int nkps = keypoints.size();
		for (int i = 0; i < nkps; i++)
		{
			keypoints[i].pt.x += min_border_x;
			keypoints[i].pt.y += min_border_y;
			keypoints[i].octave = level;
			keypoints[i].size = scaled_path_size;
		}

		cout << "经过四叉树筛选，第" << level+1 << "层剩余" << result_keys.size() << "个特征点" << endl;
	}
	
	cout << "****************************************" << endl;

	/******************计算特征点的方向，尽管本题不涉及，但是为了大家方便理解ORB_SLAM2还是写一下********/

	// 准备数学工具，在特征计算时，每个v坐标对应的最大u坐标
	vector<int> u_max;	// 在半径等于HALF_SIZE的范围内，v=x下，u所能取到的最大坐标的绝对值
	u_max.resize(HALF_PATH_SIZE + 1);
	// 将v坐标分两部分计算，xy方向比较对称
	int v, v0, vmax = cvFloor(HALF_PATH_SIZE * sqrt(2.f) / 2 + 1);
	int vmin = cvCeil(HALF_PATH_SIZE * sqrt(2.f) / 2);
	// 勾股定理
	const double hp2 = HALF_PATH_SIZE * HALF_PATH_SIZE;
	for (v = 0; v <= vmax; v++)
	{
		u_max[v] = cvRound(sqrt(hp2 - v * v));
	}
	// 确保对称是个圆
	for (v = HALF_PATH_SIZE, v0 = 0; v >= vmin; --v)
	{
		while (u_max[v0] == u_max[v0+1])
		{
			++v0;
		}
		u_max[v] = v0;
		++v0;
	}

	for (int level = 0; level <levels_num; level++)
	{
		// 指针提取该层特征点
		vector<KeyPoint>& keypoints = all_keypoints[level];

		// 逐个特征点计算
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			// 计算特征点这个区域内所有像素与其x坐标的乘积以及所有像素与其对应y坐标的乘积
			int m_01 = 0, m_10 = 0;
			// 得到每个特征点的中心位置
			const uchar* center = &vec_image_pyramid[level].at<uchar>(cvRound(keypoints[i].pt.y),
				cvRound(keypoints[i].pt.y));

			// 当v=0时单独计算
			for (int u = -HALF_PATH_SIZE；u < HALF_PATH_SIZE; u++)
			{
				m_10 += u * center[u];
			}

			int step = (int)vec_image_pyramid[level].step1();

			for (int v = 1; v <= HALF_PATH_SIZE; v++)
			{
				int v_sum = 0;
				int d = u_max[v];
				for (int u = -d; u <=d; u++)
				{
					int val_plus = center[u+v*step];
					int val_minus = center[u-v*step];
					v_sum += val_plus - val_minus;
					m_10 += u * (val_plus + val_minus);
				}
				m_01 += v * v_sum;
			}

			keypoints[i].angle = cv::fastAtan2((float)m_01, (float)m_01, (float)m_10);
			// cout << "检测到第" << level << "层，第" << i << "个特征点角度为" << keypoints[i].angle << "度" << endl;
		}
	}

	/*********设计描述子**********/
	// 拷贝模板
	// bit_pattern_31存储的是通过学习后选取的256个点对相交于特征点的位置，
	// 其中四个元素分别表示pair_l.x, pair_l.y, pair_r.x, pair_r.y,
	// 共有256个点对，产生256维描述子，每个描述子长32个字节，可以imshow查看描述子的样子
	vector<Point> pattern;
	const int num_points = 512;
	const Point* pattern0 = (const Point*)bit_pattern_31;
	// 将从pattern0到pattern0+512序列（也就是前面定义一串数字）拷贝到pattern中
	std::copy(pattern0, pattern0+num_points, std::back_inserter(pattern));

	Mat descriptors;
	int num_keypoints = 0;
	for (int level = 0; level < levles_num; level++)
	{
		num_keypoints += (int)all_keypoints[level].size(); // 统计所有特征点数总和
	}
	cout << "共有" << num_keypoints << "个特征点，现在开始添加描述子..." << endl;

	// 设计总输出特征点容器
	vector<KeyPoint> out_put_keypoints(num_keypoints);

	if (num_keypoints != 0)
	{
		descriptors.create(num_keypoints, 32, CV_8U);
	}

	int offset = 0;
	for (int level = 0; level <levels_num; level)
	{
		vector<KeyPoint>& keypoints = all_keypoints[level];
		int num_keypoints_level = (int)keypoints[level].size();
		if (num_keypoints_level == 0)
		{
			continue;
		}

		// 高斯模糊是为了计算BRIEF时去噪
		cout << "开始将原图层高斯模糊，正在模糊第" << level+1 << "张图..." << endl;
		Mat working_mat = vec_image_pyramid[level].clone();
		GaussianBlur(working_mat, working_mat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
		imshow("img", working_mat);
		waitKey(100);

		// 计算每一层的描述子
		Mat_descriptors_per_level = descriptors.rowRange(offset, offset+num_keypoints_level);
		descriptors_per_level = Mat::zero((int)keypoints.size(), 32, CV_8UC1); // 每一层的描述子

		const float factorPI = (float)(CV_PI / 180.f);
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			uchar* desc = decriptors_per_level.ptr((int)i);
			Point* ppattern = &pattern[0];

			float angle = (float)keypoints[i].angle * factorPI; // 转化为弧度
			float a = (float)cos(angle), b = (float) sin(angle);

			const uchar* center = &working_mat.at<uchar>(cvRound(keypoints[i].pt.y),
				cvRound(keypoints[i].pt.x));
			const int step = (int)working_mat.step;

			// 取旋转后一个像素点的值
			#define GET_VALUE(idx) \ 
				center[ cvRound(ppattern[idx].x*b + ppattern[idx].y*a) * step + \
					cvRound(ppattern[idx].x*a - ppattern[idx].y*b)]

			// 循环32次，pattern取值16*32=512,也就是说每次取16个点，形成8个点对，8个点对比较可以形成8bit长度的特征描述数据
			for (int i = 0; i < 32; i++, ppattern += 16)
			{
				int t0, t1, val;
				
				t0 = GET_VALUE(0);
				t1 = GET_VALUE(1);
				val = t0 < t1;

				t0 = GET_VALUE(2);
				t1 = GET_VALUE(3);
				val |= (t0 < t1) << 1;

				t0 = GET_VALUE(4);
				t1 = GET_VALUE(5);
				val |= (t0 < t1) << 2;

				t0 = GET_VALUE(6);
				t1 = GET_VALUE(7);
				val |= (t0 < t1) << 3;

				t0 = GET_VALUE(8);
				t1 = GET_VALUE(9);
				val |= (t0 < t1) << 4;

				t0 = GET_VALUE(10);
				t1 = GET_VALUE(11);
				val |= (t0 < t1) << 5;
				
				t0 = GET_VALUE(12);
				t1 = GET_VALUE(13);
				val |= (t0 < t1) << 6;

				t0 = GET_VALUE(14);
				t1 = GET_VALUE(15);
				val |= (t0 < t1) << 7;
				
				desc[i] = (uchar)val; // 一共32*8组描述子
			}

			#undef GET_VALUE
		}

		offset += num_keypoints_level;

		// 对关键点进行尺度恢复，恢复到原图位置
		if (level != 0)
		{
			float scale = vec_scale_factor[level];
			for (vector<KeyPoint>::iterator keypoint = keypoints.begin(); 
				keypoint != keypoints.end(); keypoint++)
			{
				keypoint->pt *= scale;
			}
		}

		out_put_keypoints.insert(out_put_keypoints.end(), keypoints.begin(), keypoints.end());
	}

	destroyAllWindows();

	Mat out_img1;
	drawKeypoints(img, out_put_keypoints, out_img1, Scalar::all(-1), DrawMatchedFlags::DEFAULT);
	imshow("四叉树法ORB", out_img1);
	imwrite("NewORB.png", out_img1);
	waitKey(0);

	vector<KeyPoint> orb_keypoints;
	Ptr<ORB> orb = ORB::create(1000);
	orb->detect(img, orb_keypoints);
	cout << "共找到了" << orb_keypoints.size() << "个特征点;" << endl;

	Mat img_orb;
	drawKeypoints(img, orb_keypoints, img_orb, Scalar::all(-1), DrawMatchedFlags::DEFAULT);
	imshow("普通ORB算法", img_orb);
	imwrite("NormalORB.png", img_orb);
	waitKey(0);

	return 0;
}


#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;

class CLidarAC
{
public:
	CLidarAC();
	virtual ~CLidarAC();
	void SetRanges(float* inData);
	bool OutLine();
	int GetMinIndex(int inBeginIndex, int inEndIndex);
	vector<int> arBlobIndex;

	float arFrontRanges[180];
	float arPnt_x[180];
	float arPnt_y[180];
	bool arACPnt[180];
	float nACWidth;
	float nACDist;
	float nIgnDist;
	float nBlobDist;
	int nLeftIndex;
	int nRightIndex;
	float kx;
	float ky;
	float fVx;
	float fVy;
};
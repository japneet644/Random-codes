#include <iostream>
#define  Size 20
#define Length 8
typedef struct Point
{
    int x,y;
}point;
class Enviroment
{
public:
    int *** grid = new int**[22];
    point ** obst = new point*[6];
    point state;
    int theta;
    for (int i = 0; i < 4; i++) {point[i] = new point[4]; grid[i] = new int*[Size];}
    for (int i = 4; i <22 ; i++)
    /*

    // */
    // for (int i = 0; i < Size; i++)
    // {
    //     grid[i] = new int[Size];
    //     for (int j = 0; j < Size; j++) {
    //         grid[i][j] = 0;
    //     }
    // }
    float***Q;
    bool Isoverlap(float a,float b)
    {
        for (int i = 0; i < 6; i++)
        {
            if( ((b-obst[i][0].y)(obst[i][1].x - obst[i][0].x) - (a-obst[i][0].x)(obst[i][1].y - obst[i][0].y))*((obst[i][2].y-obst[i][0].y)(obst[i][1].x - obst[i][0].x) - (obst[i][2].x-obst[i][0].x)(obst[i][1].y - obst[i][0].y)) >0)
            {
            if( ((b-obst[i][1].y)(obst[i][2].x - obst[i][1].x) - (a-obst[i][1].x)(obst[i][2].y - obst[i][1].y))*((obst[i][3].y-obst[i][1].y)(obst[i][2].x - obst[i][1].x) - (obst[i][3].x-obst[i][1].x)(obst[i][2].y - obst[i][1].y)) >0 )
            {
            if( ((b-obst[i][2].y)(obst[i][3].x - obst[i][2].x) - (a-obst[i][2].x)(obst[i][3].y - obst[i][2].y))*((obst[i][0].y-obst[i][2].y)(obst[i][3].x - obst[i][2].x) - (obst[i][0].x-obst[i][2].x)(obst[i][3].y - obst[i][2].y)) >0 )
            {
            if( ((b-obst[i][3].y)(obst[i][0].x - obst[i][3].x) - (a-obst[i][3].x)(obst[i][0].y - obst[i][3].y))*((obst[i][1].y-obst[i][3].y)(obst[i][0].x - obst[i][3].x) - (obst[i][1].x-obst[i][3].x)(obst[i][0].y - obst[i][3].y)) >0  )
            return 1;
            }
            }
            }
        }
    }
};
int main(int argc, char const *argv[]) {


    return 0;
}

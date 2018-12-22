#include"iostream"
#include <stdlib.h>
float N0 = 100.0;
int step( int &ps,int &ds, int action)
{
    if(action)
    {
        int x = rand()%3;
        if(x==0) ps-= ((rand()%10) + 1);
        else ps += ((rand()%10) + 1);
        if(ps<1 || ps>21) return -1;
    }
    return 0;
}
float max(float a,float b)
{
    if(a>b) return a;
    return b;
}

double doubleRand()
{
    return double(rand()) / (double(RAND_MAX) + 1.0);
}
void generate_episold(float ***Q,int **q,int **f)
{
    int ps,ds,count = 0;
    ps = (rand()%10) + 1;
    ds = (rand()%10) + 1;
    int r=0;
    // std::cout << ps<<' '<< ds;
    while(r==0)
    {
        f[ps-1][ds-1]++;    q[0][count] = ps; q[1][count] = ds;
        float u = doubleRand();
        if(u<N0/(N0+f[ps-1][ds-1]))    q[2][count++] = rand()%2;
        else
        {
            if(Q[1][ps-1][ds-1] > Q[0][ps-1][ds-1]) q[2][count++] = 1;
            else q[2][count++] = 0;
        }
        if(q[2][count-1]==0) break;
        r = step(ps,ds,q[2][count-1]);
    }
    // f[ps-1][ds-1]++;    q[0][count] = ps; q[1][count] = ds;  q[2][count++] = 0;
    while(r==0 && ds<17 && ps>ds)
    {
        // f[ps-1][ds-1]++;    q[0][count] = ps; q[1][count] = ds; q[2][count++] = 0;
        int x = rand()%3;
        if(x==0) ds -= ((rand()%10) + 1);
        else ds += ((rand()%10) + 1);
        if(ds<1 || ds>21) r = 1;
        if(ds>ps) r=-1;
    }
    if(ps>ds && r==0)
    {
        r = 1;
    }
    else if(ps==ds) r = 0;
    // else if(r==0)r = -1;
    // std::cout <<' '<<ps<<' '<< ds<<' '<< r <<'c'<<count<<'\n';
}

int main(int argc, char const *argv[]) {
    // bool * action = new bool[100];
    float ***Q = new float**[2];
    Q[0] = new float*[21]; Q[1] = new float*[21];
    int **f = new int*[21];
    int **q = new int*[3];
    q[0] = new int[100]; q[1] = new int[100]; q[2] = new int[100];
    for (int i = 0; i < 21; i++)
    {
        f[i] = new int[21];
        Q[0][i] = new float[10];
        Q[1][i] = new float[10];
        for (int j = 0; j < 21; j++)
        {
            f[i][j] = 0;
            if(j<10&& i<12){ Q[1][i][j] = 0.0;     Q[0][i][j] = 0.0;}
        }
    }
    for (int x = 0; x <= 960000; x++)
    {
        generate_episold(Q,q,f);
    }
    for(int i=0;i<21;i++)
    {
        for(int j=0;j<10;j++)
        {
            // std::cout << max(Q[0][i][j],Q[1][i][j]) << ' ';
            if(Q[0][i][j]>Q[1][i][j]) std::cout << '0' << ' ';
            else std::cout << "1" << ' ';
        }
        std::cout << '\n';
    }
    //
    // f[ps]++;    q[0][count] = ps; q[1][count++] = ds;
    return 0;
}

#include"iostream"
#include"math.h"
#define Goal 100
float ph=0.45;

float max(float * Q,int n, int * P,int u)
{
    float m = -100;
    for (int i = 0; i <n ; i++) {
        if(Q[i]>m) { m = Q[i]; P[u] = i+1;}
    }
    return m;
}
int min(int a,int b)
{
    if(a<b) return a;
    return b;
}
int main() {
    float * state_value = new float[Goal+1];
    int * policy = new int[Goal];
    for(int i=0;i<Goal;i++)  state_value[i] = 0.0;
    state_value[Goal] = 1.0;
    float theta = 1.0;
    int count =0;
    while( count <1000)
    {
        theta = 0;
        count++;
        for (int i = 1; i < Goal; i++)
        {
            float * Q = new float[min(i,Goal-i)+1];
            float x;
            for(int j= 0;j<min(i,Goal-i);j++)
            {
                Q[j] = ph*state_value[i+j+1] + (1.0 -ph)*state_value[i-j-1];
            }
            x = max(Q,min(i,Goal-i),policy,i);
            theta = (theta > fabs(state_value[i] - x)?  theta:fabs(state_value[i] - x));
            state_value[i] = x;
        }
    }
    std::cout << count << ' '<<theta<< '\n';
    for (int i = 0; i < Goal; i++)
    {
        std::cout << policy[i] << ' ';
    }
    std::cout << '\n';
    int x,y;
    for(y=100;y>=0;y--)
    {
       for(x=0;x<100;x++)
       {
           if(y==policy[x]) std::cout << '*';
           else  std::cout << ' ';
       }
       std::cout << '\n';
    }
    // for (int i = 0; i <= Goal; i++)
    // {
    //     std::cout << state_value[i] << ' ';
    // }
    // std::cout << '\n';
    return 0;
}
